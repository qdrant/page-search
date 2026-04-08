import asyncio
import concurrent.futures
import hashlib
import uuid
from itertools import islice
from typing import Callable
from urllib.parse import urljoin

import requests
import tqdm
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from openai import AsyncOpenAI, DefaultAioHttpClient
from openai.types.responses import Response
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    Distance,
    Document,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from qdrant_client.models import PayloadSchemaType

from site_search.config import (
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SNIPPET_COLLECTION_NAME,
    SNIPPET_ENCODER,
)
from site_search.sections import _all_sitemap_urls


class TooManyRetriesError(Exception):
    pass


def retry(
    fn: Callable,
    max_retries: int | None,
    wait: int = 1,
) -> Callable:
    def inner(*args, **kwargs):
        num_tries = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if max_retries is not None and num_tries >= max_retries:
                    raise TooManyRetriesError

                if isinstance(e, ResponseHandlingException):
                    logger.warning(
                        f"{repr(fn)} failed with {repr(e)}, retrying after {wait}"
                    )
                    num_tries += 1
                    # await asyncio.sleep(wait)
                    continue
                else:
                    raise

    return inner


# python >= 3.12 has this builtin
def batched(iterable, n, *, strict=False):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


PROMPT = """
You are creating a searchable description for a code snippet. The description will be used for vector search by both humans and AI agents.

Context before the snippet:
```
{context_before}
```

Code snippet:
```
{code}
```

Context after the snippet:
```
{context_after}
```

Write a concise, keyword-rich description (2-4 sentences) that includes:
1. What the code does (primary functionality and purpose)
2. Key technical concepts, methods, classes, or libraries used
3. The use case or problem it solves
4. Any important parameters, return values, or side effects

Focus on searchable terms that developers would use when looking for this functionality. Be specific and technical.
"""


class SourceInfo(BaseModel):
    url: str
    hash: str
    lines: tuple[int, int] | None = None


class SnippetContext(BaseModel):
    before: str
    after: str


class Snippet(BaseModel):
    code: str
    language: str
    version: str
    revision: int
    package_name: str
    source: SourceInfo
    context: SnippetContext
    description: str | None = None

    async def generate_description(self, client: AsyncOpenAI) -> None:
        response: Response = await client.responses.create(
            model="gpt-5-nano",
            input=PROMPT.format(
                context_before=self.context.before,
                context_after=self.context.after,
                code=self.code,
            ),
            truncation="auto",
        )

        if response is None:
            logger.error(f"OpenAI request failed for {self.source.url}")
            raise ConnectionError

        self.description = response.output_text

    @property
    def document(self) -> str:
        if self.description is not None:
            return self.description
        return f"{self.context.before}\n```{self.language}\n{self.code}\n```\n{self.context.after}"

    @property
    def metadata(self) -> dict[str, str]:
        metadata = self.dict()
        metadata["description"] = self.document
        return metadata

    @property
    def uuid(self) -> str:
        content = str(
            self.dict(
                include={
                    "code": True,
                    "package_name": True,
                    "source": {"url"},
                    "context": True,
                }
            )
        )
        # Create a SHA-256 hash of the content
        content_hash = hashlib.sha256(content.encode("utf-8")).digest()
        # Use the first 16 bytes of the hash to create a UUID
        return str(uuid.UUID(bytes=content_hash[:16]))

    def as_point(self, model: str) -> PointStruct:
        return PointStruct(
            id=self.uuid,
            payload=self.metadata,
            vector=Document(text=self.document, model=model),
        )


class _ParsingResult(BaseModel):
    url: str
    snippets: list[Snippet]


_language_map = {
    "bash": "shell",
    "py": "python",
    "console": "shell",
    "env": "shell",
    "sh": "shell",
    "jsx": "javascript",
    "http request": "http",
    "js": "javascript",
    "txt": "text",
}


def _normalize_language(language: str) -> str:
    language = language.lower().strip()
    return _language_map.get(language, language)


def _format_context(
    node: SyntaxTreeNode, context: str, offset: int = 10
) -> SnippetContext:
    assert node.map
    start = node.map[0] + 1
    end = node.map[1] - 1
    lines = context.split("\n")

    # skip preceding and following code blocks for context
    # NOTE: should we only skip if language doesn't match?
    prev = node
    while (prev := prev.previous_sibling) and prev.type in ["fence", "code_block"]:
        continue

    if prev is not None:
        assert prev.map
        start = prev.map[1]

    next = node
    while (next := next.next_sibling) and next.type in ["fence", "code_block"]:
        continue

    if next is not None:
        assert next.map
        end = next.map[0]

    return SnippetContext(
        before="\n".join(lines[max(0, start - offset) : start]),
        after="\n".join(lines[end : min(len(lines), end + offset)]),
    )


def _extract_from_markdown_tree(
    content: str, root: SyntaxTreeNode, source: str, source_hash: str
) -> list[Snippet]:
    snippets: list[Snippet] = []

    for node in root.children:
        # Code fence, optionally with language info
        if node.type == "fence":
            snippets.append(
                Snippet(
                    code=node.content,
                    language=_normalize_language(node.info),
                    package_name="qdrant-client",
                    source=SourceInfo(
                        url=source,
                        hash=source_hash,
                        lines=node.map,
                    ),
                    context=_format_context(node, content),
                    version="latest",
                    revision=1,
                )
            )
    return snippets


async def _generate_descriptions(snippets: list[Snippet]):
    async with AsyncOpenAI(
        api_key=OPENAI_API_KEY, http_client=DefaultAioHttpClient()
    ) as oai_client:
        tasks = [snippet.generate_description(oai_client) for snippet in snippets]
        for task in asyncio.as_completed(tasks):
            await task


def _parse_markdown(url: str) -> _ParsingResult:
    resp = requests.get(urljoin(url, "index.md"))
    if not resp.ok:
        return _ParsingResult(snippets=[], url=url)

    document = resp.text
    md_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()

    tokens = MarkdownIt("commonmark").parse(document)
    root = SyntaxTreeNode(tokens)
    snippets = _extract_from_markdown_tree(document, root, url, md_hash)
    asyncio.run(_generate_descriptions(snippets))
    return _ParsingResult(snippets=snippets, url=url)


def main():
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=int(QDRANT_PORT),
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
        timeout=30,
    )

    if qdrant_client.collection_exists(SNIPPET_COLLECTION_NAME):
        qdrant_client.delete_collection(SNIPPET_COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=SNIPPET_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=qdrant_client.get_embedding_size(SNIPPET_ENCODER),
            distance=Distance.COSINE,
        ),
    )

    qdrant_client.create_payload_index(
        collection_name=SNIPPET_COLLECTION_NAME,
        field_name="language",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    qdrant_client.create_payload_index(
        collection_name=SNIPPET_COLLECTION_NAME,
        field_name="source.url",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    qdrant_client.create_payload_index(
        collection_name=SNIPPET_COLLECTION_NAME,
        field_name="source.hash",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    urls = _all_sitemap_urls("https://qdrant.tech/", "https://qdrant.tech/sitemap.xml")

    snippets = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_parse_markdown, url) for url in urls]

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(urls)
        ):
            result = future.result()

            if len(result.snippets) == 0:
                continue

            snippets.extend(result.snippets)

    batches = list(batched(snippets, 8))
    for batch in tqdm.tqdm(batches, total=len(batches)):
        retry(qdrant_client.upsert, max_retries=10)(
            SNIPPET_COLLECTION_NAME,
            points=[snippet.as_point(SNIPPET_ENCODER) for snippet in batch],
        )


if __name__ == "__main__":
    main()

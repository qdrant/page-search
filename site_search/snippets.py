import asyncio
import concurrent.futures
import hashlib
import uuid
from urllib.parse import urljoin

import requests
import tqdm
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from openai import AsyncOpenAI, DefaultAioHttpClient
from openai.types.responses import Response
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
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
    NEURAL_ENCODER,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SNIPPET_COLLECTION_NAME,
)
from site_search.sections import _all_sitemap_urls

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
                    language=node.info,
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
        # Indented code block
        elif node.type == "code_block":
            logger.info(f"Code block in {source}")
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
        prefer_grpc=True,
        local_inference_batch_size=32,
    )

    if qdrant_client.collection_exists(SNIPPET_COLLECTION_NAME):
        qdrant_client.delete_collection(SNIPPET_COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=SNIPPET_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=qdrant_client.get_embedding_size(NEURAL_ENCODER),
            distance=Distance.COSINE,
        ),
    )

    qdrant_client.create_payload_index(
        collection_name=SNIPPET_COLLECTION_NAME,
        field_name="context.after",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            min_token_len=1,
            max_token_len=20,
            lowercase=True,
        ),
        wait=True,
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_parse_markdown, url) for url in urls]

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(urls)
        ):
            result = future.result()

            if len(result.snippets) == 0:
                continue

            qdrant_client.upsert(
                SNIPPET_COLLECTION_NAME,
                points=[
                    snippet.as_point(NEURAL_ENCODER) for snippet in result.snippets
                ],
            )


if __name__ == "__main__":
    main()

import asyncio
import concurrent.futures
import hashlib
import time
import uuid
from collections.abc import Coroutine
from itertools import islice
from typing import Any
from urllib.parse import urljoin
from uuid import UUID

import requests
import tqdm
from loguru import logger
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from openai import AsyncOpenAI, DefaultAioHttpClient
from openai.types.responses import Response
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Document,
    Filter,
    HasIdCondition,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)
from qdrant_client.models import PayloadSchemaType, VectorStruct

from site_search.config import (
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SNIPPET_COLLECTION_NAME,
    SNIPPET_ENCODER,
)
from site_search.retry import retry, retry_async
from site_search.sections import _all_sitemap_urls


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
    revision: int | None = None
    package_name: str
    source: SourceInfo
    context: SnippetContext
    description: str | None = None

    async def generate_description(self, client: AsyncOpenAI) -> None:
        response: Response | None = await retry_async(client.responses.create, 5)(
            model="gpt-5-nano",
            input=PROMPT.format(
                context_before=self.context.before,
                context_after=self.context.after,
                code=self.code,
            ),
            truncation="auto",
        )  # ty:ignore[invalid-assignment]

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

    def as_point(self, model: str, vector: VectorStruct | None = None) -> PointStruct:
        return PointStruct(
            id=self.uuid,
            payload=self.metadata,
            vector=vector
            if vector is not None
            else {
                "dense": Document(text=self.document, model=model),
                "sparse": Document(
                    text=self.code,
                    model="qdrant/bm25",
                    options={"language": "none"},
                ),
            },
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
                )
            )
    return snippets


async def _generate_descriptions(
    snippets: list[Snippet], existing: dict[int | str | UUID, str | None]
):
    async with AsyncOpenAI(
        api_key=OPENAI_API_KEY, http_client=DefaultAioHttpClient()
    ) as oai_client:
        tasks: list[Coroutine[Any, Any, None]] = []
        for snippet in snippets:
            snippet.description = existing.get(snippet.uuid)
            if snippet.description is None:
                tasks.append(snippet.generate_description(oai_client))
        for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await task
        return len(tasks)


def _parse_markdown(url: str) -> _ParsingResult:
    resp = retry(requests.get, 10)(urljoin(url, "index.md"))
    if not resp.ok:
        return _ParsingResult(snippets=[], url=url)

    document = resp.text
    md_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()

    tokens = MarkdownIt("commonmark").parse(document)
    root = SyntaxTreeNode(tokens)
    snippets = _extract_from_markdown_tree(document, root, url, md_hash)
    return _ParsingResult(snippets=snippets, url=url)


def main():
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=int(QDRANT_PORT),
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
        timeout=30,
    )

    if not retry(qdrant_client.collection_exists, 10)(SNIPPET_COLLECTION_NAME):
        retry(qdrant_client.create_collection, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=qdrant_client.get_embedding_size(SNIPPET_ENCODER),
                distance=Distance.COSINE,
            ),
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="code",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=1,
                max_token_len=20,
                lowercase=True,
            ),
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="context.before",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=1,
                max_token_len=20,
                lowercase=True,
            ),
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
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

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="description",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=1,
                max_token_len=20,
                lowercase=True,
            ),
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="language",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="version",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="revision",
            field_schema=PayloadSchemaType.INTEGER,
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="package_name",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="source.url",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        retry(qdrant_client.create_payload_index, 10)(
            collection_name=SNIPPET_COLLECTION_NAME,
            field_name="source.hash",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    run_ts = int(time.time())
    urls = retry(_all_sitemap_urls, 10)(
        "https://qdrant.tech/", "https://qdrant.tech/sitemap.xml"
    )

    snippets: list[Snippet] = []
    current_uuids: set[str] = set()
    existing_uuids: set[str] = set()
    existing_descriptions: dict[str | int | UUID, str | None] = {}
    failed_urls: list[str] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_parse_markdown, url): url for url in urls}

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(urls)
        ):
            url = futures[future]
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Failed to parse {url}: {e}")
                failed_urls.append(url)
                continue

            if len(result.snippets) == 0:
                continue

            current_uuids.update(snippet.uuid for snippet in result.snippets)

            existing_points = retry(qdrant_client.retrieve, 10)(
                SNIPPET_COLLECTION_NAME,
                ids=[snippet.uuid for snippet in result.snippets],
                with_payload=True,
                with_vectors=False,
            )

            for point in existing_points:
                existing_uuids.add(str(point.id))
                existing_descriptions[point.id] = (point.payload or {}).get(
                    "description"
                )

            snippets.extend(result.snippets)

    new_snippets = [s for s in snippets if s.uuid not in existing_uuids]
    for snippet in new_snippets:
        snippet.revision = run_ts

    num_generated = asyncio.run(
        _generate_descriptions(new_snippets, existing_descriptions)
    )

    batches: list[list[Snippet]] = list(batched(new_snippets, 8))
    for batch in tqdm.tqdm(batches, total=len(batches)):
        retry(qdrant_client.upsert, max_retries=10)(
            SNIPPET_COLLECTION_NAME,
            points=[snippet.as_point(SNIPPET_ENCODER) for snippet in batch],
        )
    logger.info(
        f"Upserted {len(new_snippets)} new snippets, "
        f"generated {num_generated} descriptions"
    )

    if failed_urls:
        logger.warning(
            f"Skipping stale-point deletion: {len(failed_urls)} URL(s) failed "
            f"({failed_urls[:5]}{'...' if len(failed_urls) > 5 else ''}). "
            f"Re-run indexing to clean up stale snippets."
        )
        return
    if len(new_snippets) == 0:
        logger.warning(
            "Skipping stale-point deletion: found 0 snippets. "
            "Re-run indexing to clean up stale snippets."
        )
        return

    retry(qdrant_client.delete, 10)(
        SNIPPET_COLLECTION_NAME,
        points_selector=Filter(must_not=[HasIdCondition(has_id=list(current_uuids))]),
        wait=True,
    )
    logger.info(
        f"Deleted stale snippets not in current crawl of {len(current_uuids)} UUIDs"
    )


if __name__ == "__main__":
    main()

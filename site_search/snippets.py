import concurrent.futures
from itertools import accumulate
import hashlib
import uuid
from urllib.parse import urljoin, urlsplit

import requests
import tqdm
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from pydantic import BaseModel
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
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SNIPPET_COLLECTION_NAME,
)
from site_search.sections import _all_sitemap_urls


class SourceInfo(BaseModel):
    url: str
    hash: str
    lines: tuple[int, int] | None = None


class SnippetContext(BaseModel):
    before: str
    after: str


class PartialSnippet(BaseModel):
    code: str
    language: str
    source: SourceInfo
    context: SnippetContext

    @property
    def description(self) -> str:
        return f"{self.context.before}\n{self.code}\n{self.context.after}"

    @property
    def uuid(self) -> str:
        content = str(
            self.dict(
                include={
                    "code": True,
                    "source": {"url"},
                    "context": True,
                }
            )
        )
        # Create a SHA-256 hash of the content
        content_hash = hashlib.sha256(content.encode("utf-8")).digest()
        # Use the first 16 bytes of the hash to create a UUID
        return str(uuid.UUID(bytes=content_hash[:16]))


class Snippet(BaseModel):
    description: str
    code: str
    language: str
    version: str
    revision: int
    package_name: str
    source: SourceInfo
    context: SnippetContext

    @property
    def document(self) -> str:
        return self.description

    @property
    def metadata(self) -> dict[str, str]:
        return self.dict()

    @property
    def uuid(self) -> str:
        content = str(
            self.dict(
                include={
                    "code": True,
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
) -> list[PartialSnippet]:
    snippets: list[PartialSnippet] = []

    for node in root.children:
        # Code fence, optionally with language info
        if node.type == "fence":
            snippets.append(
                PartialSnippet(
                    code=node.content,
                    language=node.info,
                    source=SourceInfo(
                        url=source,
                        hash=source_hash,
                        lines=node.map,
                    ),
                    context=_format_context(node, content),
                )
            )
        # Indented code block
        elif node.type == "code_block":
            print(f"Code block on {source}")
    return snippets


def _fill_snippet(
    snippet: PartialSnippet, should_generate_description: bool = True
) -> Snippet:
    assert snippet.source.lines
    description = snippet.description
    # if should_generate_description:
    #     description = await generate_description(
    #         url=snippet.source.url,
    #         context=snippet.context,
    #         code=snippet.code,
    #         client=oai_client,
    #     )

    return Snippet(
        code=snippet.code,
        description=description,
        language=snippet.language,
        version="latest",
        package_name="qdrant",
        revision=1,
        source=snippet.source,
        context=snippet.context,
    )


def _parse_markdown(url: str) -> _ParsingResult:
    resp = requests.get(urljoin(url, "index.md"))
    if not resp.ok:
        return _ParsingResult(snippets=[], url=url)

    document = resp.text
    md_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()

    tokens = MarkdownIt("commonmark").parse(document)
    root = SyntaxTreeNode(tokens)
    partial_snippets = _extract_from_markdown_tree(document, root, url, md_hash)
    snippets = [_fill_snippet(snippet) for snippet in partial_snippets]
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
    #
    # qdrant_client.create_payload_index(
    #     collection_name=SNIPPET_COLLECTION_NAME,
    #     field_name="url",
    #     field_schema=TextIndexParams(
    #         type=TextIndexType.TEXT,
    #         tokenizer=TokenizerType.WORD,
    #         min_token_len=1,
    #         max_token_len=20,
    #         lowercase=True,
    #     ),
    #     wait=True,
    # )
    #
    qdrant_client.create_payload_index(
        collection_name=SNIPPET_COLLECTION_NAME,
        field_name="language",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
    #
    # qdrant_client.create_payload_index(
    #     collection_name=SNIPPET_COLLECTION_NAME,
    #     field_name="slug",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=True,
    # )
    #
    # qdrant_client.create_payload_index(
    #     collection_name=SNIPPET_COLLECTION_NAME,
    #     field_name="parent_sections",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=True,
    # )
    #
    # qdrant_client.create_payload_index(
    #     collection_name=SNIPPET_COLLECTION_NAME,
    #     field_name="parent_pages",
    #     field_schema=PayloadSchemaType.KEYWORD,
    #     wait=True,
    # )

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

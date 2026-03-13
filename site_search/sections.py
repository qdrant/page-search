from qdrant_client.models import PayloadSchemaType
from collections import namedtuple
from loguru import logger
import concurrent.futures
import hashlib
import uuid
import re
from urllib.parse import urljoin, urlsplit

import requests
import tqdm
from pydantic import BaseModel
from usp.fetch_parse import SitemapFetcher
from usp.objects.sitemap import IndexWebsiteSitemap, InvalidSitemap
from usp.tree import sitemap_tree_for_homepage
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Document,
    Filter,
    FieldCondition,
    MatchValue,
    TokenizerType,
    TextIndexParams,
    TextIndexType,
)
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode

from site_search.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    NEURAL_ENCODER,
    SECTION_COLLECTION_NAME,
)

# https://spec.commonmark.org/0.31.2/#atx-heading
section_pattern = re.compile(r"\s{0,3}(#+)\s+(.*?)\s*#*\s*")


def _slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[\s\-_]+", "-", s)
    s = re.sub(r"[^\w\-]+", "", s)
    return s


class Parent(BaseModel):
    title: str
    slug: str


class Section(BaseModel):
    title: str
    slug: str
    content: str
    url: str
    parents: list[Parent]
    level: int
    line: int

    @property
    def metadata(self):
        return self.dict()

    @property
    def uuid(self) -> str:
        content = str(self.dict(include={"url", "line", "content"}))
        # Create a SHA-256 hash of the content
        content_hash = hashlib.sha256(content.encode("utf-8")).digest()
        # Use the first 16 bytes of the hash to create a UUID
        return str(uuid.UUID(bytes=content_hash[:16]))

    def as_point(self, model: str) -> PointStruct:
        return PointStruct(
            id=self.uuid,
            payload=self.metadata,
            vector=Document(text=self.content, model=model),
        )


class _ParsingResult(BaseModel):
    url: str
    sections: list[Section]


class _Heading(BaseModel):
    tag: str
    line: int
    title: str

    @property
    def level(self) -> int:
        return int(self.tag[-1])


def _parse_markdown(url: str) -> _ParsingResult:
    resp = requests.get(urljoin(url, "index.md"))
    if not resp.ok:
        return _ParsingResult(sections=[], url=url)

    document = resp.text
    tokens = MarkdownIt("commonmark").parse(document)
    root = SyntaxTreeNode(tokens)
    headings: list[_Heading] = [
        _Heading(tag=node.tag, line=node.map[0], title=node.children[0].content)  # ty:ignore[not-subscriptable]
        for node in root.children
        if node.type == "heading"
    ]

    if headings[0].line != 0:
        headings.insert(0, _Heading(tag="h1", line=0, title=""))

    lines = document.splitlines()

    base_parents: list[Parent] = [
        Parent(title=p, slug=p) for p in urlsplit(url).path.strip("/").split("/")
    ]

    # needs to be a dict because the first section does not have to be level 1
    last: dict[int, Section | None] = {}

    sections: list[Section] = []

    for i, heading in enumerate(headings):
        # higher level sections end here
        for j in range(heading.level + 1, 6):
            last[j] = None

        # lower level sections are parents
        parents: list[Parent] = []
        for j in range(1, heading.level):
            if (parent := last.get(j)) is not None:
                parents.append(Parent(title=parent.title, slug=parent.slug))

        # content should go from start of a section to the start of the next
        if i < len(headings) - 1:
            content = lines[heading.line : headings[i + 1].line]
        else:
            content = lines[heading.line :]

        section = Section(
            title=heading.title,
            content="\n".join(content),
            slug=_slugify(heading.title),
            url=url,
            parents=base_parents + parents,
            level=heading.level,
            line=heading.line,
        )

        last[heading.level] = section
        sections.append(section)

    return _ParsingResult(sections=sections, url=url)


def _all_sitemap_urls(url: str, sitemap_url: str | None = None) -> list[str]:
    if not sitemap_url:
        tree = sitemap_tree_for_homepage(url)
        all_pages = tree.all_pages()
    else:
        sitemaps = []
        unpublished_sitemap_fetcher = SitemapFetcher(
            url=sitemap_url,
            web_client=None,
            recursion_level=0,
        )
        unpublished_sitemap = unpublished_sitemap_fetcher.sitemap()

        # Skip the ones that weren't found
        if not isinstance(unpublished_sitemap, InvalidSitemap):
            sitemaps.append(unpublished_sitemap)

        index_sitemap = IndexWebsiteSitemap(url=url, sub_sitemaps=sitemaps)
        all_pages = index_sitemap.all_pages()

    return [urljoin(url, p.url) for p in all_pages]


class SectionSearcher:
    def __init__(self):
        self.client: QdrantClient
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            local_inference_batch_size=32,
        )

    def search(self, query: str | None, path: str) -> list[Section]:
        conditions = [
            FieldCondition(key=f"parents[{i}].title", match=MatchValue(value=title))
            for i, title in enumerate(path.strip("/").split("/"))
        ]

        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query_filter=Filter(
                must=conditions + []
                if query is None
                else [FieldCondition(key="title", match=MatchValue(value=query))]
            ),
        )
        if len(result.points) > 0:
            return [Section.parse_obj(p.payload) for p in result.points]

        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query=None if query is None else Document(text=query, model=NEURAL_ENCODER),
            query_filter=Filter(must=conditions),
        )
        return [Section.parse_obj(p.payload) for p in result.points]


def main():
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=int(QDRANT_PORT),
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
        local_inference_batch_size=32,
    )

    if qdrant_client.collection_exists(SECTION_COLLECTION_NAME):
        qdrant_client.delete_collection(SECTION_COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=SECTION_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=qdrant_client.get_embedding_size(NEURAL_ENCODER),
            distance=Distance.COSINE,
        ),
    )
    
    qdrant_client.create_payload_index(
        collection_name=SECTION_COLLECTION_NAME,
        field_name="title",
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
        collection_name=SECTION_COLLECTION_NAME,
        field_name="slug",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    qdrant_client.create_payload_index(
        collection_name=SECTION_COLLECTION_NAME,
        field_name="parents[].title",
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
        collection_name=SECTION_COLLECTION_NAME,
        field_name="parents[].slug",
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

            if len(result.sections) == 0:
                continue

            qdrant_client.upsert(
                SECTION_COLLECTION_NAME,
                points=[
                    section.as_point(NEURAL_ENCODER) for section in result.sections
                ],
            )


if __name__ == "__main__":
    main()

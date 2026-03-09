import concurrent.futures
import hashlib
import uuid
import re
from urllib.parse import urljoin

import requests
import tqdm
from pydantic import BaseModel
from usp.fetch_parse import SitemapFetcher
from usp.objects.sitemap import IndexWebsiteSitemap, InvalidSitemap
from usp.tree import sitemap_tree_for_homepage
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Document

from site_search.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    NEURAL_ENCODER,
    SECTION_COLLECTION_NAME,
)

# https://spec.commonmark.org/0.31.2/#atx-heading
section_pattern = re.compile(r"\s{0,3}(#+)\s+(.*?)\s*#*\s*")


class Section(BaseModel):
    title: str
    content: str
    url: str
    parents: list[str]
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


def _parse_markdown(url: str) -> _ParsingResult:
    resp = requests.get(urljoin(url, "index.md"))
    if not resp.ok:
        return _ParsingResult(sections=[], url=url)

    lines = resp.text.splitlines()

    # needs to be a dict because the first section does not have to be level 1
    last: dict[int, Section | None] = {}

    sections: list[Section] = []

    for lnum, line in enumerate(lines):
        if match := section_pattern.fullmatch(line):
            level = len(match.group(1))
            title = match.group(2)

            # higher level sections end here
            for i in range(level + 1, 6):
                last[i] = None

            # lower level sections are parents
            parents: list[str] = []
            for i in range(1, level):
                if (parent := last.get(i)) is not None:
                    parents.append(parent.title)

            section = Section(
                title=title,
                content=line,
                url=url,
                parents=parents,
                level=level,
                line=lnum,
            )

            last[level] = section
            sections.append(section)
        else:
            # text before any heading
            if len(sections) == 0:
                sections.append(
                    Section(
                        title="",
                        content="",
                        url=url,
                        parents=[],
                        level=1,
                        line=lnum,
                    )
                )

            # append text to last section content
            sections[-1].content += "\n" + line

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


def main():
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
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

    urls = _all_sitemap_urls("https://qdrant.tech/", "https://qdrant.tech/sitemap.xml")

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_parse_markdown, url) for url in urls]

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(urls)):
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

import concurrent.futures
import hashlib
import uuid
from itertools import accumulate
from urllib.parse import urljoin, urlsplit

import requests
import tqdm
import yaml
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
from usp.fetch_parse import SitemapFetcher
from usp.objects.sitemap import IndexWebsiteSitemap, InvalidSitemap
from usp.tree import sitemap_tree_for_homepage

from site_search.config import (
    SNIPPET_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SKILLS_COLLECTION_NAME,
)


class Skill(BaseModel):
    name: str
    description: str
    content: str
    url: str
    page: str
    parent_pages: list[str]

    @property
    def metadata(self):
        return self.dict()

    @property
    def frontmatter(self):
        return f"---\nname: {self.name}\ndescription: {self.description}\n---"

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
    skills: list[Skill]


class _SkillMetadata(BaseModel):
    name: str
    description: str
    content: str


def _parse_frontmatter(content: str) -> _SkillMetadata | None:
    lines = content.splitlines()
    if lines[0] != "---":
        # Not a valid skill
        return None

    front_lines: list[str] = []
    i = 1
    for line in lines[1:]:
        i += 1
        if line == "---":
            break
        front_lines.append(line)
    frontmatter = "\n".join(front_lines)
    metadata = yaml.safe_load(frontmatter)
    metadata["content"] = "\n".join(lines[i:])
    return _SkillMetadata.parse_obj(metadata)


def _parse_markdown(url: str) -> _ParsingResult:
    resp = requests.get(url)
    if not resp.ok:
        return _ParsingResult(skills=[], url=url)

    document = resp.text

    metadata = _parse_frontmatter(document)
    if metadata is None:
        return _ParsingResult(skills=[], url=url)

    page = "/".join(urlsplit(url).path.strip("/").split("/")[:-1])
    parent_pages: list[str] = list(
        accumulate(page.split("/"), lambda a, b: a + "/" + b)
    )

    skill = Skill(
        name=metadata.name,
        description=metadata.description,
        content=metadata.content,
        url=url,
        parent_pages=parent_pages,
        page=page,
    )

    return _ParsingResult(skills=[skill], url=url)


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
        port=int(QDRANT_PORT),
        api_key=QDRANT_API_KEY,
        cloud_inference=True,
        timeout=30,
    )

    if qdrant_client.collection_exists(SKILLS_COLLECTION_NAME):
        qdrant_client.delete_collection(SKILLS_COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=SKILLS_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=qdrant_client.get_embedding_size(SNIPPET_ENCODER),
            distance=Distance.COSINE,
        ),
    )

    qdrant_client.create_payload_index(
        collection_name=SKILLS_COLLECTION_NAME,
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

    qdrant_client.create_payload_index(
        collection_name=SKILLS_COLLECTION_NAME,
        field_name="content",
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
        collection_name=SKILLS_COLLECTION_NAME,
        field_name="url",
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
        collection_name=SKILLS_COLLECTION_NAME,
        field_name="name",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    qdrant_client.create_payload_index(
        collection_name=SKILLS_COLLECTION_NAME,
        field_name="page",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    qdrant_client.create_payload_index(
        collection_name=SKILLS_COLLECTION_NAME,
        field_name="parent_pages",
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )

    urls = _all_sitemap_urls(
        "https://skills.qdrant.tech", "https://skills.qdrant.tech/sitemap.xml"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_parse_markdown, url) for url in urls]

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(urls)
        ):
            result = future.result()

            if len(result.skills) == 0:
                continue

            qdrant_client.upsert(
                SKILLS_COLLECTION_NAME,
                points=[skill.as_point(SNIPPET_ENCODER) for skill in result.skills],
            )


if __name__ == "__main__":
    main()

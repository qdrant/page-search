from urllib.parse import urljoin, urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Document,
    FieldCondition,
    Filter,
    MatchValue,
)
from starlette.datastructures import URL

from site_search.config import (
    NEURAL_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SECTION_COLLECTION_NAME,
    SECTIONS_EXACT_LIMIT,
    SECTIONS_SEARCH_LIMIT,
)
from site_search.sections import Section, all_sitemap_urls, slugify_heading

RESULT_TEMPLATE = """Read one level up: {up_url}

{sections_text}"""

SECTION_TEMPLATE = """{content}
Search within this section: {section_url}
"""


class SectionSearchResult(BaseModel):
    sections: list[Section]
    sublinks: list[str] | None = None

    def markdown(self, request_url: URL) -> str:
        up_url = request_url.replace(
            path="/".join(request_url.path.strip("/").split("/")[:-1])
        )
        sections_text = "\n".join(
            SECTION_TEMPLATE.format(
                content=section.content,
                section_url=request_url.replace(
                    path=section.parents[-1] + "/" + section.slug
                ),
            )
            for section in sorted(self.sections, key=lambda s: (s.url, s.line))
        )

        result = RESULT_TEMPLATE.format(up_url=up_url, sections_text=sections_text)
        if self.sublinks is not None and len(self.sublinks) > 0:
            result += "\n## Subsites to Search\n"
            for sub in self.sublinks:
                result += f"\n{request_url.replace(path=sub)}"

        return result


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

        self._base_url = "https://qdrant.tech/"
        self._all_paths = {
            urlsplit(url).path.strip("/")
            for url in all_sitemap_urls(
                self._base_url, "https://qdrant.tech/sitemap.xml"
            )
        }

    def search(self, query: str | None, path: str) -> SectionSearchResult:
        if query is None and path.strip("/") in self._all_paths:
            # return everything on this page
            result = self.client.query_points(
                SECTION_COLLECTION_NAME,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="url",
                            match=MatchValue(value=urljoin(self._base_url, path)),
                        )
                    ]
                ),
                limit=SECTIONS_EXACT_LIMIT,
            )
            return SectionSearchResult(
                sections=[Section.parse_obj(p.payload) for p in result.points],
                sublinks=[
                    u
                    for u in self._all_paths
                    if u.startswith(path.strip("/")) and u != path.strip("/")
                ],
            )

        conditions = [
            FieldCondition(key="parents", match=MatchValue(value=path.strip("/")))
        ]

        # try to find an exact matching heading for the query
        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query_filter=Filter(
                must=conditions + []
                if query is None
                else [
                    FieldCondition(
                        key="slug", match=MatchValue(value=slugify_heading(query))
                    )
                ]
            ),
            limit=SECTIONS_EXACT_LIMIT,
        )
        if len(result.points) > 0:
            return SectionSearchResult(
                sections=[Section.parse_obj(p.payload) for p in result.points]
            )

        # just search for the query
        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query=None if query is None else Document(text=query, model=NEURAL_ENCODER),
            query_filter=Filter(must=conditions),
            limit=SECTIONS_SEARCH_LIMIT,
        )
        return SectionSearchResult(
            sections=[Section.parse_obj(p.payload) for p in result.points]
        )


searcher = SectionSearcher()
app = FastAPI()

add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/{path:path}")
async def read_item(path: str, request: Request, q: str | None = None):
    section_result = searcher.search(query=q, path=path)
    logger.info(f"{len(section_result.sections)=}")
    return Response(
        content=section_result.markdown(request_url=request.url),
        media_type="text/markdown; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

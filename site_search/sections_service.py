from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Condition,
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
from site_search.sections import Section, slugify_heading

RESULT_TEMPLATE = """Read one level up: {up_url}

{sections_text}"""

SECTION_TEMPLATE = """{content}"""


class SectionSearchResult(BaseModel):
    sections: list[Section]
    sublinks: list[str] | None = None

    def markdown(self, request_url: URL) -> str:
        sections = sorted(self.sections, key=lambda s: (s.url, s.line))
        path = "/".join(request_url.path.strip("/").split("/")[:-1])
        query = None
        if (
            "s=" in request_url.query
            and len(sections) > 0
            and len(sections[0].parent_sections) > 1
        ):
            query = f"s={sections[0].parent_sections[-2]}"
            path = request_url.path.strip("/")

        up_url = request_url.replace(
            path=path, query=query
        )
        sections_text = "\n".join(
            SECTION_TEMPLATE.format(content=section.content)
            for section in sections
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

    def search(
        self, query: str | None, path: str, section: str | None
    ) -> SectionSearchResult:
        if section is None and query is not None:
            # Search in all sub-pages by query
            conditions: list[Condition] = [
                FieldCondition(
                    key="parent_pages", match=MatchValue(value=path.strip("/"))
                )
            ]
        else:
            # Retrieve a specific page
            conditions: list[Condition] = [
                FieldCondition(key="page", match=MatchValue(value=path.strip("/")))
            ]

        if section is not None:
            conditions.append(
                FieldCondition(key="parent_sections", match=MatchValue(value=section))
            )

        if query is not None:

            # Try to do exact match first
            result = self.client.query_points(
                SECTION_COLLECTION_NAME,
                query_filter=Filter(
                    must=conditions
                    + [
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
            
            # If no results, fallback to approximate search
            result = self.client.query_points(
                SECTION_COLLECTION_NAME,
                query=Document(text=query, model=NEURAL_ENCODER),
                query_filter=Filter(must=conditions),
                limit=SECTIONS_SEARCH_LIMIT,
            )
            return SectionSearchResult(
                sections=[Section.parse_obj(p.payload) for p in result.points]
            )

        # everything on this page and under this section
        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query_filter=Filter(must=conditions),
            limit=SECTIONS_EXACT_LIMIT,
        )

        sublinks = None
        if section is None:
            sublinks: list[str] = sorted(
                [
                    str(p.value)
                    for p in self.client.facet(
                        SECTION_COLLECTION_NAME,
                        key="page",
                        facet_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_pages",
                                    match=MatchValue(value=path.strip("/")),
                                )
                            ],
                            must_not=[
                                FieldCondition(
                                    key="page", match=MatchValue(value=path.strip("/"))
                                )
                            ],
                        ),
                        limit=SECTIONS_EXACT_LIMIT,
                    ).hits
                ]
            )

        return SectionSearchResult(
            sections=[Section.parse_obj(p.payload) for p in result.points],
            sublinks=sublinks,
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
async def read_item(
    path: str, request: Request, q: str | None = None, s: str | None = None
):
    section_result = searcher.search(query=q, path=path, section=s)
    logger.info(f"{len(section_result.sections)=}")
    return Response(
        content=section_result.markdown(request_url=request.url),
        media_type="text/markdown; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

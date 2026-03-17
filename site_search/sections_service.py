from starlette.datastructures import URL
from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Document,
    FieldCondition,
    Filter,
    MatchValue,
)

from site_search.config import (
    NEURAL_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SECTION_COLLECTION_NAME,
)
from site_search.sections import Section, slugify_heading


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
            FieldCondition(key="parents[]", match=MatchValue(value=path.strip("/")))
        ]

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
        )
        if len(result.points) > 0:
            return [Section.parse_obj(p.payload) for p in result.points]

        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query=None if query is None else Document(text=query, model=NEURAL_ENCODER),
            query_filter=Filter(must=conditions),
        )
        return [Section.parse_obj(p.payload) for p in result.points]


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

RESULT_TEMPLATE = """Search one level up: {up_url}

{sections_text}"""

SECTION_TEMPLATE = """{content}
Search within this section: {section_url}
"""


def _section_list_to_markdown(sections: list[Section], url: URL, path: str) -> str:
    up_url = url.replace(path="/".join(url.path.strip("/").split("/")[:-1]))
    sections_text = "\n".join(
        SECTION_TEMPLATE.format(
            content=section.content,
            section_url=url.replace(path=section.parents[-1] + "/" + section.slug),
        )
        for section in sections
    )

    return RESULT_TEMPLATE.format(up_url=up_url, sections_text=sections_text)


@app.get("/{path:path}")
async def read_item(path: str, request: Request, q: str | None = None):
    sections = searcher.search(query=q, path=path)
    logger.info(f"{len(sections)=}")
    return Response(
        content=_section_list_to_markdown(sections, request.url, path),
        media_type="text/markdown; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

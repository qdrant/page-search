from fastapi import FastAPI, Response
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


def _section_list_to_markdown(sections: list[Section]) -> str:
    return "\n".join(
        section.content + f"\n{section.url}#{section.slug}\n" for section in sections
    )


@app.get("/{path:path}")
async def read_item(path: str, q: str | None = None):
    sections = searcher.search(query=q, path=path)
    logger.info(f"{len(sections)=}")
    return Response(
        content=_section_list_to_markdown(sections),
        media_type="text/markdown; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

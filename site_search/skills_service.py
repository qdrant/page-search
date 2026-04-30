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
    SNIPPET_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SKILLS_COLLECTION_NAME,
    SKILLS_EXACT_LIMIT,
    SKILLS_SEARCH_LIMIT,
)
from site_search.skills import Skill
from site_search.sections import slugify_heading


class SkillSearchResult(BaseModel):
    skills: list[Skill]

    def markdown(self) -> str:
        return "\n\n".join(
            skill.frontmatter + "\n" + skill.content for skill in self.skills
        )


class SkillSearcher:
    def __init__(self):
        self.client: QdrantClient
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            local_inference_batch_size=32,
        )

    def search(self, query: str | None, path: str) -> SkillSearchResult:
        if query is not None:
            # Search in all sub-pages by query
            conditions: list[Condition] = [
                FieldCondition(
                    key="parent_pages", match=MatchValue(value=path.strip("/"))
                )
            ]
        elif len(path) > 0:
            # Retrieve a specific page
            conditions: list[Condition] = [
                FieldCondition(key="page", match=MatchValue(value=path.strip("/")))
            ]
        else:
            conditions: list[Condition] = []

        if query is not None:
            # Try to do exact match first
            result = self.client.query_points(
                SKILLS_COLLECTION_NAME,
                query_filter=Filter(
                    must=conditions
                    + [
                        FieldCondition(
                            key="name", match=MatchValue(value=slugify_heading(query))
                        )
                    ]
                ),
                limit=SKILLS_EXACT_LIMIT,
            )
            if len(result.points) > 0:
                return SkillSearchResult(
                    skills=[Skill.parse_obj(p.payload) for p in result.points]
                )

            # If no results, fallback to approximate search
            result = self.client.query_points(
                SKILLS_COLLECTION_NAME,
                query=Document(text=query, model=SNIPPET_ENCODER),
                query_filter=Filter(must=conditions),
                limit=SKILLS_SEARCH_LIMIT,
            )
            return SkillSearchResult(
                skills=[Skill.parse_obj(p.payload) for p in result.points]
            )

        # everything on this page and under this section
        result = self.client.query_points(
            SKILLS_COLLECTION_NAME,
            query_filter=Filter(must=conditions),
            limit=SKILLS_EXACT_LIMIT,
        )

        return SkillSearchResult(
            skills=[Skill.parse_obj(p.payload) for p in result.points],
        )


searcher = SkillSearcher()
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
    skill_result = searcher.search(query=q, path=path)
    logger.info(f"{len(skill_result.skills)=}")
    return Response(
        content=skill_result.markdown(),
        media_type="text/markdown; charset=utf-8",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

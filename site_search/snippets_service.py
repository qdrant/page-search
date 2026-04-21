import json
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security.api_key import APIKeyHeader
from fastapi_utils.timing import add_timing_middleware
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models

from site_search.config import (
    SNIPPET_ENCODER,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    SNIPPET_COLLECTION_NAME,
)
from site_search.snippets import Snippet


class Searcher:
    def __init__(self) -> None:
        self.client: QdrantClient
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            local_inference_batch_size=32,
        )

    def find_latest_revision(self) -> int:
        revisions = [
            int(hit.value)
            for hit in self.client.facet(
                SNIPPET_COLLECTION_NAME, key="revision", limit=1000000
            ).hits
        ]
        return max(revisions, default=0)

    def search(self, query: str, language: str, limit: int = 3) -> list[Snippet]:
        revision = self.find_latest_revision()
        points = self.client.query_points(
            collection_name=SNIPPET_COLLECTION_NAME,
            prefetch=[
                qdrant_models.Prefetch(
                    query=qdrant_models.Document(text=query, model=SNIPPET_ENCODER),
                    using="dense",
                    limit=20,
                ),
                qdrant_models.Prefetch(
                    query=qdrant_models.Document(
                        text=query, model="qdrant/bm25", options={"language": "none"}
                    ),
                    using="sparse",
                    limit=20,
                ),
            ],
            query=qdrant_models.RrfQuery(rrf=qdrant_models.Rrf(k=1)),
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="language",
                        match=qdrant_models.MatchValue(value=language),
                    ),
                    qdrant_models.FieldCondition(
                        key="revision",
                        match=qdrant_models.MatchValue(value=revision),
                    ),
                ]
            ),
            limit=limit,
        )

        return [Snippet.parse_obj(p.payload) for p in points.points]


app = FastAPI()
api_key_header = APIKeyHeader(name="api-key", auto_error=False)
searcher = Searcher()


def _resolve_format(
    format_param: str | None,
    accept: str | None,
) -> Literal["json", "markdown"]:
    """Resolve response format: query param overrides Accept header; default is markdown."""
    if format_param is not None:
        format_param = format_param.strip().lower()
        if format_param in ("json", "markdown"):
            return format_param
    if accept:
        for part in (p.strip() for p in accept.split(",")):
            part = part.split(";")[0].strip().lower() if part else ""
            if part == "application/json":
                return "json"
            if part == "text/markdown":
                return "markdown"
    return "markdown"


def _snippets_to_markdown(snippets: list[Snippet]) -> str:
    """Format snippet list as markdown (code blocks + source/metadata)."""
    blocks: list[str] = []
    for i, s in enumerate(snippets, 1):
        lang = s.language or "text"
        blocks.append(f"## Snippet {i}\n")
        blocks.append(f"*{s.package_name}* (v{s.version}) — {s.source.url}\n")
        if s.description:
            blocks.append(f"{s.description}\n")
        blocks.append(f"```{lang}\n{s.code}\n```\n")
    return "\n".join(blocks) if blocks else "No snippets found."


add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search")
async def search_snippets(
    request: Request,
    language: str,
    query: str,
    limit: int = 3,
    format: str | None = None,
) -> Response:
    snippets = searcher.search(query=query, language=language, limit=limit)
    accept = request.headers.get("accept")
    response_format = _resolve_format(format, accept)

    if response_format == "json":
        body = json.dumps(
            {"result": [s.dict() for s in snippets]},
            ensure_ascii=False,
        )
        return Response(content=body, media_type="application/json")
    else:
        return Response(
            content=_snippets_to_markdown(snippets),
            media_type="text/markdown; charset=utf-8",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

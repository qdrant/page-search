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
    Condition,
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

# RESULT_TEMPLATE = """Read one level up: {up_url}
#
# {sections_text}"""
#
# SECTION_TEMPLATE = """{content}
# Search within this section: {section_url}
# """
RESULT_TEMPLATE = """{sections_text}"""

SECTION_TEMPLATE = """{content}"""


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
                # section_url=request_url.replace(
                #     path=section.parents[-1] + "/" + section.slug
                # ),
            )
            for section in sorted(self.sections, key=lambda s: (s.url, s.line))
        )

        # result = RESULT_TEMPLATE.format(up_url=up_url, sections_text=sections_text)
        result = RESULT_TEMPLATE.format(sections_text=sections_text)
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
        # if query is not None and section is None:
        #     # TODO: Search through everything on and under path
        #     raise NotImplementedError

        # exact: try to match section, if query also try to match to section
        conditions: list[Condition] = [
            FieldCondition(key="page", match=MatchValue(value=path.strip("/")))
        ]
        if section is not None:
            conditions.append(
                FieldCondition(key="parent_sections", match=MatchValue(value=section))
            )
        #
        # if query is not None:
        #     result = self.client.query_points(
        #         SECTION_COLLECTION_NAME,
        #         query_filter=Filter(
        #             must=conditions
        #             + [
        #                 FieldCondition(
        #                     key="slug", match=MatchValue(value=slugify_heading(query))
        #                 )
        #             ]
        #         ),
        #         limit=SECTIONS_EXACT_LIMIT,
        #     )
        #     if len(result.points) > 0:
        #         return SectionSearchResult(
        #             sections=[Section.parse_obj(p.payload) for p in result.points]
        #         )
        #     result = self.client.query_points(
        #         SECTION_COLLECTION_NAME,
        #         query=Document(text=query, model=NEURAL_ENCODER),
        #         query_filter=Filter(must=conditions),
        #         limit=SECTIONS_SEARCH_LIMIT,
        #     )
        #     return SectionSearchResult(
        #         sections=[Section.parse_obj(p.payload) for p in result.points]
        #     )

        # everything on this page and under this section
        result = self.client.query_points(
            SECTION_COLLECTION_NAME,
            query_filter=Filter(must=conditions),
            limit=SECTIONS_EXACT_LIMIT,
        )
        return SectionSearchResult(
            sections=[Section.parse_obj(p.payload) for p in result.points]
        )
        # TODO: sub-links if section is None

        # if query is None:
        #     # try to find a section that exactly matches this path
        #     sections: list[Section] = [
        #         Section.parse_obj(p.payload)
        #         for p in self.client.query_points(
        #             SECTION_COLLECTION_NAME,
        #             query_filter=Filter(
        #                 must=[
        #                     FieldCondition(
        #                         key="path", match=MatchValue(value=path.strip("/"))
        #                     )
        #                 ]
        #             ),
        #             limit=SECTIONS_EXACT_LIMIT,
        #         ).points
        #     ]
        #
        #     # if we found no sections with this exact path, the path might point to a page
        #     page = sections[0].page if len(sections) > 0 else path.strip("/")
        #
        #     # return everything on this page that's below this path
        #     sub_sections: list[Section] = [
        #         Section.parse_obj(p.payload)
        #         for p in self.client.query_points(
        #             SECTION_COLLECTION_NAME,
        #             query_filter=Filter(
        #                 must=conditions
        #                 + [FieldCondition(key="page", match=MatchValue(value=page))]
        #             ),
        #             limit=SECTIONS_EXACT_LIMIT,
        #         ).points
        #     ]
        #
        #     # add links to all pages that are below this one in the hierarchy
        #     sub_pages: list[Section] = [
        #         Section.parse_obj(p.payload)
        #         for p in self.client.query_points(
        #             SECTION_COLLECTION_NAME,
        #             query_filter=Filter(
        #                 must=conditions,
        #                 must_not=[
        #                     FieldCondition(key="page", match=MatchValue(value=page))
        #                 ],
        #             ),
        #             limit=SECTIONS_EXACT_LIMIT,
        #         ).points
        #     ]
        #     return SectionSearchResult(
        #         sections=sections + sub_sections,
        #         sublinks=[s.path for s in sub_pages],
        #     )
        #
        # # try to find an exact matching heading for the query
        # result = self.client.query_points(
        #     SECTION_COLLECTION_NAME,
        #     query_filter=Filter(
        #         must=conditions
        #         + [
        #             FieldCondition(
        #                 key="slug", match=MatchValue(value=slugify_heading(query))
        #             )
        #         ]
        #     ),
        #     limit=SECTIONS_EXACT_LIMIT,
        # )
        # if len(result.points) > 0:
        #     return SectionSearchResult(
        #         sections=[Section.parse_obj(p.payload) for p in result.points]
        #     )
        #
        # # just search for the query
        # result = self.client.query_points(
        #     SECTION_COLLECTION_NAME,
        #     query=Document(text=query, model=NEURAL_ENCODER),
        #     query_filter=Filter(must=conditions),
        #     limit=SECTIONS_SEARCH_LIMIT,
        # )
        # return SectionSearchResult(
        #     sections=[Section.parse_obj(p.payload) for p in result.points]
        # )


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

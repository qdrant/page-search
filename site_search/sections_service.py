from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_utils.timing import add_timing_middleware
from loguru import logger

from site_search.sections import Section, SectionSearcher

app = FastAPI()

add_timing_middleware(app, record=logger.info, prefix="app", exclude="untimed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher = SectionSearcher()


def _section_list_to_markdown(sections: list[Section]) -> str:
    return "\n".join(section.content + f"\n{section.url}#{section.slug}\n" for section in sections)


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

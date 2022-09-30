from fastapi import FastAPI

from site_search.config import COLLECTION_NAME
from site_search.neural_searcher import NeuralSearcher
from fastapi.middleware.cors import CORSMiddleware

from site_search.site_searcher import SiteSearcher

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

searcher = SiteSearcher(collection_name=COLLECTION_NAME)


@app.get("/api/search")
async def read_item(q: str, section=None):
    return {
        "result": searcher.search(text=q, section=section)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

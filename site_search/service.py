from fastapi import FastAPI

from site_search.config import COLLECTION_NAME
from site_search.neural_searcher import NeuralSearcher

app = FastAPI()

neural_searcher = NeuralSearcher(collection_name=COLLECTION_NAME)


@app.get("/api/search")
async def read_item(q: str, section=None):
    section_filter = {
        "must": [
            {"key": "sections", "match": {"value": section}},
        ]
    } if section else None
    return {
        "result": neural_searcher.search(text=q, filter_=section_filter)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

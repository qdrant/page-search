from typing import List, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from sentence_transformers import SentenceTransformer

from site_search.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY

BATCH_SIZE = 32


def iter_batch(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class NeuralSearcher:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L12-v2', device='cpu')
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, https=True, api_key=QDRANT_API_KEY)

    def search(self, text: str, filter_: dict = None) -> List[dict]:
        vector = self.model.encode(text).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=Filter(**filter_) if filter_ else None,
            limit=5,
            with_payload=True,
            with_vector=False,
        )
        payloads = [{"payload": hit.payload, "score": hit.score} for hit in search_result]
        return payloads

    def encode_iter(self, texts: Iterable[str]) -> Iterable[list]:
        for batch in iter_batch(texts, BATCH_SIZE):
            vectors = self.model.encode(batch).tolist()
            for vector in vectors:
                yield vector


if __name__ == '__main__':
    searcher = NeuralSearcher(collection_name='test')
    vectors = list(searcher.encode_iter(['hello', 'world']))
    for vec in vectors:
        print(vec)

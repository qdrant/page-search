import json
import os
from typing import Iterable

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams, TextIndexParams, TokenizerType

from site_search.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, DATA_DIR, QDRANT_API_KEY
from site_search.neural_searcher import NeuralSearcher

BATCH_SIZE = 256

encoder = NeuralSearcher(COLLECTION_NAME)


def read_records(filename: str) -> Iterable[dict]:
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)


def read_text_records(filename: str) -> Iterable[str]:
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)['text']


if __name__ == '__main__':
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY,
                                 prefer_grpc=True)

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )

    records_path = os.path.join(DATA_DIR, 'abstracts.jsonl')

    vectors = encoder.encode_iter(read_text_records(records_path))
    payloads = read_records(records_path)

    index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='sections',
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True
    )

    text_index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='text',
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.PREFIX,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        ),
        wait=True
    )

    qdrant_client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=BATCH_SIZE,
        parallel=2
    )

import json
import os
from typing import Iterable

import tqdm
from blingfire import text_to_sentences
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from site_search.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, DATA_DIR, QDRANT_API_KEY
from site_search.neural_searcher import NeuralSearcher

BATCH_SIZE = 256

encoder = NeuralSearcher(COLLECTION_NAME)


def read_records(filename: str) -> Iterable[dict]:
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)


def read_sentence_records(filename: str) -> Iterable[dict]:
    for record in read_records(filename):
        sentences = text_to_sentences(record['text']).split("\n")
        for i in range(len(sentences)):
            yield {
                **record,
                'text': " ".join(sentences[i:i + 1]),
            }


def read_text_records(filename: str, reader=read_records) -> Iterable[str]:
    for record in reader(filename):
        yield record['text']


if __name__ == '__main__':
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
        timeout=600,
        cloud_inference=True
    )

    qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=encoder.get_model_dim(),
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True
                ),
                modifier=models.Modifier.IDF,
            )
        }
    )

    records_path = os.path.join(DATA_DIR, 'abstracts.jsonl')
    records_reader = read_sentence_records

    payloads = records_reader(records_path)
    vectors = tqdm.tqdm(encoder.encode_iter(read_text_records(records_path, reader=records_reader)))

    index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='sections',
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    tags_index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='tag',
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    partition_index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='partition',
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    text_index_response = qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name='text',
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.PREFIX,
            min_token_len=1,
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

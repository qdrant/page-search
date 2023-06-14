import json
import os
from typing import Iterable

import tqdm
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams, TextIndexParams, TokenizerType

from site_search.common import section_hash
from site_search.config import QDRANT_HOST, QDRANT_PORT, DATA_DIR, QDRANT_API_KEY
from site_search.neural_searcher import NeuralSearcher

BATCH_SIZE = 256

encoder = NeuralSearcher("")

SECTION_COLLECTION_NAME = "sections"
ABS_COLLECTION_NAME = "abstracts"


def read_records(filename: str) -> Iterable[dict]:
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)


def read_text_records(filename: str, reader=read_records) -> Iterable[str]:
    for record in reader(filename):
        yield record['text']


def read_section_texts(filename: str, reader=read_records) -> Iterable[dict]:
    for record in reader(filename):
        yield " ".join(record['titles'])


def upload_sections(qdrant_client: QdrantClient):
    qdrant_client.recreate_collection(
        collection_name=SECTION_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2
        )
    )

    records_path = os.path.join(DATA_DIR, 'sections.jsonl')

    payloads = read_records(records_path)
    vectors = encoder.encode_iter(tqdm.tqdm(read_section_texts(records_path, reader=read_records)))
    ids = [section_hash(record['url'], record['titles']) for record in read_records(records_path)]

    text_index_response = qdrant_client.create_payload_index(
        collection_name=SECTION_COLLECTION_NAME,
        field_name='titles',
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.PREFIX,
            min_token_len=1,
            max_token_len=20,
            lowercase=True,
        ),
        wait=True
    )

    qdrant_client.upload_collection(
        collection_name=SECTION_COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=ids,
        batch_size=BATCH_SIZE,
        parallel=1
    )


def upload_abstracts(qdrant_client):
    qdrant_client.recreate_collection(
        collection_name=ABS_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=encoder.model.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2
        )
    )

    records_path = os.path.join(DATA_DIR, 'section_abstracts.jsonl')
    records_reader = read_records

    payloads = records_reader(records_path)
    vectors = encoder.encode_iter(tqdm.tqdm(read_text_records(records_path, reader=records_reader)))

    index_response = qdrant_client.create_payload_index(
        collection_name=ABS_COLLECTION_NAME,
        field_name='section_id',
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True
    )

    text_index_response = qdrant_client.create_payload_index(
        collection_name=ABS_COLLECTION_NAME,
        field_name='text',
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.PREFIX,
            min_token_len=1,
            max_token_len=20,
            lowercase=True,
        ),
        wait=True
    )

    qdrant_client.upload_collection(
        collection_name=ABS_COLLECTION_NAME,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=BATCH_SIZE,
        parallel=2
    )


if __name__ == '__main__':
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY,
                                 prefer_grpc=True)

    upload_sections(qdrant_client)

    upload_abstracts(qdrant_client)

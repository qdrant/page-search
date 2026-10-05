import unittest
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient, models

from site_search import sections


class SectionsPreflightTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.client.create_collection(
            sections.SECTION_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        self.client.upsert(
            sections.SECTION_COLLECTION_NAME,
            points=[models.PointStruct(id=1, vector=[1.0, 0.0, 0.0])],
        )
        self.client_factory = self.enterContext(
            patch.object(sections, "QdrantClient", return_value=self.client)
        )
        self.encoder_factory = self.enterContext(patch.object(sections, "TextEmbedding"))
        self.sitemap = self.enterContext(
            patch.object(sections, "_all_sitemap_urls", return_value=[])
        )
        self.pool = self.enterContext(
            patch.object(sections.concurrent.futures, "ProcessPoolExecutor")
        ).return_value.__enter__.return_value

    def assert_existing_index_preserved(self):
        points = self.client.retrieve(sections.SECTION_COLLECTION_NAME, ids=[1])
        self.assertEqual([point.id for point in points], [1])
        self.client_factory.assert_not_called()
        self.sitemap.assert_not_called()

    def test_model_load_failure_preserves_existing_index(self):
        self.encoder_factory.side_effect = ValueError("Could not load model")

        with self.assertRaisesRegex(ValueError, "Could not load model"):
            sections.main()

        self.assert_existing_index_preserved()

    def test_lazy_inference_failure_preserves_existing_index(self):
        def failed_embeddings(*args, **kwargs):
            raise RuntimeError("Inference failed")
            yield  # Keep the failure lazy, like FastEmbed's embedding iterator.

        self.encoder_factory.return_value.embed.side_effect = failed_embeddings

        with self.assertRaisesRegex(RuntimeError, "Inference failed"):
            sections.main()

        self.assert_existing_index_preserved()

    def test_successful_preflight_reuses_encoder_to_index_sections(self):
        section = sections.Section(
            title="Example", slug="example", content="Example documentation",
            url="https://qdrant.tech/documentation/example/#example",
            page="documentation/example", parent_sections=[], parent_pages=[],
            level=1, line=0,
        )
        future = Future()
        future.set_result(sections._ParsingResult(url=section.url, sections=[section]))
        self.sitemap.return_value = [section.url]
        self.pool.submit.return_value = future

        def embeddings(texts, **kwargs):
            if self.encoder_factory.return_value.embed.call_count == 1:
                self.assert_existing_index_preserved()
            yield np.array([0.0, 1.0, 0.0])

        self.encoder_factory.return_value.embed.side_effect = embeddings
        sections.main()

        self.encoder_factory.assert_called_once_with(model_name=sections.NEURAL_ENCODER)
        self.encoder_factory.return_value.embed.assert_called_with(
            [section.content], batch_size=32
        )
        self.assertEqual(self.encoder_factory.return_value.embed.call_count, 2)
        points, _ = self.client.scroll(sections.SECTION_COLLECTION_NAME, with_vectors=True)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].id, section.uuid)
        self.assertEqual(points[0].payload, section.metadata)
        self.assertEqual(points[0].vector, [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

import unittest

from graphify_rag.models import Chunk, Entity, Relation
from graphify_rag.retrieval import HybridRetriever


class RetrievalTests(unittest.TestCase):
    def test_retrieve_prioritizes_matching_chunks(self) -> None:
        chunks = [
            Chunk("c1", "doc", "FinRL-X supports deployment consistency and modular strategy pipelines.", 0, 9),
            Chunk("c2", "doc", "Evaluation guidebooks discuss benchmarks and domains.", 1, 7),
        ]
        entities = [
            Entity("finrl-x", "FinRL-X", "CONCEPT", frequency=2, chunk_ids=["c1"]),
            Entity("Evaluation", "Evaluation", "TOPIC", frequency=1, chunk_ids=["c2"]),
        ]
        relations = [Relation("finrl-x", "Evaluation", "related_to", 1.0, ["c1"])]

        retriever = HybridRetriever(chunks, entities, relations)
        hits = retriever.retrieve("How does FinRL-X support modular deployment?", limit=1)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.chunk_id, "c1")

    def test_dense_signal_can_promote_semantic_match(self) -> None:
        chunks = [
            Chunk("c1", "doc", "FinRL-X supports deployment consistency and modular strategy pipelines.", 0, 9),
            Chunk("c2", "doc", "A system unifies research and live trading execution semantics.", 1, 10),
        ]
        entities = [Entity("finrl-x", "FinRL-X", "CONCEPT", frequency=2, chunk_ids=["c1", "c2"])]
        relations = []
        embeddings = {
            "c1": [0.1, 0.2, 0.1],
            "c2": [0.9, 0.8, 0.9],
        }

        retriever = HybridRetriever(chunks, entities, relations, embeddings=embeddings)
        hits = retriever.retrieve(
            "Which chunk is semantically closest to unified execution?",
            limit=1,
            lexical_weight=0.05,
            dense_weight=0.9,
            graph_chunk_boost=0.05,
            query_embedding=[0.95, 0.9, 0.95],
        )

        self.assertEqual(hits[0].chunk.chunk_id, "c2")


if __name__ == "__main__":
    unittest.main()

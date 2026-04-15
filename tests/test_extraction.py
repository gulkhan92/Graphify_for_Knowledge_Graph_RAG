import unittest

from graphify_rag.extraction import extract_entities, extract_relations
from graphify_rag.models import Chunk


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="doc-1-chunk-0",
                doc_id="doc-1",
                text="FinRL-X integrates LLM signals and Portfolio Allocation for quantitative trading.",
                index=0,
                token_count=10,
            ),
            Chunk(
                chunk_id="doc-1-chunk-1",
                doc_id="doc-1",
                text="The LLM Evaluation Guidebook presents evaluation domains and benchmark lifecycle design.",
                index=1,
                token_count=11,
            ),
        ]

    def test_extract_entities_finds_title_case_and_acronyms(self) -> None:
        entities = extract_entities(self.chunks, min_entity_length=3)
        names = {entity.name for entity in entities}

        self.assertIn("LLM", names)
        self.assertIn("Portfolio Allocation", names)

    def test_extract_relations_finds_sentence_level_links(self) -> None:
        entities = extract_entities(self.chunks, min_entity_length=3)
        relations = extract_relations(self.chunks, entities)

        self.assertTrue(relations)
        self.assertTrue(any(relation.relation in {"integrates", "introduces", "related_to"} for relation in relations))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

from graphify_rag.chunking import chunk_document
from graphify_rag.models import Document


class ChunkingTests(unittest.TestCase):
    def test_chunk_document_creates_ordered_chunks_with_overlap(self) -> None:
        document = Document(
            doc_id="doc-1",
            title="Sample",
            path=Path("sample.pdf"),
            content="Sentence one introduces FinRL-X. Sentence two adds modular architecture. "
            "Sentence three covers deployment consistency. Sentence four discusses execution realism.",
        )

        chunks = chunk_document(document, chunk_size=80, chunk_overlap=15, max_sentences_per_chunk=2)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, "doc-1-chunk-0")
        self.assertEqual(chunks[1].chunk_id, "doc-1-chunk-1")
        self.assertIn("architecture", chunks[1].text)


if __name__ == "__main__":
    unittest.main()

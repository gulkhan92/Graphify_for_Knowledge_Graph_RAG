from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graphify_rag.config import PipelineConfig
from graphify_rag.service import GraphRagService


class ServiceTests(unittest.TestCase):
    @patch("graphify_rag.service.load_documents")
    def test_ingest_and_answer(self, mock_load_documents) -> None:
        from graphify_rag.models import Document

        mock_load_documents.return_value = [
            Document(
                doc_id="doc-1",
                title="FinRL-X",
                path=Path("doc.pdf"),
                content="FinRL-X integrates LLM signals. LLM Evaluation Guidebook presents benchmarks.",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(input_dir=Path(tmp_dir), artifacts_dir=Path(tmp_dir) / "artifacts")
            config.input_dir.mkdir(parents=True, exist_ok=True)
            service = GraphRagService(config)

            snapshot = service.ingest()
            answer = service.answer("What does FinRL-X integrate?")

            self.assertEqual(len(snapshot.documents), 1)
            self.assertTrue(answer.evidence)
            self.assertIn("Grounded answer draft", answer.answer)


if __name__ == "__main__":
    unittest.main()

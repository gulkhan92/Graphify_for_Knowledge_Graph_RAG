from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

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
            self.assertEqual(answer.llm_provider, "deterministic")

    @patch("graphify_rag.service.GraphifyAdapter")
    def test_ingest_prefers_graphify_when_available(self, mock_graphify_adapter) -> None:
        from graphify_rag.models import Chunk, Document, Entity, GraphSnapshot, Relation

        mock_adapter_instance = mock_graphify_adapter.return_value
        mock_adapter_instance.build_snapshot.return_value = GraphSnapshot(
            documents=[Document("doc-1", "Graphify Doc", Path("doc.pdf"), "graphify text", metadata={"source_type": "graphify"})],
            chunks=[Chunk("c1", "doc-1", "graphify text", 0, 2)],
            entities=[Entity("e1", "FinRL-X", "Concept", chunk_ids=["c1"])],
            relations=[Relation("e1", "e1", "related_to", 1.0, [])],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(input_dir=Path(tmp_dir), artifacts_dir=Path(tmp_dir) / "artifacts")
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            service = GraphRagService(config)

            snapshot = service.ingest()

            self.assertEqual(snapshot.documents[0].metadata["source_type"], "graphify")
            mock_adapter_instance.build_snapshot.assert_called_once()

    @patch("graphify_rag.service.load_documents")
    def test_answer_uses_openai_client_when_configured(self, mock_load_documents) -> None:
        from graphify_rag.models import Document

        mock_load_documents.return_value = [
            Document(
                doc_id="doc-1",
                title="FinRL-X",
                path=Path("doc.pdf"),
                content="FinRL-X integrates LLM signals for modular trading workflows.",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                input_dir=Path(tmp_dir),
                artifacts_dir=Path(tmp_dir) / "artifacts",
                openai_api_key="test-key",
                use_openai_generation=True,
                use_openai_embeddings=False,
                use_openai_guardrails=True,
            )
            config.input_dir.mkdir(parents=True, exist_ok=True)
            service = GraphRagService(config)
            service.openai_client = Mock()
            service.openai_client.chat_completion.side_effect = [
                "OpenAI grounded answer.",
                '{"verdict":"PASS","issues":[],"revised_requirements":[]}',
            ]

            service.ingest()
            answer = service.answer("What does FinRL-X integrate?")

            self.assertEqual(answer.answer, "OpenAI grounded answer.")
            self.assertEqual(answer.llm_provider, "openai")
            self.assertEqual(answer.llm_model, config.openai_chat_model)
            self.assertEqual(answer.guardrail_status, "passed")
            self.assertEqual(service.openai_client.chat_completion.call_count, 2)

    @patch("graphify_rag.service.load_documents")
    def test_guardrail_can_trigger_regeneration(self, mock_load_documents) -> None:
        from graphify_rag.models import Document

        mock_load_documents.return_value = [
            Document(
                doc_id="doc-1",
                title="FinRL-X",
                path=Path("doc.pdf"),
                content="FinRL-X integrates LLM signals for modular trading workflows.",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = PipelineConfig(
                input_dir=Path(tmp_dir),
                artifacts_dir=Path(tmp_dir) / "artifacts",
                openai_api_key="test-key",
                use_openai_generation=True,
                use_openai_embeddings=False,
                use_openai_guardrails=True,
                max_guardrail_loops=1,
            )
            config.input_dir.mkdir(parents=True, exist_ok=True)
            service = GraphRagService(config)
            service.openai_client = Mock()
            service.openai_client.chat_completion.side_effect = [
                "Initial answer with unsupported claim.",
                '{"verdict":"FAIL","issues":["Unsupported claim detected."],"revised_requirements":["Remove unsupported claim and answer only from evidence."]}',
                "Revised grounded answer.",
                '{"verdict":"PASS","issues":[],"revised_requirements":[]}',
            ]

            service.ingest()
            answer = service.answer("What does FinRL-X integrate?")

            self.assertEqual(answer.answer, "Revised grounded answer.")
            self.assertEqual(answer.guardrail_status, "passed_after_retry")
            self.assertIn("Unsupported claim detected.", answer.guardrail_feedback)
            self.assertEqual(service.openai_client.chat_completion.call_count, 4)


if __name__ == "__main__":
    unittest.main()

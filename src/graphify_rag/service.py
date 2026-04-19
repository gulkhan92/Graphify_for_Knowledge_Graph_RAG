from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from graphify_rag.chunking import chunk_document
from graphify_rag.config import PipelineConfig
from graphify_rag.extraction import extract_entities, extract_relations
from graphify_rag.graphify_adapter import GraphifyAdapter, GraphifyError
from graphify_rag.graph_store import GraphStore
from graphify_rag.logging_utils import get_logger
from graphify_rag.models import AnswerPayload, GraphSnapshot
from graphify_rag.openai_client import OpenAIAPIError, OpenAIClient
from graphify_rag.pdf import load_documents
from graphify_rag.prompts import (
    GUARDRAIL_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_chat_turn,
    build_guardrail_turn,
    build_regeneration_turn,
    parse_guardrail_payload,
)
from graphify_rag.retrieval import HybridRetriever
from graphify_rag.validation import validate_chunks, validate_documents, validate_entities
from graphify_rag.vector_store import VectorStore


LOGGER = get_logger(__name__)


class GraphRagService:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.graph_store = GraphStore(config.artifacts_dir)
        self.vector_store = VectorStore(config.artifacts_dir)
        self.graphify_adapter = GraphifyAdapter(config.input_dir, config.artifacts_dir)
        self.snapshot: GraphSnapshot | None = None
        self.retriever: HybridRetriever | None = None
        self.embeddings: dict[str, list[float]] = {}
        self.openai_client = OpenAIClient(config.openai_api_key) if config.openai_api_key else None

    def ingest(self) -> GraphSnapshot:
        LOGGER.info("Starting ingestion from %s", self.config.input_dir)
        if self.config.prefer_graphify:
            try:
                snapshot = self.graphify_adapter.build_snapshot()
                LOGGER.info("Using Graphify-generated graph snapshot.")
                self.graph_store.save(snapshot)
                self.embeddings = self._build_embeddings(snapshot.chunks)
                if self.embeddings:
                    self.vector_store.save(self.embeddings)
                self.snapshot = snapshot
                self.retriever = HybridRetriever(
                    snapshot.chunks,
                    snapshot.entities,
                    snapshot.relations,
                    embeddings=self.embeddings,
                )
                return snapshot
            except GraphifyError as exc:
                LOGGER.warning("Graphify unavailable or failed, falling back to local extractor: %s", exc)

        documents = load_documents(self.config.input_dir)
        validate_documents(documents)
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(
                document=document,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                max_sentences_per_chunk=self.config.max_sentences_per_chunk,
            )
        ]
        validate_chunks(chunks)
        entities = extract_entities(chunks, min_entity_length=self.config.min_entity_length)
        validate_entities(entities)
        relations = extract_relations(chunks, entities)
        snapshot = GraphSnapshot(documents=documents, chunks=chunks, entities=entities, relations=relations)
        self.graph_store.save(snapshot)
        self.embeddings = self._build_embeddings(chunks)
        if self.embeddings:
            self.vector_store.save(self.embeddings)
        self.snapshot = snapshot
        self.retriever = HybridRetriever(chunks, entities, relations, embeddings=self.embeddings)
        LOGGER.info(
            "Completed ingestion: documents=%s chunks=%s entities=%s relations=%s embeddings=%s",
            len(documents),
            len(chunks),
            len(entities),
            len(relations),
            len(self.embeddings),
        )
        return snapshot

    def load(self) -> GraphSnapshot:
        self.snapshot = self.graph_store.load()
        try:
            self.embeddings = self.vector_store.load()
        except FileNotFoundError:
            self.embeddings = {}
        self.retriever = HybridRetriever(
            self.snapshot.chunks,
            self.snapshot.entities,
            self.snapshot.relations,
            embeddings=self.embeddings,
        )
        return self.snapshot

    def ensure_loaded(self) -> None:
        if self.snapshot is None or self.retriever is None:
            self.load()

    def corpus_summary(self) -> dict[str, object]:
        self.ensure_loaded()
        assert self.snapshot is not None
        graph_provider = next(
            (
                document.metadata.get("source_type")
                for document in self.snapshot.documents
                if document.metadata.get("source_type")
            ),
            "local",
        )
        return {
            "documents": len(self.snapshot.documents),
            "chunks": len(self.snapshot.chunks),
            "entities": len(self.snapshot.entities),
            "relations": len(self.snapshot.relations),
            "graph_provider": graph_provider,
            "top_entities": [asdict(entity) for entity in self.snapshot.entities[:10]],
        }

    def answer(self, question: str) -> AnswerPayload:
        self.ensure_loaded()
        assert self.retriever is not None
        query_embedding = self._embed_query(question)
        hits = self.retriever.retrieve(
            question,
            limit=self.config.max_evidence_chunks,
            lexical_weight=self.config.lexical_weight,
            dense_weight=self.config.dense_weight,
            graph_chunk_boost=self.config.graph_chunk_boost,
            query_embedding=query_embedding,
        )
        graph_context = self.retriever.graph_context(question, max_neighbors=self.config.max_graph_neighbors)

        evidence = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "score": round(hit.score, 4),
                "lexical_score": round(hit.lexical_score, 4),
                "dense_score": round(hit.dense_score, 4),
                "graph_score": round(hit.graph_score, 4),
                "text": hit.chunk.text,
            }
            for hit in hits
        ]

        if not evidence:
            answer = "No grounded answer could be produced from the indexed corpus."
            retrieval_mode = "empty"
            llm_provider = "deterministic"
            llm_model = None
            guardrail_status = "not_run"
            guardrail_feedback: list[str] = []
        else:
            answer, llm_provider, llm_model, guardrail_status, guardrail_feedback = self._generate_answer(
                question,
                evidence,
                graph_context,
            )
            retrieval_mode = "hybrid_dense_graph" if query_embedding is not None else "hybrid_lexical_graph"

        return AnswerPayload(
            answer=answer,
            question=question,
            evidence=evidence,
            graph_context=graph_context,
            retrieval_mode=retrieval_mode,
            llm_provider=llm_provider,
            llm_model=llm_model,
            guardrail_status=guardrail_status,
            guardrail_feedback=guardrail_feedback,
        )

    def validate_paths(self) -> None:
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _build_embeddings(self, chunks: list[object]) -> dict[str, list[float]]:
        if not self.openai_client or not self.config.use_openai_embeddings:
            return {}
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        try:
            vectors = self.openai_client.embed_texts(texts, self.config.openai_embedding_model)
        except OpenAIAPIError as exc:
            LOGGER.warning("Embedding generation failed; continuing without dense index: %s", exc)
            return {}
        return {chunk_id: vector for chunk_id, vector in zip(chunk_ids, vectors)}

    def _embed_query(self, question: str) -> list[float] | None:
        if not self.openai_client or not self.config.use_openai_embeddings or not self.embeddings:
            return None
        try:
            vectors = self.openai_client.embed_texts([question], self.config.openai_embedding_model)
        except OpenAIAPIError as exc:
            LOGGER.warning("Query embedding failed; falling back to lexical retrieval: %s", exc)
            return None
        return vectors[0] if vectors else None

    def _generate_answer(
        self,
        question: str,
        evidence: list[dict[str, object]],
        graph_context: list[dict[str, object]],
    ) -> tuple[str, str, str | None, str, list[str]]:
        if self.openai_client and self.config.use_openai_generation:
            try:
                response = self._generate_with_guardrails(question, evidence, graph_context)
                return (
                    response["answer"],
                    "openai",
                    self.config.openai_chat_model,
                    response["guardrail_status"],
                    response["guardrail_feedback"],
                )
            except OpenAIAPIError as exc:
                LOGGER.warning("OpenAI answer generation failed; falling back to deterministic synthesis: %s", exc)

        supporting = " ".join(str(item["text"]) for item in evidence[:2])
        answer = f"Grounded answer draft: {supporting[:900].strip()}"
        return answer, "deterministic", None, "not_run", []

    def _generate_with_guardrails(
        self,
        question: str,
        evidence: list[dict[str, object]],
        graph_context: list[dict[str, object]],
    ) -> dict[str, object]:
        assert self.openai_client is not None
        answer = self.openai_client.chat_completion(
            model=self.config.openai_chat_model,
            system_prompt=SYSTEM_PROMPT,
            messages=[build_chat_turn(question, evidence, graph_context)],
        )
        if not self.config.use_openai_guardrails:
            return {"answer": answer, "guardrail_status": "skipped", "guardrail_feedback": []}

        guardrail_feedback: list[str] = []
        for attempt in range(self.config.max_guardrail_loops + 1):
            verdict = self._validate_answer(question, evidence, graph_context, answer)
            if verdict["verdict"] == "PASS":
                status = "passed" if attempt == 0 else "passed_after_retry"
                return {"answer": answer, "guardrail_status": status, "guardrail_feedback": guardrail_feedback}
            guardrail_feedback = [*verdict["issues"], *verdict["revised_requirements"]]
            if attempt >= self.config.max_guardrail_loops:
                return {"answer": answer, "guardrail_status": "failed", "guardrail_feedback": guardrail_feedback}
            answer = self.openai_client.chat_completion(
                model=self.config.openai_chat_model,
                system_prompt=SYSTEM_PROMPT,
                messages=[build_regeneration_turn(question, evidence, graph_context, guardrail_feedback)],
            )
        return {"answer": answer, "guardrail_status": "failed", "guardrail_feedback": guardrail_feedback}

    def _validate_answer(
        self,
        question: str,
        evidence: list[dict[str, object]],
        graph_context: list[dict[str, object]],
        candidate_answer: str,
    ) -> dict[str, object]:
        assert self.openai_client is not None
        response = self.openai_client.chat_completion(
            model=self.config.openai_guardrail_model,
            system_prompt=GUARDRAIL_SYSTEM_PROMPT,
            messages=[build_guardrail_turn(question, evidence, graph_context, candidate_answer)],
        )
        try:
            return parse_guardrail_payload(response)
        except Exception as exc:
            LOGGER.warning("Guardrail validation payload could not be parsed: %s", exc)
            return {
                "verdict": "FAIL",
                "issues": ["Validator returned an unreadable response."],
                "revised_requirements": ["Regenerate the answer strictly from the cited evidence."],
            }

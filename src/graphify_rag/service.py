from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from graphify_rag.chunking import chunk_document
from graphify_rag.config import PipelineConfig
from graphify_rag.extraction import extract_entities, extract_relations
from graphify_rag.graph_store import GraphStore
from graphify_rag.models import AnswerPayload, GraphSnapshot
from graphify_rag.pdf import load_documents
from graphify_rag.retrieval import HybridRetriever
from graphify_rag.validation import validate_chunks, validate_documents, validate_entities


class GraphRagService:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.graph_store = GraphStore(config.artifacts_dir)
        self.snapshot: GraphSnapshot | None = None
        self.retriever: HybridRetriever | None = None

    def ingest(self) -> GraphSnapshot:
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
        self.snapshot = snapshot
        self.retriever = HybridRetriever(chunks, entities, relations)
        return snapshot

    def load(self) -> GraphSnapshot:
        self.snapshot = self.graph_store.load()
        self.retriever = HybridRetriever(self.snapshot.chunks, self.snapshot.entities, self.snapshot.relations)
        return self.snapshot

    def ensure_loaded(self) -> None:
        if self.snapshot is None or self.retriever is None:
            self.load()

    def corpus_summary(self) -> dict[str, object]:
        self.ensure_loaded()
        assert self.snapshot is not None
        return {
            "documents": len(self.snapshot.documents),
            "chunks": len(self.snapshot.chunks),
            "entities": len(self.snapshot.entities),
            "relations": len(self.snapshot.relations),
            "top_entities": [asdict(entity) for entity in self.snapshot.entities[:10]],
        }

    def answer(self, question: str) -> AnswerPayload:
        self.ensure_loaded()
        assert self.retriever is not None
        hits = self.retriever.retrieve(question, limit=self.config.max_evidence_chunks)
        graph_context = self.retriever.graph_context(question, max_neighbors=self.config.max_graph_neighbors)

        evidence = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "score": round(hit.score, 4),
                "text": hit.chunk.text,
            }
            for hit in hits
        ]

        if not evidence:
            answer = "No grounded answer could be produced from the indexed corpus."
        else:
            supporting = " ".join(item["text"] for item in evidence[:2])
            answer = f"Grounded answer draft: {supporting[:800].strip()}"

        return AnswerPayload(
            answer=answer,
            question=question,
            evidence=evidence,
            graph_context=graph_context,
        )

    def validate_paths(self) -> None:
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.config.input_dir}")
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

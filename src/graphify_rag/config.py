from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    chunk_size: int = 900
    chunk_overlap: int = 160
    max_sentences_per_chunk: int = 6
    min_entity_length: int = 3
    max_evidence_chunks: int = 5
    max_graph_neighbors: int = 8
    lexical_candidate_count: int = 10
    dense_candidate_count: int = 10
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_guardrail_model: str = os.getenv("OPENAI_GUARDRAIL_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    use_openai_generation: bool = os.getenv("USE_OPENAI_GENERATION", "true").lower() == "true"
    use_openai_embeddings: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"
    use_openai_guardrails: bool = os.getenv("USE_OPENAI_GUARDRAILS", "true").lower() == "true"
    prefer_graphify: bool = os.getenv("PREFER_GRAPHIFY", "true").lower() == "true"
    max_guardrail_loops: int = 1
    graph_chunk_boost: float = 0.12
    lexical_weight: float = 0.65
    dense_weight: float = 0.35

from __future__ import annotations

from dataclasses import dataclass
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

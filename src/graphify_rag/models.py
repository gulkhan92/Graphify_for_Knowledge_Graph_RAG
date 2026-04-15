from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    path: Path
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    token_count: int


@dataclass(slots=True)
class Entity:
    entity_id: str
    name: str
    label: str
    aliases: list[str] = field(default_factory=list)
    frequency: int = 1
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Relation:
    source: str
    target: str
    relation: str
    weight: float
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphSnapshot:
    documents: list[Document]
    chunks: list[Chunk]
    entities: list[Entity]
    relations: list[Relation]


@dataclass(slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass(slots=True)
class AnswerPayload:
    answer: str
    question: str
    evidence: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]

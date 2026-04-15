from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from graphify_rag.models import Chunk, Document, Entity, GraphSnapshot, Relation
from graphify_rag.utils import read_json, write_json


class GraphStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.snapshot_path = artifacts_dir / "graph_snapshot.json"

    def save(self, snapshot: GraphSnapshot) -> None:
        payload = {
            "documents": [document.to_dict() for document in snapshot.documents],
            "chunks": [asdict(chunk) for chunk in snapshot.chunks],
            "entities": [asdict(entity) for entity in snapshot.entities],
            "relations": [asdict(relation) for relation in snapshot.relations],
        }
        write_json(self.snapshot_path, payload)

    def load(self) -> GraphSnapshot:
        raw = read_json(self.snapshot_path)
        assert isinstance(raw, dict)
        documents = [
            Document(
                doc_id=item["doc_id"],
                title=item["title"],
                path=Path(item["path"]),
                content=item["content"],
                metadata=item.get("metadata", {}),
            )
            for item in raw["documents"]
        ]
        chunks = [Chunk(**item) for item in raw["chunks"]]
        entities = [Entity(**item) for item in raw["entities"]]
        relations = [Relation(**item) for item in raw["relations"]]
        return GraphSnapshot(documents=documents, chunks=chunks, entities=entities, relations=relations)

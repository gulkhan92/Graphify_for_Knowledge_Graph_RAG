from __future__ import annotations

from pathlib import Path

from graphify_rag.utils import read_json, write_json


class VectorStore:
    def __init__(self, artifacts_dir: Path) -> None:
        self.path = artifacts_dir / "chunk_embeddings.json"

    def save(self, embeddings: dict[str, list[float]]) -> None:
        write_json(self.path, embeddings)

    def load(self) -> dict[str, list[float]]:
        raw = read_json(self.path)
        assert isinstance(raw, dict)
        return {
            key: [float(value) for value in values]
            for key, values in raw.items()
            if isinstance(values, list)
        }

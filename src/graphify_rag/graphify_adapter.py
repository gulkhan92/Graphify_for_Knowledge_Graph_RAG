from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from graphify_rag.logging_utils import get_logger
from graphify_rag.models import Chunk, Document, Entity, GraphSnapshot, Relation
from graphify_rag.utils import slugify, tokenize, write_json


LOGGER = get_logger(__name__)


class GraphifyError(RuntimeError):
    pass


class GraphifyAdapter:
    def __init__(self, input_dir: Path, artifacts_dir: Path) -> None:
        self.input_dir = input_dir
        self.artifacts_dir = artifacts_dir
        self.graphify_output_dir = artifacts_dir / "graphify-out"

    def is_available(self) -> bool:
        return shutil.which("graphify") is not None

    def build_snapshot(self) -> GraphSnapshot:
        if not self.is_available():
            raise GraphifyError("Graphify CLI is not installed.")

        self.graphify_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "graphify",
            str(self.input_dir),
            "--no-viz",
        ]
        LOGGER.info("Running Graphify command: %s", " ".join(command))
        try:
            subprocess.run(
                command,
                check=True,
                cwd=self.artifacts_dir,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise GraphifyError(f"Graphify execution failed: {exc.stderr}") from exc

        graph_json = self._locate_graph_json()
        if graph_json is None:
            raise GraphifyError("Graphify did not produce graph.json.")
        snapshot = self._parse_graph_json(graph_json)
        self._write_graphify_manifest(graph_json)
        return snapshot

    def _locate_graph_json(self) -> Path | None:
        candidates = [
            self.artifacts_dir / "graphify-out" / "graph.json",
            self.artifacts_dir / "graph.json",
            self.input_dir / "graphify-out" / "graph.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _parse_graph_json(self, path: Path) -> GraphSnapshot:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "nodes" in payload and ("links" in payload or "edges" in payload):
            return self._from_node_link(payload)
        raise GraphifyError("Unsupported Graphify graph.json format.")

    def _from_node_link(self, payload: dict[str, Any]) -> GraphSnapshot:
        node_items = payload.get("nodes", [])
        edge_items = payload.get("links", payload.get("edges", []))
        if not isinstance(node_items, list) or not isinstance(edge_items, list):
            raise GraphifyError("Graphify graph.json nodes/edges payload is malformed.")

        documents: list[Document] = []
        chunks: list[Chunk] = []
        entities: list[Entity] = []
        relations: list[Relation] = []

        node_id_to_name: dict[str, str] = {}
        path_to_doc_id: dict[str, str] = {}

        for raw_node in node_items:
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_node.get("id", raw_node.get("name", "node")))
            label = str(raw_node.get("label", raw_node.get("type", "ENTITY")))
            name = str(raw_node.get("name", raw_node.get("title", node_id)))
            node_id_to_name[node_id] = name
            file_path = raw_node.get("path") or raw_node.get("file") or raw_node.get("source_path")
            summary_text = str(raw_node.get("summary", raw_node.get("text", raw_node.get("description", name))))

            if file_path:
                file_path_str = str(file_path)
                doc_id = slugify(Path(file_path_str).stem)
                if file_path_str not in path_to_doc_id:
                    path_to_doc_id[file_path_str] = doc_id
                    documents.append(
                        Document(
                            doc_id=doc_id,
                            title=Path(file_path_str).name,
                            path=Path(file_path_str),
                            content=summary_text,
                            metadata={"source_type": "graphify"},
                        )
                    )
                chunk_id = f"{doc_id}-graphify-node-{slugify(node_id)}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=summary_text,
                        index=len(chunks),
                        token_count=len(tokenize(summary_text)),
                    )
                )

            entities.append(
                Entity(
                    entity_id=slugify(node_id),
                    name=name,
                    label=label,
                    frequency=1,
                    chunk_ids=[chunks[-1].chunk_id] if file_path and chunks else [],
                )
            )

        for raw_edge in edge_items:
            if not isinstance(raw_edge, dict):
                continue
            source = str(raw_edge.get("source", raw_edge.get("from", "")))
            target = str(raw_edge.get("target", raw_edge.get("to", "")))
            if not source or not target:
                continue
            relation_name = str(raw_edge.get("label", raw_edge.get("type", "related_to")))
            weight = float(raw_edge.get("weight", raw_edge.get("confidence", 1.0)))
            relations.append(
                Relation(
                    source=slugify(source),
                    target=slugify(target),
                    relation=relation_name,
                    weight=weight,
                    evidence_chunk_ids=[],
                )
            )

        if not documents:
            documents.append(
                Document(
                    doc_id="graphify-corpus",
                    title="Graphify Corpus",
                    path=self.input_dir,
                    content="Graphify graph export",
                    metadata={"source_type": "graphify"},
                )
            )

        return GraphSnapshot(documents=documents, chunks=chunks, entities=entities, relations=relations)

    def _write_graphify_manifest(self, graph_json: Path) -> None:
        manifest = {
            "provider": "graphify",
            "graph_json": str(graph_json),
        }
        write_json(self.artifacts_dir / "graphify_manifest.json", manifest)

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

from graphify_rag.config import PipelineConfig
from graphify_rag.service import GraphRagService

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("FastAPI is required to run the API. Install project dependencies first.") from exc


def create_app() -> FastAPI:
    config = PipelineConfig(input_dir=Path("data"), artifacts_dir=Path("artifacts"))
    service = GraphRagService(config)
    app = FastAPI(title="Graphify Knowledge Graph RAG API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        try:
            return service.corpus_summary()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/ingest")
    def ingest() -> dict[str, object]:
        service.validate_paths()
        service.ingest()
        return service.corpus_summary()

    @app.get("/api/ask")
    def ask(question: str) -> dict[str, object]:
        try:
            payload = service.answer(question)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return asdict(payload)

    return app

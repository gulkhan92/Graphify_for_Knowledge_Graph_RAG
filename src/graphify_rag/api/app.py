from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from graphify_rag.logging_utils import get_logger
from graphify_rag.monitoring import MetricsRegistry, Timer
from graphify_rag.config import PipelineConfig
from graphify_rag.service import GraphRagService

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("FastAPI is required to run the API. Install project dependencies first.") from exc


LOGGER = get_logger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3)


def create_app() -> FastAPI:
    config = PipelineConfig(input_dir=Path("data"), artifacts_dir=Path("artifacts"))
    service = GraphRagService(config)
    metrics = MetricsRegistry()
    app = FastAPI(title="Graphify Knowledge Graph RAG API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request, call_next):  # type: ignore[no-untyped-def]
        with Timer() as timer:
            response = await call_next(request)
        metrics.record_request(timer.elapsed)
        LOGGER.info("%s %s status=%s duration=%.4fs", request.method, request.url.path, response.status_code, timer.elapsed)
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def get_metrics() -> dict[str, float | int]:
        return metrics.snapshot()

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
        metrics.record_ingest()
        return service.corpus_summary()

    @app.get("/api/ask")
    def ask(question: str) -> dict[str, object]:
        try:
            payload = service.answer(question)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        metrics.record_question()
        return asdict(payload)

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> dict[str, object]:
        try:
            response = service.answer(payload.question)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        metrics.record_question()
        return asdict(response)

    return app

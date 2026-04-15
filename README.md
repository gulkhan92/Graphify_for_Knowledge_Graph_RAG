# Graphify Knowledge Graph RAG

Production-oriented knowledge graph and retrieval-augmented generation pipeline for PDF corpora. The project ingests PDFs from `data/`, extracts entities and relations into a graph, builds lexical retrieval indexes, and exposes the system through both a Python CLI and a React/FastAPI application surface.

## Architecture

- `src/graphify_rag/`: backend package with ingestion, graph building, retrieval, service layer, and API app.
- `frontend/`: React + TypeScript UI for corpus stats, graph summaries, and question answering.
- `tests/`: backend unit tests covering chunking, extraction, retrieval, graph persistence, and service orchestration.
- `artifacts/`: generated indexes and graph snapshots after ingestion runs.

## Backend capabilities

- PDF extraction via `pdftotext` subprocess integration.
- Deterministic chunking with overlap.
- Entity extraction using heuristic NER patterns tuned for technical PDF content.
- Relation extraction using sentence co-occurrence and contextual linking.
- Hybrid retrieval across chunks, entities, and graph neighborhoods.
- Answer synthesis that grounds answers in retrieved evidence.
- FastAPI app factory in `src/graphify_rag/api/app.py`.
- CLI entrypoint in `main.py`.

## Frontend capabilities

- Corpus summary dashboard.
- Graph insights panels.
- QA workflow with evidence rendering.
- Ingestion trigger wired to the backend contract.

## Quick start

1. Create a virtual environment and install backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

2. Install frontend dependencies:

```bash
cd frontend
npm install
```

3. Build the knowledge graph from PDFs in `data/`:

```bash
python main.py ingest --input-dir data --artifacts-dir artifacts
```

4. Run the API:

```bash
uvicorn graphify_rag.api.app:create_app --factory --reload
```

5. Run the frontend:

```bash
cd frontend
npm run dev
```

## Notes

- The backend core is intentionally dependency-light, which keeps tests fast and deterministic.
- `pdftotext` must be available on the host machine for ingestion from PDFs.
- The API/UI layers declare their own dependencies in project metadata.

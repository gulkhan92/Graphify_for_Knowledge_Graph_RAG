from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from graphify_rag.config import PipelineConfig
from graphify_rag.service import GraphRagService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graphify knowledge graph RAG CLI")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--input-dir", default="data")
    shared.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build graph artifacts from the PDF corpus", parents=[shared])

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the stored graph", parents=[shared])
    ask_parser.add_argument("question")

    subparsers.add_parser("summary", help="Print corpus summary", parents=[shared])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_dir = getattr(args, "input_dir", "data")
    artifacts_dir = getattr(args, "artifacts_dir", "artifacts")
    config = PipelineConfig(
        input_dir=Path(input_dir),
        artifacts_dir=Path(artifacts_dir),
    )
    service = GraphRagService(config)
    service.validate_paths()

    if args.command == "ingest":
        snapshot = service.ingest()
        print(json.dumps(service.corpus_summary(), indent=2))
        print(f"Ingested {len(snapshot.documents)} documents into {config.artifacts_dir}")
        return 0

    if args.command == "summary":
        print(json.dumps(service.corpus_summary(), indent=2))
        return 0

    if args.command == "ask":
        payload = service.answer(args.question)
        print(json.dumps(asdict(payload), indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2

from __future__ import annotations

from graphify_rag.models import Chunk, Document, Entity


def validate_documents(documents: list[Document]) -> None:
    if not documents:
        raise ValueError("No documents were loaded from the input directory.")
    for document in documents:
        if not document.content.strip():
            raise ValueError(f"Document {document.path} is empty after extraction.")


def validate_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        raise ValueError("Chunking produced no output.")
    for chunk in chunks:
        if not chunk.text.strip():
            raise ValueError(f"Chunk {chunk.chunk_id} is empty.")


def validate_entities(entities: list[Entity]) -> None:
    if not entities:
        raise ValueError("Entity extraction produced no entities.")

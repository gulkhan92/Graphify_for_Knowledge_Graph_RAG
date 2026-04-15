from __future__ import annotations

from graphify_rag.models import Chunk, Document
from graphify_rag.utils import split_sentences, tokenize


def chunk_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    max_sentences_per_chunk: int,
) -> list[Chunk]:
    sentences = split_sentences(document.content)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_sentences and (
            current_len + sentence_len > chunk_size or len(current_sentences) >= max_sentences_per_chunk
        ):
            text = " ".join(current_sentences).strip()
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                    doc_id=document.doc_id,
                    text=text,
                    index=chunk_index,
                    token_count=len(tokenize(text)),
                )
            )
            chunk_index += 1
            overlap_text = text[-chunk_overlap:].strip()
            current_sentences = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current_sentences.append(sentence)
        current_len += sentence_len

    if current_sentences:
        text = " ".join(current_sentences).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}-chunk-{chunk_index}",
                doc_id=document.doc_id,
                text=text,
                index=chunk_index,
                token_count=len(tokenize(text)),
            )
        )
    return chunks

from __future__ import annotations

from graphify_rag.models import ChatTurn


SYSTEM_PROMPT = """You are a knowledge-graph-grounded assistant.
Answer only from the supplied evidence and graph context.
If the evidence is insufficient, say so directly.
Prefer concise, technically precise answers with short bullet points when useful."""


def build_chat_turn(question: str, evidence: list[dict[str, object]], graph_context: list[dict[str, object]]) -> ChatTurn:
    evidence_text = "\n\n".join(
        f"[{index + 1}] ({item['doc_id']}::{item['chunk_id']}, score={item['score']})\n{item['text']}"
        for index, item in enumerate(evidence)
    )
    graph_text = "\n".join(
        f"- {item['source']} {item['relation']} {item['target']} (weight={item['weight']})"
        for item in graph_context
    )
    return ChatTurn(
        role="user",
        content=(
            f"Question:\n{question}\n\n"
            f"Retrieved evidence:\n{evidence_text or 'No evidence retrieved.'}\n\n"
            f"Knowledge graph context:\n{graph_text or 'No graph context matched.'}\n\n"
            "Produce a grounded answer and do not invent facts outside the supplied context."
        ),
    )

from __future__ import annotations

import json

from graphify_rag.models import ChatTurn


SYSTEM_PROMPT = """You are a knowledge-graph-grounded assistant.
Answer only from the supplied evidence and graph context.
If the evidence is insufficient, say so directly.
Prefer concise, technically precise answers with short bullet points when useful."""

GUARDRAIL_SYSTEM_PROMPT = """You are a strict answer validation agent.
Evaluate whether the candidate answer is fully supported by the supplied evidence and graph context.
Return JSON only with keys:
- verdict: PASS or FAIL
- issues: array of short strings describing missing, unsupported, or inaccurate claims
- revised_requirements: array of short instructions for a regeneration step
Do not include markdown fences or extra prose."""


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


def build_regeneration_turn(
    question: str,
    evidence: list[dict[str, object]],
    graph_context: list[dict[str, object]],
    feedback: list[str],
) -> ChatTurn:
    base_turn = build_chat_turn(question, evidence, graph_context)
    feedback_text = "\n".join(f"- {item}" for item in feedback)
    return ChatTurn(
        role="user",
        content=f"{base_turn.content}\n\nValidator feedback to address before answering again:\n{feedback_text}",
    )


def build_guardrail_turn(
    question: str,
    evidence: list[dict[str, object]],
    graph_context: list[dict[str, object]],
    candidate_answer: str,
) -> ChatTurn:
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
            f"Evidence:\n{evidence_text or 'No evidence retrieved.'}\n\n"
            f"Graph context:\n{graph_text or 'No graph context matched.'}\n\n"
            f"Candidate answer:\n{candidate_answer}\n\n"
            "Validate the answer against the evidence and return the required JSON."
        ),
    )


def parse_guardrail_payload(payload: str) -> dict[str, object]:
    parsed = json.loads(payload)
    verdict = str(parsed.get("verdict", "FAIL")).upper()
    issues = [str(item) for item in parsed.get("issues", []) if str(item).strip()]
    revised_requirements = [str(item) for item in parsed.get("revised_requirements", []) if str(item).strip()]
    return {
        "verdict": verdict if verdict in {"PASS", "FAIL"} else "FAIL",
        "issues": issues,
        "revised_requirements": revised_requirements,
    }

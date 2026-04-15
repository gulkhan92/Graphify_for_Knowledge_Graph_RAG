from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "item"


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def split_sentences(text: str) -> list[str]:
    segments = [segment.strip() for segment in SENTENCE_PATTERN.split(text.strip()) if segment.strip()]
    return segments or ([text.strip()] if text.strip() else [])


def term_frequency(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))

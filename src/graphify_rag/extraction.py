from __future__ import annotations

import re
from collections import defaultdict

from graphify_rag.models import Chunk, Entity, Relation
from graphify_rag.utils import slugify, split_sentences


TITLE_CASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z0-9\-]+)+)\b")
ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")
MIXED_TECH_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

STOP_ENTITY_TERMS = {
    "Abstract",
    "Introduction",
    "Keywords",
    "Figure",
    "Table",
}


def classify_entity(name: str) -> str:
    if YEAR_PATTERN.fullmatch(name):
        return "DATE"
    if name.isupper():
        return "ACRONYM"
    if any(token in name for token in ("Guidebook", "Framework", "Infrastructure", "System")):
        return "CONCEPT"
    return "TOPIC"


def extract_entities(chunks: list[Chunk], min_entity_length: int) -> list[Entity]:
    entity_map: dict[str, Entity] = {}

    for chunk in chunks:
        matches = set(TITLE_CASE_PATTERN.findall(chunk.text)) | set(ACRONYM_PATTERN.findall(chunk.text)) | set(
            MIXED_TECH_PATTERN.findall(chunk.text)
        )
        for match in matches:
            name = match.strip()
            if len(name) < min_entity_length or name in STOP_ENTITY_TERMS:
                continue
            entity_id = slugify(name)
            entity = entity_map.get(entity_id)
            if entity is None:
                entity = Entity(entity_id=entity_id, name=name, label=classify_entity(name))
                entity_map[entity_id] = entity
            entity.frequency += 1
            if chunk.chunk_id not in entity.chunk_ids:
                entity.chunk_ids.append(chunk.chunk_id)

    return sorted(entity_map.values(), key=lambda item: (-item.frequency, item.name))


def infer_relation(sentence: str) -> str:
    lowered = sentence.lower()
    if "introduce" in lowered or "present" in lowered:
        return "introduces"
    if "support" in lowered or "supports" in lowered:
        return "supports"
    if "integrate" in lowered or "unifies" in lowered:
        return "integrates"
    if "based on" in lowered:
        return "depends_on"
    return "related_to"


def extract_relations(chunks: list[Chunk], entities: list[Entity]) -> list[Relation]:
    entity_names = {entity.name: entity.entity_id for entity in entities}
    relation_weights: dict[tuple[str, str, str], float] = defaultdict(float)
    evidence_map: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            present = [entity_id for name, entity_id in entity_names.items() if name in sentence]
            if len(present) < 2:
                continue
            relation_name = infer_relation(sentence)
            for index, source in enumerate(present):
                for target in present[index + 1 :]:
                    key = (source, target, relation_name)
                    relation_weights[key] += 1.0
                    evidence_map[key].add(chunk.chunk_id)

    relations = [
        Relation(
            source=source,
            target=target,
            relation=relation,
            weight=weight,
            evidence_chunk_ids=sorted(evidence_map[(source, target, relation)]),
        )
        for (source, target, relation), weight in relation_weights.items()
    ]
    return sorted(relations, key=lambda item: (-item.weight, item.source, item.target, item.relation))

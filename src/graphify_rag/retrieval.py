from __future__ import annotations

from collections import Counter, defaultdict

from graphify_rag.models import Chunk, Entity, Relation, RetrievalResult
from graphify_rag.utils import cosine_similarity, term_frequency, tokenize


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], entities: list[Entity], relations: list[Relation]) -> None:
        self.chunks = chunks
        self.entities = entities
        self.relations = relations
        self.chunk_vectors = {chunk.chunk_id: term_frequency(tokenize(chunk.text)) for chunk in chunks}
        self.chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self.entity_map = {entity.entity_id: entity for entity in entities}
        self.entity_lookup = {entity.name.lower(): entity.entity_id for entity in entities}
        self.neighbors = self._build_neighbors(relations)

    def _build_neighbors(self, relations: list[Relation]) -> dict[str, list[Relation]]:
        adjacency: dict[str, list[Relation]] = defaultdict(list)
        for relation in relations:
            adjacency[relation.source].append(relation)
            adjacency[relation.target].append(relation)
        return dict(adjacency)

    def retrieve(self, query: str, limit: int = 5) -> list[RetrievalResult]:
        query_vector = term_frequency(tokenize(query))
        scores: list[RetrievalResult] = []
        hinted_entity_ids = {
            entity_id
            for name, entity_id in self.entity_lookup.items()
            if name in query.lower()
        }

        for chunk in self.chunks:
            score = cosine_similarity(query_vector, self.chunk_vectors[chunk.chunk_id])
            if hinted_entity_ids and any(chunk.chunk_id in self.entity_map[entity_id].chunk_ids for entity_id in hinted_entity_ids):
                score += 0.15
            if score > 0:
                scores.append(RetrievalResult(chunk=chunk, score=score))

        return sorted(scores, key=lambda item: item.score, reverse=True)[:limit]

    def graph_context(self, query: str, max_neighbors: int) -> list[dict[str, object]]:
        contexts: list[dict[str, object]] = []
        seen: set[str] = set()
        for entity in self.entities:
            if entity.name.lower() not in query.lower():
                continue
            for relation in sorted(self.neighbors.get(entity.entity_id, []), key=lambda item: item.weight, reverse=True):
                relation_key = f"{relation.source}:{relation.target}:{relation.relation}"
                if relation_key in seen:
                    continue
                seen.add(relation_key)
                contexts.append(
                    {
                        "source": self.entity_map.get(relation.source, Entity(relation.source, relation.source, "UNKNOWN")).name,
                        "target": self.entity_map.get(relation.target, Entity(relation.target, relation.target, "UNKNOWN")).name,
                        "relation": relation.relation,
                        "weight": relation.weight,
                    }
                )
                if len(contexts) >= max_neighbors:
                    return contexts
        return contexts

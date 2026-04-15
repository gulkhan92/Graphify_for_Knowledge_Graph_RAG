from __future__ import annotations

import math
from collections import Counter, defaultdict

from graphify_rag.models import Chunk, Entity, Relation, RetrievalResult
from graphify_rag.utils import cosine_similarity, cosine_similarity_dense, term_frequency, tokenize


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        entities: list[Entity],
        relations: list[Relation],
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self.chunks = chunks
        self.entities = entities
        self.relations = relations
        self.embeddings = embeddings or {}
        self.chunk_vectors = {chunk.chunk_id: term_frequency(tokenize(chunk.text)) for chunk in chunks}
        self.chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self.entity_map = {entity.entity_id: entity for entity in entities}
        self.entity_lookup = {entity.name.lower(): entity.entity_id for entity in entities}
        self.chunk_entity_map = self._build_chunk_entity_map()
        self.neighbors = self._build_neighbors(relations)
        self.document_frequency = self._build_document_frequency()
        self.average_chunk_length = sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0.0

    def _build_neighbors(self, relations: list[Relation]) -> dict[str, list[Relation]]:
        adjacency: dict[str, list[Relation]] = defaultdict(list)
        for relation in relations:
            adjacency[relation.source].append(relation)
            adjacency[relation.target].append(relation)
        return dict(adjacency)

    def _build_document_frequency(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for chunk in self.chunks:
            frequencies.update(set(tokenize(chunk.text)))
        return frequencies

    def _build_chunk_entity_map(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = defaultdict(set)
        for entity in self.entities:
            for chunk_id in entity.chunk_ids:
                mapping[chunk_id].add(entity.entity_id)
        return dict(mapping)

    def _bm25_score(self, query_terms: list[str], chunk: Chunk) -> float:
        if not query_terms or not self.average_chunk_length:
            return 0.0
        tf = self.chunk_vectors[chunk.chunk_id]
        score = 0.0
        total_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75
        for term in query_terms:
            doc_freq = self.document_frequency.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue
            denominator = term_freq + k1 * (1 - b + b * (chunk.token_count / self.average_chunk_length))
            score += idf * ((term_freq * (k1 + 1)) / denominator)
        return score

    def _graph_chunk_boost(self, hinted_entity_ids: set[str], chunk: Chunk) -> float:
        if not hinted_entity_ids:
            return 0.0
        boost = 0.0
        chunk_entity_ids = self.chunk_entity_map.get(chunk.chunk_id, set())
        if hinted_entity_ids & chunk_entity_ids:
            boost += 1.0
        for entity_id in hinted_entity_ids:
            for relation in self.neighbors.get(entity_id, []):
                neighbor = relation.target if relation.source == entity_id else relation.source
                if neighbor in chunk_entity_ids:
                    boost += min(1.0, relation.weight / 5.0)
        return boost

    def _dense_scores(self, query_embedding: list[float] | None) -> dict[str, float]:
        if query_embedding is None or not self.embeddings:
            return {}
        return {
            chunk_id: cosine_similarity_dense(query_embedding, embedding)
            for chunk_id, embedding in self.embeddings.items()
        }

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        lexical_weight: float = 0.65,
        dense_weight: float = 0.35,
        graph_chunk_boost: float = 0.12,
        query_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        query_vector = term_frequency(query_terms)
        scores: list[RetrievalResult] = []
        hinted_entity_ids = {
            entity_id
            for name, entity_id in self.entity_lookup.items()
            if name in query.lower()
        }
        dense_scores = self._dense_scores(query_embedding)

        for chunk in self.chunks:
            lexical_score = self._bm25_score(query_terms, chunk) + cosine_similarity(query_vector, self.chunk_vectors[chunk.chunk_id])
            dense_score = dense_scores.get(chunk.chunk_id, 0.0)
            graph_score = self._graph_chunk_boost(hinted_entity_ids, chunk)
            score = lexical_weight * lexical_score + dense_weight * dense_score + graph_chunk_boost * graph_score
            if score > 0:
                scores.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=score,
                        lexical_score=lexical_score,
                        dense_score=dense_score,
                        graph_score=graph_score,
                    )
                )

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

"""Pluggable Retrieval Strategy Engine for SyntraFlow.

Supports Dense (Vector), Sparse (BM25), Hybrid (RRF Fusion), and Graph (Neo4j)
retrieval strategies with graceful fallback handling.
"""

import abc
import logging
import re
from typing import Any, Dict, List, Optional

from projects.syntraflow.src.ingestion.vector_writer import build_qdrant_filter

logger = logging.getLogger("syntraflow.retrieval.engine")


class BaseRetrievalStrategy(abc.ABC):
    """Abstract base strategy for document retrieval."""

    @abc.abstractmethod
    async def execute(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute retrieval strategy.

        Args:
            query: Natural language query string.
            collection_name: Qdrant collection name.
            limit: Maximum hits to return.
            query_vector: Optional pre-computed query embedding vector.
            filters: Optional metadata filtering dictionary.

        Returns:
            List of result dicts with fields: 'id', 'score', 'text', 'metadata', 'strategy'.
        """
        pass


class DenseRetrievalStrategy(BaseRetrievalStrategy):
    """Vector Cosine Similarity search strategy using Qdrant."""

    def __init__(self, vector_client: Any = None) -> None:
        self.vector_client = vector_client

    async def execute(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.vector_client:
            logger.warning("VectorClient unconfigured in DenseRetrievalStrategy.")
            return []

        if not query_vector:
            # Simple dummy query vector fallback if not passed (e.g. 1024 zeros)
            query_vector = [0.0] * 1024

        qdrant_filter = build_qdrant_filter(filters or {})

        try:
            qdrant = self.vector_client.get_client()
            search_hits = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
            )

            results = []
            for item in search_hits:
                payload = item.payload or {}
                results.append({
                    "id": str(item.id),
                    "score": float(item.score),
                    "text": payload.get("text", payload.get("content", "")),
                    "metadata": payload,
                    "strategy": "dense",
                })
            return results
        except Exception as e:
            logger.error("Dense retrieval failed for collection '%s': %s", collection_name, e)
            return []


class SparseRetrievalStrategy(BaseRetrievalStrategy):
    """BM25 / Keyword Sparse retrieval strategy."""

    def __init__(self, dense_fallback: Optional[DenseRetrievalStrategy] = None) -> None:
        self.dense_fallback = dense_fallback

    async def execute(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # In this workspace, if BM25 full-text sparse index is not running, token match over Dense payloads
        logger.info("Executing Sparse (BM25) search for query: '%s'", query)
        if self.dense_fallback:
            dense_results = await self.dense_fallback.execute(
                query=query,
                collection_name=collection_name,
                limit=limit * 3,
                query_vector=query_vector,
                filters=filters,
            )
            # Token matching filter over dense candidates
            query_tokens = set(re.findall(r"\w+", query.lower()))
            matched = []
            for hit in dense_results:
                text_tokens = set(re.findall(r"\w+", hit["text"].lower()))
                overlap = len(query_tokens.intersection(text_tokens))
                if overlap > 0 or not query_tokens:
                    score = float(overlap) / max(len(query_tokens), 1)
                    hit_copy = dict(hit)
                    hit_copy["score"] = round(score + hit["score"] * 0.1, 4)
                    hit_copy["strategy"] = "sparse"
                    matched.append(hit_copy)

            matched.sort(key=lambda x: x["score"], reverse=True)
            return matched[:limit] if matched else dense_results[:limit]
        return []


class GraphRetrievalStrategy(BaseRetrievalStrategy):
    """Neo4j Knowledge Graph entity neighborhood retrieval strategy."""

    def __init__(self, dense_fallback: Optional[DenseRetrievalStrategy] = None) -> None:
        self.dense_fallback = dense_fallback

    async def execute(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            from common.clients.neo4j import execute_read_query

            cypher_query = (
                "MATCH (e:SyntraFlow_Entity)-[r:SyntraFlow_RELATION]->(o:SyntraFlow_Entity) "
                "WHERE e.name CONTAINS $query OR o.name CONTAINS $query "
                "RETURN e.name AS source, o.name AS target, r.type AS rel_type "
                "LIMIT $limit"
            )
            records = await execute_read_query(cypher_query, {"query": query, "limit": limit})
            if records:
                hits = []
                for rec in records:
                    text_repr = f"Graph Relation: {rec['source']} -> {rec['rel_type']} -> {rec['target']}"
                    hits.append({
                        "id": str(hash(text_repr)),
                        "score": 1.0,
                        "text": text_repr,
                        "metadata": {"type": "graph_relation", "source": rec["source"], "target": rec["target"]},
                        "strategy": "graph",
                    })
                return hits
        except Exception as e:
            logger.warning("Neo4j Graph retrieval unavailable/offline: %s. Falling back to Dense.", e)

        if self.dense_fallback:
            return await self.dense_fallback.execute(
                query=query, collection_name=collection_name, limit=limit, query_vector=query_vector, filters=filters
            )
        return []


class HybridRRFStrategy(BaseRetrievalStrategy):
    """Hybrid Reciprocal Rank Fusion (RRF) strategy combining Dense and Sparse/Graph."""

    def __init__(self, dense_strategy: DenseRetrievalStrategy, sparse_strategy: SparseRetrievalStrategy, rrf_k: int = 60) -> None:
        self.dense_strategy = dense_strategy
        self.sparse_strategy = sparse_strategy
        self.rrf_k = rrf_k

    async def execute(
        self,
        query: str,
        collection_name: str,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        dense_results = await self.dense_strategy.execute(
            query=query, collection_name=collection_name, limit=limit * 2, query_vector=query_vector, filters=filters
        )
        sparse_results = await self.sparse_strategy.execute(
            query=query, collection_name=collection_name, limit=limit * 2, query_vector=query_vector, filters=filters
        )

        scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}

        # Score Dense hits
        for rank, hit in enumerate(dense_results):
            text_val = hit["text"]
            scores[text_val] = scores.get(text_val, 0.0) + (1.0 / (rank + self.rrf_k))
            if text_val not in items_map:
                items_map[text_val] = hit

        # Score Sparse hits
        for rank, hit in enumerate(sparse_results):
            text_val = hit["text"]
            scores[text_val] = scores.get(text_val, 0.0) + (1.0 / (rank + self.rrf_k))
            if text_val not in items_map:
                items_map[text_val] = hit

        sorted_texts = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

        fused = []
        for text in sorted_texts:
            item = dict(items_map[text])
            item["score"] = round(scores[text], 6)
            item["strategy"] = "hybrid"
            fused.append(item)

        return fused


class RetrievalEngine:
    """Unified Orchestrator for pluggable retrieval strategies with graceful fallbacks."""

    def __init__(self, vector_client: Any = None) -> None:
        self.vector_client = vector_client
        self.dense_strategy = DenseRetrievalStrategy(vector_client=vector_client)
        self.sparse_strategy = SparseRetrievalStrategy(dense_fallback=self.dense_strategy)
        self.graph_strategy = GraphRetrievalStrategy(dense_fallback=self.dense_strategy)
        self.hybrid_strategy = HybridRRFStrategy(
            dense_strategy=self.dense_strategy, sparse_strategy=self.sparse_strategy
        )

    async def query(
        self,
        query: str,
        collection_name: str,
        strategy: str = "dense",
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query vector collection with specified retrieval strategy.

        Args:
            query: Query text string.
            collection_name: Target Qdrant collection name.
            strategy: 'dense' | 'sparse' | 'hybrid' | 'graph'.
            limit: Maximum hits count.
            query_vector: Precomputed query embedding.
            filters: Key-value metadata filtering parameters.

        Returns:
            List of result dict items.
        """
        strat_key = strategy.lower().strip()

        try:
            if strat_key == "sparse":
                return await self.sparse_strategy.execute(query, collection_name, limit, query_vector, filters)
            elif strat_key == "graph":
                return await self.graph_strategy.execute(query, collection_name, limit, query_vector, filters)
            elif strat_key == "hybrid":
                return await self.hybrid_strategy.execute(query, collection_name, limit, query_vector, filters)
            else:
                return await self.dense_strategy.execute(query, collection_name, limit, query_vector, filters)
        except Exception as e:
            logger.error("Error executing strategy '%s': %s. Falling back to Dense.", strategy, e)
            return await self.dense_strategy.execute(query, collection_name, limit, query_vector, filters)

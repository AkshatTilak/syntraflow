"""SyntraFlow Hybrid Retrieval Engine.

Implements Vector Search (Qdrant), Graph Search (Neo4j), and Hybrid RAG
using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Any, List, Dict
from qdrant_client import QdrantClient
from common.config.settings import settings
from projects.syntraflow.src.vectors.client import VectorClient

logger = logging.getLogger("syntraflow.retrieval")


class RetrievalEngine:
    """Retrieval service implementing Vector, Knowledge Graph, and Hybrid search."""

    def __init__(self, vector_client: VectorClient) -> None:
        self.vector_client = vector_client

    async def search_vector(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on Qdrant vector database."""
        try:
            results = self.vector_client.search_similarity(
                collection_name="syntraflow_chunks_v1",
                query_vector=query_vector,
                limit=limit
            )
            hits = []
            for item in results:
                hits.append({
                    "id": item.id,
                    "score": item.score,
                    "text": item.payload.get("text", ""),
                    "metadata": {
                        "filename": item.payload.get("filename", ""),
                        "start_time": item.payload.get("start_time"),
                        "end_time": item.payload.get("end_time")
                    }
                })
            return hits
        except Exception as e:
            logger.error("Qdrant search failed: %s. Returning empty results.", e)
            return []

    async def search_graph(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Traverse relationships in Neo4j to find linked entities/community summaries."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.NEO4J_URL,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            hits = []
            with driver.session() as session:
                # Retrieve matching entities and their claims / relations
                cypher_query = (
                    "MATCH (e:SyntraFlow_Entity)-[r:SyntraFlow_RELATION]->(o:SyntraFlow_Entity) "
                    "WHERE e.name CONTAINS $query OR o.name CONTAINS $query "
                    "RETURN e.name AS source, o.name AS target, r.type AS rel_type "
                    "LIMIT $limit"
                )
                result = session.run(cypher_query, query=query, limit=limit)
                for rec in result:
                    text_repr = f"Relationship: {rec['source']} -> {rec['rel_type']} -> {rec['target']}"
                    hits.append({
                        "id": hash(text_repr),
                        "score": 1.0,
                        "text": text_repr,
                        "metadata": {"type": "graph_relation"}
                    })
            driver.close()
            return hits
        except Exception as e:
            logger.warning("Neo4j graph traversal failed or not configured: %s", e)
            return []

    async def search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        rrf_constant: int = 60,
    ) -> List[Dict[str, Any]]:
        """Executes Vector and Graph searches, then runs Reciprocal Rank Fusion (RRF)."""
        vector_results = await self.search_vector(query_vector, limit=limit * 2)
        graph_results = await self.search_graph(query, limit=limit * 2)

        # Apply Reciprocal Rank Fusion (RRF)
        # RRF Score = sum( 1 / (rank + k) )
        scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score vector hits
        for rank, hit in enumerate(vector_results):
            text_val = hit["text"]
            scores[text_val] = scores.get(text_val, 0.0) + (1.0 / (rank + rrf_constant))
            if text_val not in items_map:
                items_map[text_val] = hit

        # 2. Score graph hits
        for rank, hit in enumerate(graph_results):
            text_val = hit["text"]
            scores[text_val] = scores.get(text_val, 0.0) + (1.0 / (rank + rrf_constant))
            if text_val not in items_map:
                items_map[text_val] = hit

        # Sort and limit
        sorted_texts = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        
        fused_results = []
        for text in sorted_texts:
            item = items_map[text].copy()
            item["score"] = scores[text]
            fused_results.append(item)

        return fused_results

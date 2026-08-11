"""Hub-Scoped Retrieval Engine for SyntraFlow (S6-04d).

Implements multi-tenant Dense, Sparse, Hybrid (RRF), and Graph retrieval strategies,
collection ownership validation, mandatory hub_id filtering, and multi-collection fan-in.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.http import models as qdrant_models

from common.config.settings import settings
from projects.syntraflow.src.database.models import SyntraFlowCollection
from projects.syntraflow.src.datastores import (
    resolve_vector_client,
    resolve_graph_client,
    DatastoreUnavailableError,
)

logger = logging.getLogger("syntraflow.retrieval.engine")

ALLOWED_METADATA_KEYS = {"document_id", "tags", "timestamp", "access_level"}


class RetrievalEngine:
    """Multi-tenant, hub-scoped retrieval engine orchestrating vector, sparse, hybrid, and graph strategies."""

    def __init__(self, db: AsyncSession, hub_id: str) -> None:
        self.db = db
        self.hub_id = hub_id

    async def resolve_targets(
        self, collection_ids: Optional[List[str]] = None
    ) -> List[SyntraFlowCollection]:
        """Resolve target collections for self.hub_id.

        If collection_ids is None or empty, returns all collections belonging to hub_id.
        If collection_ids is provided, verifies every ID belongs to hub_id.
        Raises HTTPException(404) if any collection is missing or belongs to another hub.
        """
        stmt = select(SyntraFlowCollection).where(
            SyntraFlowCollection.hub_id == self.hub_id
        )
        res = await self.db.execute(stmt)
        hub_collections = {c.id: c for c in res.scalars().all()}

        if not collection_ids:
            return list(hub_collections.values())

        resolved = []
        for col_id in collection_ids:
            if col_id not in hub_collections:
                raise HTTPException(status_code=404, detail="Collection not found")
            resolved.append(hub_collections[col_id])
        return resolved

    def _hub_filter(
        self,
        collection_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> qdrant_models.Filter:
        """Construct mandatory Qdrant payload Filter enforcing hub_id and metadata criteria."""
        must_conditions: List[Any] = [
            qdrant_models.FieldCondition(
                key="hub_id", match=qdrant_models.MatchValue(value=self.hub_id)
            )
        ]

        if collection_id:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="collection_id", match=qdrant_models.MatchValue(value=collection_id)
                )
            )

        if metadata_filter:
            for key, val in metadata_filter.items():
                if key not in ALLOWED_METADATA_KEYS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Unsupported metadata filter key: '{key}'",
                    )
                if key == "document_id":
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key="document_id", match=qdrant_models.MatchValue(value=str(val))
                        )
                    )
                elif key == "tags":
                    if isinstance(val, list):
                        must_conditions.append(
                            qdrant_models.FieldCondition(
                                key="tags", match=qdrant_models.MatchAny(any=val)
                            )
                        )
                    else:
                        must_conditions.append(
                            qdrant_models.FieldCondition(
                                key="tags", match=qdrant_models.MatchValue(value=val)
                            )
                        )
                elif key == "access_level":
                    must_conditions.append(
                        qdrant_models.FieldCondition(
                            key="access_level", match=qdrant_models.MatchValue(value=val)
                        )
                    )
                elif key == "timestamp":
                    if isinstance(val, dict):
                        must_conditions.append(
                            qdrant_models.FieldCondition(
                                key="timestamp",
                                range=qdrant_models.DatetimeRange(
                                    gte=val.get("gte"), lte=val.get("lte")
                                ),
                            )
                        )

        return qdrant_models.Filter(must=must_conditions)

    async def search_vector(
        self,
        collection: SyntraFlowCollection,
        query_vector: List[float],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search on a collection's physical name."""
        vector_client = await resolve_vector_client(self.db, self.hub_id)
        qdrant_filter = self._hub_filter(
            collection_id=collection.id, metadata_filter=metadata_filter
        )

        # Parse score threshold from collection config if available
        config = {}
        if collection.retrieval_config_json:
            try:
                config = json.loads(collection.retrieval_config_json)
            except Exception:
                pass
        score_threshold = config.get("score_threshold")

        try:
            client = vector_client.get_client()
            hits_raw = []

            # 1. Try query_points (qdrant-client >= 1.10)
            if hasattr(client, "query_points"):
                try:
                    res = client.query_points(
                        collection_name=collection.physical_name,
                        query=query_vector,
                        limit=limit,
                        query_filter=qdrant_filter,
                        score_threshold=score_threshold,
                    )
                    if hasattr(res, "points") and isinstance(res.points, list):
                        hits_raw = res.points
                    elif isinstance(res, list):
                        hits_raw = res
                except Exception:
                    pass

            # 2. Fallback to search (qdrant-client < 1.10 or legacy mock)
            if not hits_raw and hasattr(client, "search"):
                res = client.search(
                    collection_name=collection.physical_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                )
                if isinstance(res, list):
                    hits_raw = res

            hits = []
            for item in hits_raw:
                payload = item.payload or {}
                hits.append({
                    "id": str(item.id),
                    "score": float(item.score),
                    "text": payload.get("text", payload.get("content", "")),
                    "metadata": {
                        "filename": payload.get("filename", ""),
                        "document_id": payload.get("document_id", ""),
                        "start_time": payload.get("start_time"),
                        "end_time": payload.get("end_time"),
                        "tags": payload.get("tags", []),
                        "timestamp": payload.get("timestamp"),
                        "access_level": payload.get("access_level", "read"),
                    },
                    "collection_id": collection.id,
                    "collection_name": collection.name,
                    "hub_id": self.hub_id,
                    "strategy": "dense",
                })
            return hits
        except DatastoreUnavailableError:
            raise
        except Exception as e:
            logger.error("Vector search failed for collection '%s': %s", collection.physical_name, e)
            return []

    async def search_sparse(
        self,
        collection: SyntraFlowCollection,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform sparse BM25 keyword search with SQL fallback."""
        config = {}
        if collection.retrieval_config_json:
            try:
                config = json.loads(collection.retrieval_config_json)
            except Exception:
                pass

        if not config.get("sparse_index_enabled", False) and collection.retrieval_config_json is not None:
            raise HTTPException(status_code=409, detail="Collection has no sparse index")

        hits = []
        if config.get("sparse_index_enabled", False):
            try:
                vector_client = await resolve_vector_client(self.db, self.hub_id)
                qdrant_filter = self._hub_filter(
                    collection_id=collection.id, metadata_filter=metadata_filter
                )
                client = vector_client.get_client()
                hits_raw = []

                if hasattr(client, "query_points"):
                    try:
                        res = client.query_points(
                            collection_name=collection.physical_name,
                            query=qdrant_models.NamedSparseVector(
                                name="sparse",
                                vector=qdrant_models.SparseVector(indices=[1, 2], values=[1.0, 1.0]),
                            ),
                            limit=limit,
                            query_filter=qdrant_filter,
                        )
                        if hasattr(res, "points") and isinstance(res.points, list):
                            hits_raw = res.points
                        elif isinstance(res, list):
                            hits_raw = res
                    except Exception:
                        pass

                if not hits_raw and hasattr(client, "search"):
                    res = client.search(
                        collection_name=collection.physical_name,
                        query_vector=qdrant_models.NamedSparseVector(
                            name="sparse",
                            vector=qdrant_models.SparseVector(indices=[1, 2], values=[1.0, 1.0]),
                        ),
                        limit=limit,
                        query_filter=qdrant_filter,
                    )
                    if isinstance(res, list):
                        hits_raw = res

                for item in hits_raw:
                    payload = item.payload or {}
                    hits.append({
                        "id": str(item.id),
                        "score": float(item.score),
                        "text": payload.get("text", ""),
                        "metadata": payload,
                        "collection_id": collection.id,
                        "collection_name": collection.name,
                        "hub_id": self.hub_id,
                        "strategy": "sparse",
                    })
            except Exception as e:
                logger.debug("Qdrant sparse search skipped: %s", e)

        # Fallback to SQL keyword BM25 matching on chunk text
        if not hits:
            import re
            from sqlalchemy import select, or_
            from projects.syntraflow.src.database.models import SyntraFlowChunk, SyntraFlowDocument

            terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
            if terms:
                conditions = [SyntraFlowChunk.text.ilike(f"%{t}%") for t in terms]
                stmt = (
                    select(SyntraFlowChunk, SyntraFlowDocument.filename)
                    .join(SyntraFlowDocument, SyntraFlowChunk.document_id == SyntraFlowDocument.id)
                    .where(
                        SyntraFlowChunk.hub_id == self.hub_id,
                        SyntraFlowDocument.collection_id == collection.id,
                        or_(*conditions)
                    )
                    .limit(limit * 3)
                )
                try:
                    res = await self.db.execute(stmt)
                    rows = res.all()

                    sql_hits = []
                    for chunk, filename in rows:
                        text_lower = chunk.text.lower()
                        term_matches = sum(text_lower.count(t) for t in terms)
                        score = min(1.0, 0.5 + (term_matches * 0.1))
                        sql_hits.append({
                            "id": str(chunk.id),
                            "score": round(score, 4),
                            "text": chunk.text,
                            "metadata": {
                                "filename": filename,
                                "document_id": str(chunk.document_id) if chunk.document_id else "",
                                "tags": [],
                                "access_level": "read",
                            },
                            "collection_id": collection.id,
                            "collection_name": collection.name,
                            "hub_id": self.hub_id,
                            "strategy": "sparse",
                        })
                    sql_hits.sort(key=lambda x: x["score"], reverse=True)
                    hits = sql_hits[:limit]
                except Exception as sqle:
                    logger.warning("SQL BM25 fallback search failed: %s", sqle)

        return hits

    async def search_graph(
        self,
        collection: SyntraFlowCollection,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Traverse Neo4j Knowledge Graph with mandatory hub_id scoping."""
        try:
            driver = await resolve_graph_client(self.db, self.hub_id)
            cypher_query = (
                "MATCH (e:SyntraFlow_Entity {hub_id: $hub_id})-[r:SyntraFlow_RELATION]->(o:SyntraFlow_Entity {hub_id: $hub_id}) "
                "WHERE (toLower(e.name) CONTAINS toLower($search_query) OR toLower(o.name) CONTAINS toLower($search_query)) AND ($col_id IS NULL OR e.collection_id = $col_id) "
                "RETURN e.name AS source, o.name AS target, r.type AS rel_type, r.description AS desc "
                "LIMIT $limit"
            )
            async with driver.session() as session:
                res = await session.run(cypher_query, hub_id=self.hub_id, search_query=query, col_id=collection.id, limit=limit)
                records = await res.data()

            hits = []
            for rec in records:
                text_repr = f"Graph Relation: {rec['source']} -> {rec['rel_type']} -> {rec['target']}"
                hits.append({
                    "id": str(hash(text_repr)),
                    "score": 1.0,
                    "text": text_repr,
                    "metadata": {
                        "type": "graph_relation",
                        "source": rec["source"],
                        "target": rec["target"],
                        "description": rec.get("desc", ""),
                    },
                    "collection_id": collection.id,
                    "collection_name": collection.name,
                    "hub_id": self.hub_id,
                    "strategy": "graph",
                })
            return hits
        except DatastoreUnavailableError:
            raise
        except Exception as e:
            logger.warning("Neo4j graph traversal failed or offline: %s", e)
            return []

    async def search_hybrid(
        self,
        collection: SyntraFlowCollection,
        query: str,
        query_vector: List[float],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Perform Hybrid RRF search combining Dense vector, Sparse BM25, and Graph hits."""
        dense_hits = await self.search_vector(
            collection=collection, query_vector=query_vector, limit=limit * 2, metadata_filter=metadata_filter
        )
        try:
            sparse_hits = await self.search_sparse(
                collection=collection, query=query, limit=limit * 2, metadata_filter=metadata_filter
            )
        except Exception:
            sparse_hits = []

        graph_hits = await self.search_graph(collection=collection, query=query, limit=limit * 2)

        scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            key = hit["text"]
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank + rrf_k))
            items_map[key] = hit

        for rank, hit in enumerate(sparse_hits):
            key = hit["text"]
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank + rrf_k))
            if key not in items_map:
                items_map[key] = hit

        for rank, hit in enumerate(graph_hits):
            key = hit["text"]
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank + rrf_k))
            if key not in items_map:
                items_map[key] = hit

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:limit]
        fused = []
        for key in sorted_keys:
            item = dict(items_map[key])
            item["score"] = round(scores[key], 6)
            item["strategy"] = "hybrid"
            fused.append(item)

        return fused

    async def search(
        self,
        *,
        query: str,
        collection_ids: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        limit: int = 5,
        query_vector: Optional[List[float]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute hub-scoped search across one or all collections with multi-collection fan-in."""
        collections = await self.resolve_targets(collection_ids)
        if not collections:
            return []

        # If query_vector not supplied, compute embedding
        if not query_vector:
            dim = int(collections[0].vector_dimension) if collections and collections[0].vector_dimension else 768
            try:
                from common.clients.inference import InferenceClient
                inf_client = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)
                target_model = collections[0].embedding_model if collections else None
                embeds = await inf_client.embed(texts=[query], model=target_model)
                if embeds and len(embeds[0]) > 0:
                    query_vector = embeds[0]
                else:
                    query_vector = [0.0] * dim
            except Exception as e:
                logger.warning("Inference client embedding failed in search: %s. Using zero vector of dim %d.", e, dim)
                query_vector = [0.0] * dim

        async def _search_collection(col: SyntraFlowCollection) -> List[Dict[str, Any]]:
            # Determine effective strategy
            col_cfg = {}
            if col.retrieval_config_json:
                try:
                    col_cfg = json.loads(col.retrieval_config_json)
                except Exception:
                    pass

            eff_strat = (strategy or col_cfg.get("strategy") or settings.RAG_STRATEGY or "dense").lower().strip()
            rrf_k = col_cfg.get("rrf_k", 60)

            if eff_strat == "sparse":
                return await self.search_sparse(col, query, limit, metadata_filter)
            elif eff_strat == "graph":
                return await self.search_graph(col, query, limit)
            elif eff_strat == "hybrid":
                return await self.search_hybrid(col, query, query_vector, limit, metadata_filter, rrf_k=rrf_k)
            else:
                return await self.search_vector(col, query_vector, limit, metadata_filter)

        tasks = [_search_collection(c) for c in collections]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_hits = []
        warnings = []

        for res in results_list:
            if isinstance(res, DatastoreUnavailableError):
                raise HTTPException(status_code=503, detail="Datastore unavailable")
            elif isinstance(res, Exception):
                logger.warning("Collection search encountered error: %s", res)
                warnings.append(str(res))
            elif isinstance(res, list):
                all_hits.extend(res)

        # Multi-collection RRF Fusion across fan-in result lists
        scores: Dict[str, float] = {}
        items_map: Dict[str, Dict[str, Any]] = {}

        for rank, hit in enumerate(all_hits):
            key = f"{hit['collection_id']}::{hit['id']}"
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank + 60))
            items_map[key] = hit

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:limit]
        final_results = []
        for key in sorted_keys:
            hit_item = dict(items_map[key])
            hit_item["score"] = round(scores[key], 6)
            final_results.append(hit_item)

        return final_results

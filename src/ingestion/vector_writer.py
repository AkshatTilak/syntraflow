"""Vector payload attachment, validation, and metadata filtering pipeline helper for SyntraFlow."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger("syntraflow.ingestion.vector_writer")

REQUIRED_PAYLOAD_KEYS = {"tenant_id", "document_id"}
DEFAULT_ACCESS_LEVEL = "public"


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and format point payload with standardized metadata fields.

    Args:
        payload: Input metadata dictionary.

    Returns:
        Cleaned payload dictionary containing required standard keys.
    """
    validated = dict(payload)
    if "tenant_id" not in validated or not validated["tenant_id"]:
        validated["tenant_id"] = "default"
    if "document_id" not in validated:
        validated["document_id"] = "unknown"
    if "tags" not in validated or not isinstance(validated["tags"], list):
        validated["tags"] = []
    if "access_level" not in validated:
        validated["access_level"] = DEFAULT_ACCESS_LEVEL
    if "created_at" not in validated:
        validated["created_at"] = datetime.utcnow().isoformat()

    return validated


def build_qdrant_filter(filters: Dict[str, Any]) -> Optional[qdrant_models.Filter]:
    """Build Qdrant Filter object from a dictionary of metadata conditions.

    Supports:
        - Exact match: key -> value
        - List match: key -> list of values (MatchAny)

    Args:
        filters: Key-value dictionary of query filter conditions.

    Returns:
        Qdrant Filter object or None if filters dictionary is empty.
    """
    if not filters:
        return None

    must_conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=value),
                )
            )
        else:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=str(value)),
                )
            )

    if not must_conditions:
        return None

    return qdrant_models.Filter(must=must_conditions)


class VectorWriter:
    """Helper for batch upserts with validated metadata payloads into Qdrant collections."""

    def __init__(self, vector_client: Any) -> None:
        """Initialize VectorWriter.

        Args:
            vector_client: VectorClient wrapper instance.
        """
        self.vector_client = vector_client

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Batch upsert points with validated payloads into specified Qdrant collection.

        Args:
            collection_name: Name of target collection.
            points: List of dicts with 'id', 'vector', and 'payload'.
            batch_size: Batch size limit.

        Returns:
            Number of points successfully upserted.
        """
        if not self.vector_client:
            logger.warning("VectorClient unavailable; skipping Qdrant upsert.")
            return 0

        qdrant = self.vector_client.get_client()
        qdrant_points = []

        for item in points:
            point_id = item["id"]
            vector = item["vector"]
            payload = validate_payload(item.get("payload", {}))

            qdrant_points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        upserted_count = 0
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i : i + batch_size]
            qdrant.upsert(
                collection_name=collection_name,
                points=batch,
            )
            upserted_count += len(batch)

        logger.info("Upserted %d points to Qdrant collection '%s'.", upserted_count, collection_name)
        return upserted_count

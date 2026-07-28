"""Dynamic Qdrant & SQL Collection Lifecycle Manager for SyntraFlow."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from qdrant_client.http import models as qdrant_models

from common.clients.qdrant import VectorClient
from projects.syntraflow.src.database.models import SyntraFlowCollection, build_physical_name

logger = logging.getLogger("syntraflow.collections.manager")


class CollectionManager:
    """Manages dynamic Qdrant vector collections and SQL catalog sync."""

    def __init__(self, db: Session, vector_client: Optional[VectorClient] = None) -> None:
        """Initialize CollectionManager.

        Args:
            db: SQLAlchemy Session instance.
            vector_client: VectorClient wrapper instance.
        """
        self.db = db
        self._vector_client = vector_client

    @property
    def vector_client(self) -> VectorClient:
        """Get or lazily initialize VectorClient."""
        if self._vector_client is None:
            try:
                self._vector_client = VectorClient()
            except Exception as e:
                logger.warning("VectorClient initialization failed in CollectionManager: %s", e)
        return self._vector_client

    def create_collection(
        self,
        name: str,
        hub_id: str,
        hub_slug: str = "default",
        embedding_model: str = "jina-clip-v2",
        vector_dimension: int = 1024,
        description: Optional[str] = None,
    ) -> SyntraFlowCollection:
        """Create a new dynamic vector collection in Qdrant and store metadata in SQL.

        Args:
            name: Name for the vector collection (unique within hub).
            hub_id: Parent hub UUID string.
            hub_slug: Parent hub slug string.
            embedding_model: Associated embedding model name.
            vector_dimension: Vector dimension size (e.g. 1024).
            description: Optional textual description.

        Returns:
            Created SyntraFlowCollection SQL model instance.
        """
        physical_name = build_physical_name(hub_slug, name)

        # Check for existing record in DB
        existing = self.db.query(SyntraFlowCollection).filter_by(hub_id=hub_id, name=name).first()
        if existing:
            raise ValueError(f"Collection with name '{name}' already exists in this hub.")

        # Initialize Qdrant collection if client is available
        if self.vector_client:
            try:
                qdrant = self.vector_client.get_client()
                qdrant.create_collection(
                    collection_name=physical_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=vector_dimension,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info("Qdrant collection '%s' created successfully.", physical_name)
            except Exception as e:
                logger.error("Failed to create Qdrant collection '%s': %s", physical_name, e)

        # Create SQL model record
        col_record = SyntraFlowCollection(
            id=str(uuid.uuid4()),
            hub_id=hub_id,
            name=name,
            physical_name=physical_name,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            description=description,
        )
        self.db.add(col_record)
        self.db.commit()
        self.db.refresh(col_record)
        return col_record

    def list_collections(self, hub_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List vector collections with SQL metadata and Qdrant stats.

        Args:
            hub_id: Optional hub filter.

        Returns:
            List of dictionaries containing collection metadata and counts.
        """
        query = self.db.query(SyntraFlowCollection)
        if hub_id:
            query = query.filter_by(hub_id=hub_id)
        records = query.all()

        results = []
        qdrant = self.vector_client.get_client() if self.vector_client else None

        for rec in records:
            points_count = 0
            status = "active"

            if qdrant:
                try:
                    info = qdrant.get_collection(rec.physical_name)
                    points_count = info.points_count or 0
                except Exception:
                    status = "unreachable"

            results.append({
                "id": str(rec.id),
                "hub_id": rec.hub_id,
                "name": rec.name,
                "physical_name": rec.physical_name,
                "embedding_model": rec.embedding_model,
                "vector_dimension": int(rec.vector_dimension),
                "description": rec.description,
                "points_count": points_count,
                "status": status,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
            })

        return results

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed collection info by ID or name.

        Args:
            collection_id: UUID string or collection name.

        Returns:
            Collection detail dict or None.
        """
        rec = None
        try:
            val_uuid = uuid.UUID(collection_id)
            rec = self.db.query(SyntraFlowCollection).filter_by(id=val_uuid).first()
        except ValueError:
            rec = self.db.query(SyntraFlowCollection).filter_by(name=collection_id).first()

        if not rec:
            return None

        points_count = 0
        vectors_count = 0
        status = "active"
        qdrant = self.vector_client.get_client() if self.vector_client else None

        if qdrant:
            try:
                info = qdrant.get_collection(rec.name)
                points_count = info.points_count or 0
                vectors_count = getattr(info, "indexed_vectors_count", points_count) or points_count
            except Exception:
                status = "unreachable"

        return {
            "id": str(rec.id),
            "name": rec.name,
            "tenant_id": rec.tenant_id,
            "embedding_model": rec.embedding_model,
            "vector_dimension": int(rec.vector_dimension),
            "description": rec.description,
            "points_count": points_count,
            "vectors_count": vectors_count,
            "status": status,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }

    def delete_collection(self, collection_id: str) -> bool:
        """Delete vector collection from Qdrant and remove SQL metadata record.

        Args:
            collection_id: UUID string or collection name.

        Returns:
            True if deleted, False if not found.
        """
        rec = None
        try:
            val_uuid = uuid.UUID(collection_id)
            rec = self.db.query(SyntraFlowCollection).filter_by(id=val_uuid).first()
        except ValueError:
            rec = self.db.query(SyntraFlowCollection).filter_by(name=collection_id).first()

        if not rec:
            return False

        col_name = rec.name

        # Delete from Qdrant
        if self.vector_client:
            try:
                qdrant = self.vector_client.get_client()
                qdrant.delete_collection(collection_name=col_name)
                logger.info("Qdrant collection '%s' deleted successfully.", col_name)
            except Exception as e:
                logger.error("Failed to delete Qdrant collection '%s': %s", col_name, e)

        # Delete from SQL
        self.db.delete(rec)
        self.db.commit()
        return True

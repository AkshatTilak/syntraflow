"""Dynamic Qdrant & SQL Collection Lifecycle Manager for Ingestion Hub (S6-04a)."""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client.http import models as qdrant_models
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from common.clients.qdrant import VectorClient
from common.config.settings import get_settings
from common.models.database import Hub
from projects.syntraflow.src.collections.schemas import CollectionRetrievalConfig
from projects.syntraflow.src.database.models import SyntraFlowCollection, SyntraFlowDocument, SyntraFlowChunk, build_physical_name

logger = logging.getLogger("syntraflow.collections.manager")

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]{0,62}$")
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


class CollectionProvisioningError(Exception):
    """Raised when physical vector collection provisioning fails."""
    pass


def physical_collection_name(hub_slug: str, name: str) -> str:
    """Deterministic global Qdrant physical collection name: '{hub_slug}__{collection_name}'."""
    slug_clean = _SLUG_RE.sub("_", hub_slug.lower())
    name_clean = _SLUG_RE.sub("_", name.lower())
    return f"{slug_clean}__{name_clean}"


def validate_collection_name(name: str) -> str:
    """Validate friendly collection name format."""
    if not name or not _NAME_RE.match(name) or "__" in name:
        raise ValueError(
            f"Invalid collection name '{name}'. Must be 1-63 alphanumeric/space/dash/underscore chars and cannot contain '__'."
        )
    return name.strip()


class CollectionManager:
    """Manages dynamic Qdrant vector collections and SQL catalog sync for an Ingestion Hub."""

    def __init__(self, db: AsyncSession, vector_client: Optional[VectorClient] = None) -> None:
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

    async def _resolve_hub(self, hub_id: str) -> Hub:
        """Resolve Ingestion Hub row from DB or raise 404 ValueError."""
        stmt = select(Hub).where(Hub.id == hub_id, Hub.is_archived.is_(False))
        res = await self.db.execute(stmt)
        hub = res.scalar_one_or_none()
        if not hub:
            raise ValueError(f"Ingestion Hub '{hub_id}' not found or archived.")
        if hub.hub_type != "ingestion":
            raise ValueError(f"Hub '{hub_id}' is of type '{hub.hub_type}', not 'ingestion'.")
        return hub

    async def create_collection(
        self,
        *,
        hub_id: str,
        name: str,
        embedding_model: str = "jina-clip-v2",
        vector_dimension: int = 1024,
        description: Optional[str] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
        datastore_binding_id: Optional[str] = None,
    ) -> SyntraFlowCollection:
        """Create a new dynamic vector collection in Qdrant and store metadata in SQL."""
        hub = await self._resolve_hub(hub_id)
        valid_name = validate_collection_name(name)

        # Check per-hub name uniqueness
        stmt_exist = select(SyntraFlowCollection).where(
            SyntraFlowCollection.hub_id == hub_id,
            SyntraFlowCollection.name == valid_name,
        )
        res_exist = await self.db.execute(stmt_exist)
        if res_exist.scalar_one_or_none():
            raise ValueError(f"Collection '{valid_name}' already exists in this hub.")

        physical_name = physical_collection_name(hub.slug, valid_name)

        # Process and validate retrieval config
        settings = get_settings()
        default_strategy = getattr(settings, "RAG_STRATEGY", "hybrid")
        raw_cfg = retrieval_config or {}
        if "strategy" not in raw_cfg:
            raw_cfg["strategy"] = default_strategy
        parsed_config = CollectionRetrievalConfig(**raw_cfg)

        # Provision physical Qdrant collection
        qdrant_created = False
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
                qdrant_created = True
                logger.info("Qdrant physical collection '%s' created.", physical_name)
            except Exception as e:
                logger.error("Failed to create Qdrant physical collection '%s': %s", physical_name, e)
                raise CollectionProvisioningError(f"Failed to provision vector collection: {e}")

        # Insert SQL metadata row with compensation handling
        try:
            col_record = SyntraFlowCollection(
                id=str(uuid.uuid4()),
                hub_id=hub_id,
                name=valid_name,
                physical_name=physical_name,
                embedding_model=embedding_model,
                vector_dimension=float(vector_dimension),
                description=description,
                retrieval_config_json=parsed_config.model_dump(),
                datastore_binding_id=datastore_binding_id,
            )
            self.db.add(col_record)
            await self.db.flush()
            await self.db.commit()
            await self.db.refresh(col_record)
            return col_record
        except Exception as sql_err:
            # Compensation: delete created physical collection
            if qdrant_created and self.vector_client:
                try:
                    self.vector_client.get_client().delete_collection(physical_name)
                except Exception as del_err:
                    logger.error("Compensation cleanup failed for '%s': %s", physical_name, del_err)
            await self.db.rollback()
            raise CollectionProvisioningError(f"Failed to persist collection metadata: {sql_err}")

    async def list_collections(self, *, hub_id: str) -> List[Dict[str, Any]]:
        """List all collections scoped to hub_id with SQL metadata and vector counts."""
        await self._resolve_hub(hub_id)

        stmt = (
            select(SyntraFlowCollection)
            .where(SyntraFlowCollection.hub_id == hub_id)
            .order_by(SyntraFlowCollection.created_at.desc())
        )
        res = await self.db.execute(stmt)
        records = res.scalars().all()

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
                "retrieval_config": rec.retrieval_config_json or {},
                "datastore_binding_id": rec.datastore_binding_id,
                "points_count": points_count,
                "status": status,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
            })

        return results

    async def get_collection(self, *, hub_id: str, collection_id: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed collection metadata filtered strictly by hub_id."""
        await self._resolve_hub(hub_id)

        stmt = select(SyntraFlowCollection).where(SyntraFlowCollection.hub_id == hub_id)
        try:
            val_uuid = uuid.UUID(collection_id)
            stmt = stmt.where(SyntraFlowCollection.id == str(val_uuid))
        except ValueError:
            stmt = stmt.where(SyntraFlowCollection.name == collection_id)

        res = await self.db.execute(stmt)
        rec = res.scalar_one_or_none()
        if not rec:
            return None

        points_count = 0
        vectors_count = 0
        status = "active"
        qdrant = self.vector_client.get_client() if self.vector_client else None

        if qdrant:
            try:
                info = qdrant.get_collection(rec.physical_name)
                points_count = info.points_count or 0
                vectors_count = getattr(info, "indexed_vectors_count", points_count) or points_count
            except Exception:
                status = "unreachable"

        return {
            "id": str(rec.id),
            "hub_id": rec.hub_id,
            "name": rec.name,
            "physical_name": rec.physical_name,
            "embedding_model": rec.embedding_model,
            "vector_dimension": int(rec.vector_dimension),
            "description": rec.description,
            "retrieval_config": rec.retrieval_config_json or {},
            "datastore_binding_id": rec.datastore_binding_id,
            "points_count": points_count,
            "vectors_count": vectors_count,
            "status": status,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }

    async def update_collection(
        self,
        *,
        hub_id: str,
        collection_id: str,
        description: Optional[str] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
        datastore_binding_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update collection metadata and retrieval strategy configuration."""
        await self._resolve_hub(hub_id)

        stmt = select(SyntraFlowCollection).where(SyntraFlowCollection.hub_id == hub_id)
        try:
            val_uuid = uuid.UUID(collection_id)
            stmt = stmt.where(SyntraFlowCollection.id == str(val_uuid))
        except ValueError:
            stmt = stmt.where(SyntraFlowCollection.name == collection_id)

        res = await self.db.execute(stmt)
        rec = res.scalar_one_or_none()
        if not rec:
            return None

        if description is not None:
            rec.description = description
        if datastore_binding_id is not None:
            rec.datastore_binding_id = datastore_binding_id
        if retrieval_config is not None:
            parsed = CollectionRetrievalConfig(**retrieval_config)
            rec.retrieval_config_json = parsed.model_dump()

        await self.db.commit()
        await self.db.refresh(rec)
        return await self.get_collection(hub_id=hub_id, collection_id=str(rec.id))

    async def delete_collection(
        self,
        *,
        hub_id: str,
        collection_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Delete collection from vector store and SQL database with delete safety checks."""
        await self._resolve_hub(hub_id)

        stmt = select(SyntraFlowCollection).where(SyntraFlowCollection.hub_id == hub_id)
        try:
            val_uuid = uuid.UUID(collection_id)
            stmt = stmt.where(SyntraFlowCollection.id == str(val_uuid))
        except ValueError:
            stmt = stmt.where(SyntraFlowCollection.name == collection_id)

        res = await self.db.execute(stmt)
        rec = res.scalar_one_or_none()
        if not rec:
            raise ValueError(f"Collection '{collection_id}' not found in this hub.")

        # Check associated documents count for delete safety
        doc_stmt = select(func.count(SyntraFlowDocument.id)).where(SyntraFlowDocument.hub_id == hub_id)
        doc_res = await self.db.execute(doc_stmt)
        doc_count = doc_res.scalar() or 0

        if doc_count > 0 and not force:
            raise ValueError(f"Collection '{rec.name}' is not empty ({doc_count} documents). Pass force=True to delete.")

        # Count chunks
        chunk_stmt = select(func.count(SyntraFlowChunk.id)).where(SyntraFlowChunk.hub_id == hub_id)
        chunk_count = (await self.db.execute(chunk_stmt)).scalar() or 0

        # Delete physical vector collection
        if self.vector_client:
            try:
                qdrant = self.vector_client.get_client()
                qdrant.delete_collection(collection_name=rec.physical_name)
                logger.info("Deleted Qdrant physical collection '%s'.", rec.physical_name)
            except Exception as e:
                logger.error("Failed to delete Qdrant physical collection '%s': %s", rec.physical_name, e)

        # Delete SQL metadata
        await self.db.delete(rec)
        await self.db.commit()

        return {
            "deleted": {
                "collection_id": str(rec.id),
                "name": rec.name,
                "physical_name": rec.physical_name,
                "documents": doc_count,
                "chunks": chunk_count,
            }
        }

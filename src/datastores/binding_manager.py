"""Datastore Binding Manager with CRUD, synthetic default surfacing, dependency checks, and audit logging (S6-04b)."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from common.config.settings import get_settings
from common.models.database import AuditLog, DatastoreBinding, Hub
from common.models.hub_enums import STORE_TYPES
from projects.syntraflow.src.database.models import SyntraFlowCollection
from projects.syntraflow.src.datastores.crypto import (
    decrypt_credentials,
    encrypt_credentials,
    mask_uri,
)
from projects.syntraflow.src.datastores.health import test_store_connection
from projects.syntraflow.src.datastores.resolver import invalidate_hub_clients
from projects.syntraflow.src.datastores.schemas import (
    ConnectionTestResult,
    DatastoreBindingResponse,
)

logger = logging.getLogger("syntraflow.datastores.binding_manager")


def get_platform_default_uri(store_type: str) -> str:
    """Get standard platform default URI for a store type."""
    settings = get_settings()
    if store_type == "qdrant":
        return settings.QDRANT_URL
    elif store_type == "neo4j":
        return settings.NEO4J_URL
    elif store_type == "postgres":
        return settings.DATABASE_URL
    elif store_type == "opensearch":
        return getattr(settings, "OPENSEARCH_URL", "http://localhost:9200")
    return ""


class DatastoreBindingManager:
    """Manages physical datastore bindings owned by ingestion hubs."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _verify_ingestion_hub(self, hub_id: str) -> Hub:
        """Verify hub exists and is an ingestion hub."""
        hub = await self.db.get(Hub, hub_id)
        if not hub or hub.hub_type != "ingestion":
            raise ValueError(f"Hub '{hub_id}' not found or is not an ingestion hub")
        return hub

    async def create_binding(
        self,
        *,
        hub_id: str,
        name: str,
        store_type: str,
        connection_uri: str,
        credentials: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
        config: Optional[Dict[str, Any]] = None,
        actor_user_id: Optional[str] = None,
    ) -> DatastoreBinding:
        """Create a new datastore binding for an ingestion hub."""
        await self._verify_ingestion_hub(hub_id)

        if store_type not in STORE_TYPES:
            raise ValueError(f"Invalid store_type '{store_type}'. Must be one of {STORE_TYPES}")

        # Check unique constraint (hub_id, name)
        stmt = select(DatastoreBinding).where(
            DatastoreBinding.hub_id == hub_id,
            DatastoreBinding.name == name,
        )
        res = await self.db.execute(stmt)
        if res.scalar_one_or_none():
            raise ValueError(f"Datastore binding '{name}' already exists in this hub")

        # If setting as default, clear default flag on existing siblings of same store_type
        if is_default:
            clear_stmt = (
                update(DatastoreBinding)
                .where(
                    DatastoreBinding.hub_id == hub_id,
                    DatastoreBinding.store_type == store_type,
                )
                .values(is_default=False)
            )
            await self.db.execute(clear_stmt)

        enc_creds = encrypt_credentials(credentials) if credentials else None

        binding = DatastoreBinding(
            hub_id=hub_id,
            name=name,
            store_type=store_type,
            connection_uri=connection_uri,
            credentials_encrypted=enc_creds,
            is_default=is_default,
            health_status="unknown",
            config_json=config or {},
        )
        self.db.add(binding)
        await self.db.flush()

        # Audit log entry
        after_json = {
            "id": binding.id,
            "hub_id": hub_id,
            "name": name,
            "store_type": store_type,
            "connection_uri": mask_uri(connection_uri),
            "is_default": is_default,
        }
        audit = AuditLog(
            hub_id=hub_id,
            actor_user_id=actor_user_id,
            action="create",
            resource_type="datastore_binding",
            resource_id=binding.id,
            summary=f"Created {store_type} datastore binding '{name}'",
            after_json=after_json,
        )
        self.db.add(audit)
        await self.db.commit()

        invalidate_hub_clients(hub_id, store_type)
        return binding

    async def list_bindings(
        self, *, hub_id: str, store_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List datastore bindings for a hub, including synthetic platform defaults."""
        await self._verify_ingestion_hub(hub_id)

        stmt = select(DatastoreBinding).where(DatastoreBinding.hub_id == hub_id)
        if store_type:
            if store_type not in STORE_TYPES:
                raise ValueError(f"Invalid store_type '{store_type}'")
            stmt = stmt.where(DatastoreBinding.store_type == store_type)

        res = await self.db.execute(stmt)
        bindings = list(res.scalars().all())

        results: List[Dict[str, Any]] = []
        found_types = set()

        for b in bindings:
            found_types.add(b.store_type)
            resp = DatastoreBindingResponse(
                id=b.id,
                hub_id=b.hub_id,
                name=b.name,
                store_type=b.store_type,
                connection_uri=b.connection_uri,
                is_default=b.is_default,
                health_status=b.health_status,
                last_health_check=b.last_health_check,
                is_synthetic=False,
                config_json=b.config_json or {},
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            results.append(resp.model_dump())

        # Check for synthetic platform defaults
        target_types = [store_type] if store_type else list(STORE_TYPES)
        for st in target_types:
            if st not in found_types:
                def_uri = get_platform_default_uri(st)
                synth = DatastoreBindingResponse(
                    id=f"platform-default:{st}",
                    hub_id=hub_id,
                    name="Platform Default",
                    store_type=st,
                    connection_uri=def_uri,
                    is_default=True,
                    health_status="healthy",
                    last_health_check=None,
                    is_synthetic=True,
                    config_json={},
                    created_at=None,
                    updated_at=None,
                )
                results.append(synth.model_dump())

        return results

    async def get_binding(self, *, hub_id: str, binding_id: str) -> Optional[Dict[str, Any]]:
        """Get binding by hub_id and binding_id."""
        await self._verify_ingestion_hub(hub_id)

        if binding_id.startswith("platform-default:"):
            st = binding_id.split(":", 1)[1]
            if st in STORE_TYPES:
                def_uri = get_platform_default_uri(st)
                synth = DatastoreBindingResponse(
                    id=binding_id,
                    hub_id=hub_id,
                    name="Platform Default",
                    store_type=st,
                    connection_uri=def_uri,
                    is_default=True,
                    health_status="healthy",
                    last_health_check=None,
                    is_synthetic=True,
                    config_json={},
                )
                return synth.model_dump()
            return None

        stmt = select(DatastoreBinding).where(
            DatastoreBinding.id == binding_id,
            DatastoreBinding.hub_id == hub_id,
        )
        res = await self.db.execute(stmt)
        b = res.scalar_one_or_none()
        if not b:
            return None

        resp = DatastoreBindingResponse(
            id=b.id,
            hub_id=b.hub_id,
            name=b.name,
            store_type=b.store_type,
            connection_uri=b.connection_uri,
            is_default=b.is_default,
            health_status=b.health_status,
            last_health_check=b.last_health_check,
            is_synthetic=False,
            config_json=b.config_json or {},
            created_at=b.created_at,
            updated_at=b.updated_at,
        )
        return resp.model_dump()

    async def update_binding(
        self,
        *,
        hub_id: str,
        binding_id: str,
        actor_user_id: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Update an existing datastore binding."""
        await self._verify_ingestion_hub(hub_id)

        if binding_id.startswith("platform-default:"):
            raise ValueError("Platform default binding is read-only")

        stmt = select(DatastoreBinding).where(
            DatastoreBinding.id == binding_id,
            DatastoreBinding.hub_id == hub_id,
        )
        res = await self.db.execute(stmt)
        b = res.scalar_one_or_none()
        if not b:
            raise ValueError("Datastore binding not found")

        before_json = {
            "id": b.id,
            "hub_id": b.hub_id,
            "name": b.name,
            "store_type": b.store_type,
            "connection_uri": mask_uri(b.connection_uri),
            "is_default": b.is_default,
        }

        if "name" in fields and fields["name"] and fields["name"] != b.name:
            dup_stmt = select(DatastoreBinding).where(
                DatastoreBinding.hub_id == hub_id,
                DatastoreBinding.name == fields["name"],
                DatastoreBinding.id != binding_id,
            )
            dup_res = await self.db.execute(dup_stmt)
            if dup_res.scalar_one_or_none():
                raise ValueError(f"Datastore binding '{fields['name']}' already exists in this hub")
            b.name = fields["name"]

        if "connection_uri" in fields and fields["connection_uri"]:
            b.connection_uri = fields["connection_uri"]

        if "credentials" in fields:
            creds = fields["credentials"]
            b.credentials_encrypted = encrypt_credentials(creds) if creds else None

        if "is_default" in fields and fields["is_default"] is True:
            clear_stmt = (
                update(DatastoreBinding)
                .where(
                    DatastoreBinding.hub_id == hub_id,
                    DatastoreBinding.store_type == b.store_type,
                    DatastoreBinding.id != binding_id,
                )
                .values(is_default=False)
            )
            await self.db.execute(clear_stmt)
            b.is_default = True

        if "config" in fields and fields["config"] is not None:
            b.config_json = fields["config"]

        await self.db.flush()

        after_json = {
            "id": b.id,
            "hub_id": b.hub_id,
            "name": b.name,
            "store_type": b.store_type,
            "connection_uri": mask_uri(b.connection_uri),
            "is_default": b.is_default,
        }

        audit = AuditLog(
            hub_id=hub_id,
            actor_user_id=actor_user_id,
            action="update",
            resource_type="datastore_binding",
            resource_id=b.id,
            summary=f"Updated datastore binding '{b.name}'",
            before_json=before_json,
            after_json=after_json,
        )
        self.db.add(audit)
        await self.db.commit()

        invalidate_hub_clients(hub_id, b.store_type)

        resp = DatastoreBindingResponse(
            id=b.id,
            hub_id=b.hub_id,
            name=b.name,
            store_type=b.store_type,
            connection_uri=b.connection_uri,
            is_default=b.is_default,
            health_status=b.health_status,
            last_health_check=b.last_health_check,
            is_synthetic=False,
            config_json=b.config_json or {},
            created_at=b.created_at,
            updated_at=b.updated_at,
        )
        return resp.model_dump()

    async def delete_binding(
        self, *, hub_id: str, binding_id: str, actor_user_id: Optional[str] = None
    ) -> None:
        """Delete a datastore binding if not referenced by active collections."""
        await self._verify_ingestion_hub(hub_id)

        if binding_id.startswith("platform-default:"):
            raise ValueError("Platform default binding is read-only")

        stmt = select(DatastoreBinding).where(
            DatastoreBinding.id == binding_id,
            DatastoreBinding.hub_id == hub_id,
        )
        res = await self.db.execute(stmt)
        b = res.scalar_one_or_none()
        if not b:
            raise ValueError("Datastore binding not found")

        # Dependency check: check syntraflow_collections
        col_stmt = select(SyntraFlowCollection.name).where(
            SyntraFlowCollection.datastore_binding_id == binding_id
        )
        col_res = await self.db.execute(col_stmt)
        col_names = list(col_res.scalars().all())
        if col_names:
            raise ValueError(f"Binding is in use by collections: {col_names}")

        before_json = {
            "id": b.id,
            "hub_id": b.hub_id,
            "name": b.name,
            "store_type": b.store_type,
            "connection_uri": mask_uri(b.connection_uri),
        }

        store_type = b.store_type
        binding_name = b.name

        await self.db.delete(b)

        audit = AuditLog(
            hub_id=hub_id,
            actor_user_id=actor_user_id,
            action="delete",
            resource_type="datastore_binding",
            resource_id=binding_id,
            summary=f"Deleted datastore binding '{binding_name}'",
            before_json=before_json,
        )
        self.db.add(audit)
        await self.db.commit()

        invalidate_hub_clients(hub_id, store_type)

    async def test_connection(
        self,
        *,
        hub_id: str,
        binding_id: Optional[str] = None,
        draft: Optional[Dict[str, Any]] = None,
    ) -> ConnectionTestResult:
        """Test datastore connection for an existing binding or draft payload."""
        await self._verify_ingestion_hub(hub_id)

        if draft:
            st = draft.get("store_type")
            uri = draft.get("connection_uri")
            creds = draft.get("credentials")
            cfg = draft.get("config")
            if not st or not uri:
                raise ValueError("Draft payload requires store_type and connection_uri")
            return await test_store_connection(
                store_type=st, connection_uri=uri, credentials=creds, config=cfg
            )

        if not binding_id:
            raise ValueError("Must provide either binding_id or draft payload")

        if binding_id.startswith("platform-default:"):
            st = binding_id.split(":", 1)[1]
            def_uri = get_platform_default_uri(st)
            return await test_store_connection(store_type=st, connection_uri=def_uri)

        b_dict = await self.get_binding(hub_id=hub_id, binding_id=binding_id)
        if not b_dict:
            raise ValueError("Datastore binding not found")

        stmt = select(DatastoreBinding).where(
            DatastoreBinding.id == binding_id, DatastoreBinding.hub_id == hub_id
        )
        res = await self.db.execute(stmt)
        b = res.scalar_one()
        creds = decrypt_credentials(b.credentials_encrypted)

        return await test_store_connection(
            store_type=b.store_type,
            connection_uri=b.connection_uri,
            credentials=creds,
            config=b.config_json,
        )

"""Centralized client resolution & caching for Hub datastores (S6-04b)."""

import logging
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from common.clients.qdrant import VectorClient
from common.config.settings import get_settings
from common.models.database import DatastoreBinding
from projects.syntraflow.src.datastores.crypto import decrypt_credentials

logger = logging.getLogger("syntraflow.datastores.resolver")


class DatastoreUnavailableError(Exception):
    """Raised when a requested hub datastore binding is unreachable."""

    def __init__(self, hub_id: str, store_type: str, binding_name: str) -> None:
        self.hub_id = hub_id
        self.store_type = store_type
        self.binding_name = binding_name
        super().__init__(
            f"Datastore binding '{binding_name}' ({store_type}) for hub '{hub_id}' is unreachable"
        )


# Global module cache for constructed clients: key -> (hub_id, store_type)
_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}


def invalidate_hub_clients(hub_id: str, store_type: Optional[str] = None) -> None:
    """Invalidate cached client instances for a hub."""
    global _CLIENT_CACHE
    if store_type:
        _CLIENT_CACHE.pop((hub_id, store_type), None)
    else:
        keys_to_del = [k for k in _CLIENT_CACHE if k[0] == hub_id]
        for k in keys_to_del:
            _CLIENT_CACHE.pop(k, None)


async def resolve_binding_for_store(
    session: AsyncSession, hub_id: str, store_type: str
) -> Optional[DatastoreBinding]:
    """Select the active DatastoreBinding for a hub and store type.

    Selection order:
    1. Hub's binding with `is_default == True` for `store_type`.
    2. Single binding for that `store_type` if only 1 exists.
    3. None (indicating fallback to platform default).
    """
    stmt = select(DatastoreBinding).where(
        DatastoreBinding.hub_id == hub_id,
        DatastoreBinding.store_type == store_type,
    )
    res = await session.execute(stmt)
    bindings = list(res.scalars().all())

    if not bindings:
        return None

    # Priority 1: explicitly marked as default
    for b in bindings:
        if b.is_default:
            return b

    # Priority 2: if only 1 binding exists
    if len(bindings) == 1:
        return bindings[0]

    # Default to first binding if multiple exist without explicit default flag
    return bindings[0]


async def resolve_vector_client(session: AsyncSession, hub_id: str) -> VectorClient:
    """Resolve vector client (Qdrant) for an ingestion hub."""
    cache_key = (hub_id, "qdrant")
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    binding = await resolve_binding_for_store(session, hub_id, "qdrant")

    if binding is not None:
        if binding.health_status == "unreachable":
            raise DatastoreUnavailableError(hub_id, "qdrant", binding.name)

        creds = decrypt_credentials(binding.credentials_encrypted)
        api_key = creds.get("api_key")
        client = VectorClient(url=binding.connection_uri, api_key=api_key)
    else:
        # Synthetic platform default fallback
        settings = get_settings()
        client = VectorClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    _CLIENT_CACHE[cache_key] = client
    return client


async def resolve_graph_client(session: AsyncSession, hub_id: str) -> Any:
    """Resolve Neo4j graph driver for an ingestion hub."""
    cache_key = (hub_id, "neo4j")
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    from neo4j import AsyncGraphDatabase

    binding = await resolve_binding_for_store(session, hub_id, "neo4j")

    if binding is not None:
        if binding.health_status == "unreachable":
            raise DatastoreUnavailableError(hub_id, "neo4j", binding.name)

        creds = decrypt_credentials(binding.credentials_encrypted)
        auth = None
        if creds.get("username") and creds.get("password"):
            auth = (creds["username"], creds["password"])

        driver = AsyncGraphDatabase.driver(binding.connection_uri, auth=auth)
    else:
        # Synthetic platform default fallback
        settings = get_settings()
        auth = None
        if settings.NEO4J_USER and settings.NEO4J_PASSWORD:
            auth = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

        driver = AsyncGraphDatabase.driver(settings.NEO4J_URL, auth=auth)

    _CLIENT_CACHE[cache_key] = driver
    return driver


async def resolve_relational_engine(session: AsyncSession, hub_id: str) -> AsyncEngine:
    """Resolve relational SQL engine (Postgres) for an ingestion hub."""
    cache_key = (hub_id, "postgres")
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    binding = await resolve_binding_for_store(session, hub_id, "postgres")

    if binding is not None:
        if binding.health_status == "unreachable":
            raise DatastoreUnavailableError(hub_id, "postgres", binding.name)

        pg_uri = binding.connection_uri
        if pg_uri.startswith("postgresql://"):
            pg_uri = pg_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(pg_uri)
    else:
        # Synthetic platform default fallback
        settings = get_settings()
        pg_uri = settings.DATABASE_URL
        if pg_uri.startswith("postgresql://"):
            pg_uri = pg_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

        engine = create_async_engine(pg_uri)

    _CLIENT_CACHE[cache_key] = engine
    return engine

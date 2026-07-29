"""Connection testing and background health check runner for datastore bindings (S6-04b)."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from common.models.database import DatastoreBinding
from common.models.hub_enums import STORE_TYPES
from projects.syntraflow.src.datastores.crypto import decrypt_credentials, mask_uri
from projects.syntraflow.src.datastores.schemas import ConnectionTestResult

logger = logging.getLogger("syntraflow.datastores.health")


async def test_store_connection(
    store_type: str,
    connection_uri: str,
    credentials: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 5.0,
) -> ConnectionTestResult:
    """Test physical datastore connection with a 5-second timeout."""
    if store_type not in STORE_TYPES:
        return ConnectionTestResult(
            ok=False,
            latency_ms=0.0,
            detail=f"Unsupported store_type '{store_type}'",
        )

    creds = credentials or {}
    start_time = time.perf_counter()

    try:
        if store_type == "qdrant":
            from qdrant_client import QdrantClient

            api_key = creds.get("api_key")
            # Wrap synchronous qdrant call in executor with timeout
            def _check():
                client = QdrantClient(url=connection_uri, api_key=api_key, timeout=timeout_seconds)
                res = client.get_collections()
                return len(res.collections)

            count = await asyncio.wait_for(asyncio.to_thread(_check), timeout=timeout_seconds)
            latency = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                ok=True,
                latency_ms=round(latency, 2),
                detail=f"Qdrant reachable ({count} collections)",
            )

        elif store_type == "neo4j":
            from neo4j import AsyncGraphDatabase

            auth = None
            if creds.get("username") and creds.get("password"):
                auth = (creds["username"], creds["password"])

            async with AsyncGraphDatabase.driver(connection_uri, auth=auth) as driver:
                await asyncio.wait_for(driver.verify_connectivity(), timeout=timeout_seconds)

            latency = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                ok=True,
                latency_ms=round(latency, 2),
                detail="Neo4j connectivity verified",
            )

        elif store_type == "postgres":
            # Strip postgresql:// to postgresql+asyncpg:// if needed
            pg_uri = connection_uri
            if pg_uri.startswith("postgresql://"):
                pg_uri = pg_uri.replace("postgresql://", "postgresql+asyncpg://", 1)

            temp_engine = create_async_engine(pg_uri, connect_args={"timeout": timeout_seconds})
            try:
                async with temp_engine.connect() as conn:
                    result = await asyncio.wait_for(conn.execute(text("SELECT version();")), timeout=timeout_seconds)
                    ver = result.scalar()
                latency = (time.perf_counter() - start_time) * 1000.0
                return ConnectionTestResult(
                    ok=True,
                    latency_ms=round(latency, 2),
                    detail="Postgres query successful",
                    version=str(ver)[:50] if ver else None,
                )
            finally:
                await temp_engine.dispose()

        elif store_type == "opensearch":
            auth = None
            if creds.get("username") and creds.get("password"):
                auth = (creds["username"], creds["password"])

            async with httpx.AsyncClient(timeout=timeout_seconds, verify=False) as client:
                res = await client.get(connection_uri, auth=auth)
                res.raise_for_status()
                data = res.json()
                ver = data.get("version", {}).get("number")

            latency = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                ok=True,
                latency_ms=round(latency, 2),
                detail="OpenSearch cluster reachable",
                version=str(ver) if ver else None,
            )

        else:
            return ConnectionTestResult(
                ok=False,
                latency_ms=0.0,
                detail=f"Unknown store type '{store_type}'",
            )

    except asyncio.TimeoutError:
        latency = (time.perf_counter() - start_time) * 1000.0
        return ConnectionTestResult(
            ok=False,
            latency_ms=round(latency, 2),
            detail="Connection timed out after 5.0 seconds",
        )
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"{type(e).__name__}: {str(e)[:100]}"
        # Ensure credentials/passwords in error string are masked
        err_msg = mask_uri(err_msg)
        return ConnectionTestResult(
            ok=False,
            latency_ms=round(latency, 2),
            detail=err_msg,
        )


async def run_health_checks(session: AsyncSession) -> None:
    """Background task to run health checks for all datastore bindings."""
    try:
        stmt = select(DatastoreBinding)
        res = await session.execute(stmt)
        bindings = list(res.scalars().all())

        for binding in bindings:
            creds = decrypt_credentials(binding.credentials_encrypted)
            result = await test_store_connection(
                store_type=binding.store_type,
                connection_uri=binding.connection_uri,
                credentials=creds,
                config=binding.config_json,
            )

            binding.health_status = "healthy" if result.ok else "unreachable"
            binding.last_health_check = datetime.utcnow()

        await session.commit()
    except Exception as e:
        logger.warning("Error running background health checks: %s", e)
        await session.rollback()

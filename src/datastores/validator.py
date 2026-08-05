"""Datastore Binding Validator for Ingestion Hub Collection Creation (sub_07_01)."""

import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from common.models.database import DatastoreBinding

logger = logging.getLogger("syntraflow.datastores.validator")


class DatastoreValidationError(Exception):
    """Raised when target datastore binding is invalid, unreachable, or unhealthy."""
    pass


async def validate_datastore_binding(
    db: AsyncSession,
    hub_id: str,
    datastore_binding_id: Optional[str] = None,
    store_type: str = "qdrant",
) -> Optional[DatastoreBinding]:
    """Validate datastore binding connectivity for a hub before creating collections.
    
    If datastore_binding_id is provided, verify it belongs to hub_id and store_type.
    If not provided, check if hub has a default binding or platform fallback.
    Returns the valid DatastoreBinding or None if using platform default.
    """
    if datastore_binding_id and not datastore_binding_id.startswith("platform-default") and datastore_binding_id.lower() not in ("default", "none"):
        stmt = select(DatastoreBinding).where(
            DatastoreBinding.id == datastore_binding_id,
            DatastoreBinding.hub_id == hub_id,
        )
        res = await db.execute(stmt)
        binding = res.scalar_one_or_none()
        if not binding:
            raise DatastoreValidationError(
                f"Datastore binding '{datastore_binding_id}' not found for hub '{hub_id}'."
            )
        if binding.store_type != store_type:
            raise DatastoreValidationError(
                f"Datastore binding '{datastore_binding_id}' has store_type '{binding.store_type}', expected '{store_type}'."
            )
        if binding.health_status in ("unhealthy", "unreachable"):
            raise DatastoreValidationError(
                f"Datastore binding '{binding.name}' is in '{binding.health_status}' state."
            )
        return binding

    # Check for hub default binding for store_type
    stmt = select(DatastoreBinding).where(
        DatastoreBinding.hub_id == hub_id,
        DatastoreBinding.store_type == store_type,
        DatastoreBinding.is_default.is_(True),
    )
    res = await db.execute(stmt)
    binding = res.scalar_one_or_none()
    if binding and binding.health_status in ("unhealthy", "unreachable"):
        raise DatastoreValidationError(
            f"Default datastore binding '{binding.name}' for hub '{hub_id}' is in '{binding.health_status}' state."
        )
    return binding

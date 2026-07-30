"""Pydantic schemas for Datastore Binding API operations (S6-04b)."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from common.models.hub_enums import STORE_TYPES
from projects.syntraflow.src.datastores.crypto import mask_uri


class DatastoreBindingCreate(BaseModel):
    """Payload for creating a hub datastore binding."""
    name: str = Field(..., min_length=1, max_length=120)
    store_type: str = Field(...)
    connection_uri: str = Field(..., min_length=1, max_length=500)
    credentials: Optional[Dict[str, Any]] = None
    is_default: bool = False
    config: Optional[Dict[str, Any]] = None

    @field_validator("store_type")
    @classmethod
    def validate_store_type(cls, v: str) -> str:
        if v not in STORE_TYPES:
            raise ValueError(f"Invalid store_type '{v}'. Must be one of {STORE_TYPES}")
        return v


class DatastoreBindingUpdate(BaseModel):
    """Payload for updating an existing datastore binding."""
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    connection_uri: Optional[str] = Field(None, min_length=1, max_length=500)
    credentials: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None


class DatastoreBindingResponse(BaseModel):
    """Public read model for a datastore binding. Explicitly excludes credential fields."""
    id: str
    hub_id: str
    name: str
    store_type: str
    connection_uri: str
    is_default: bool
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None
    is_synthetic: bool = False
    config_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("connection_uri", mode="before")
    @classmethod
    def mask_connection_uri(cls, v: str) -> str:
        return mask_uri(v)


class ConnectionTestResult(BaseModel):
    """Result summary of a datastore connectivity test."""
    ok: bool
    latency_ms: float
    detail: str
    version: Optional[str] = None

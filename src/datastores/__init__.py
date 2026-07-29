"""Ingestion Hub Datastore Bindings & Client Resolver Package (S6-04b)."""

from projects.syntraflow.src.datastores.binding_manager import DatastoreBindingManager
from projects.syntraflow.src.datastores.crypto import (
    decrypt_credentials,
    encrypt_credentials,
    mask_uri,
    verify_encryption_key_configured,
)
from projects.syntraflow.src.datastores.health import (
    run_health_checks,
    test_store_connection,
)
from projects.syntraflow.src.datastores.resolver import (
    DatastoreUnavailableError,
    invalidate_hub_clients,
    resolve_graph_client,
    resolve_relational_engine,
    resolve_vector_client,
)

__all__ = [
    "DatastoreBindingManager",
    "encrypt_credentials",
    "decrypt_credentials",
    "mask_uri",
    "verify_encryption_key_configured",
    "test_store_connection",
    "run_health_checks",
    "resolve_vector_client",
    "resolve_graph_client",
    "resolve_relational_engine",
    "invalidate_hub_clients",
    "DatastoreUnavailableError",
]

"""Unit tests for metadata payload validation and Qdrant filter building."""

import pytest
from qdrant_client.http import models as qdrant_models
from projects.syntraflow.src.ingestion.vector_writer import (
    validate_payload,
    build_qdrant_filter,
)


def test_validate_payload_defaults():
    """Verify validate_payload injects standard defaults when missing."""
    raw = {"document_id": "doc_123"}
    res = validate_payload(raw)

    assert res["hub_id"] == "default"
    assert res["document_id"] == "doc_123"
    assert res["access_level"] == "public"
    assert res["tags"] == []
    assert "created_at" in res


def test_validate_payload_custom_values():
    """Verify custom metadata fields are preserved."""
    raw = {
        "hub_id": "org_acme",
        "document_id": "doc_456",
        "access_level": "restricted",
        "tags": ["finance", "report"],
        "custom_key": "custom_val",
    }
    res = validate_payload(raw)

    assert res["hub_id"] == "org_acme"
    assert res["document_id"] == "doc_456"
    assert res["access_level"] == "restricted"
    assert res["tags"] == ["finance", "report"]
    assert res["custom_key"] == "custom_val"


def test_build_qdrant_filter_empty():
    """Verify build_qdrant_filter returns None for empty input."""
    assert build_qdrant_filter({}) is None
    assert build_qdrant_filter({"key": None}) is None


def test_build_qdrant_filter_single_and_list():
    """Verify filter construction for exact string and match any list."""
    filters = {
        "hub_id": "org_acme",
        "tags": ["finance", "tax"],
    }
    q_filter = build_qdrant_filter(filters)

    assert q_filter is not None
    assert len(q_filter.must) == 2

    # Check hub_id condition
    c1 = q_filter.must[0]
    assert c1.key == "hub_id"
    assert c1.match.value == "org_acme"

    # Check tags condition
    c2 = q_filter.must[1]
    assert c2.key == "tags"
    assert c2.match.any == ["finance", "tax"]

"""Unit tests for pluggable RetrievalEngine and strategies."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from projects.syntraflow.src.retrieval.engine import (
    RetrievalEngine,
    DenseRetrievalStrategy,
    SparseRetrievalStrategy,
    HybridRRFStrategy,
    GraphRetrievalStrategy,
)


@pytest.mark.asyncio
async def test_dense_retrieval_strategy_mocked():
    """Verify DenseRetrievalStrategy formats hits correctly from mocked Qdrant response."""
    mock_vector_client = MagicMock()
    mock_qdrant = MagicMock()
    mock_vector_client.get_client.return_value = mock_qdrant

    mock_hit = MagicMock()
    mock_hit.id = "hit_1"
    mock_hit.score = 0.95
    mock_hit.payload = {"text": "Dense retrieval test text", "tenant_id": "test_tenant"}
    mock_qdrant.search.return_value = [mock_hit]

    strategy = DenseRetrievalStrategy(vector_client=mock_vector_client)
    hits = await strategy.execute(
        query="test query",
        collection_name="test_collection",
        limit=5,
        query_vector=[0.1] * 1024,
    )

    assert len(hits) == 1
    assert hits[0]["id"] == "hit_1"
    assert hits[0]["score"] == 0.95
    assert hits[0]["text"] == "Dense retrieval test text"
    assert hits[0]["strategy"] == "dense"


@pytest.mark.asyncio
async def test_sparse_retrieval_token_match():
    """Verify SparseRetrievalStrategy scores text token matches."""
    mock_dense = AsyncMock()
    mock_dense.execute.return_value = [
        {"id": "1", "score": 0.8, "text": "Apple fruit pie recipe", "strategy": "dense"},
        {"id": "2", "score": 0.7, "text": "Car auto engine repair", "strategy": "dense"},
    ]

    strategy = SparseRetrievalStrategy(dense_fallback=mock_dense)
    hits = await strategy.execute(
        query="apple pie",
        collection_name="test_collection",
        limit=5,
    )

    assert len(hits) >= 1
    assert hits[0]["id"] == "1"
    assert hits[0]["strategy"] == "sparse"


@pytest.mark.asyncio
async def test_retrieval_engine_query_dispatch():
    """Verify RetrievalEngine dispatches query to designated strategy."""
    mock_vector_client = MagicMock()
    engine = RetrievalEngine(vector_client=mock_vector_client)

    # Mock dense strategy inside engine with AsyncMock
    engine.dense_strategy.execute = AsyncMock()
    engine.dense_strategy.execute.return_value = [
        {"id": "1", "score": 0.9, "text": "Sample text", "strategy": "dense"}
    ]

    hits = await engine.query(
        query="sample query",
        collection_name="test_col",
        strategy="dense",
        limit=3,
    )

    assert len(hits) == 1
    assert hits[0]["strategy"] == "dense"


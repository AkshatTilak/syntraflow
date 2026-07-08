"""Unit tests for the SyntraFlow Ingestion API endpoints.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.clients.postgres import get_async_db as get_db
from projects.syntraflow.api import router as syntraflow_router

# Create a test FastAPI app
test_app = FastAPI()
test_app.include_router(syntraflow_router)


@pytest.fixture
def mock_inference():
    return AsyncMock()


@pytest.fixture
def client(mock_inference):
    test_app.state.syntraflow_inference = mock_inference
    with TestClient(test_app) as c:
        yield c


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.add = MagicMock()  # db.add is synchronous in AsyncSession
    return db


@pytest.fixture(autouse=True)
def override_db(mock_db):
    test_app.dependency_overrides[get_db] = lambda: mock_db
    yield
    test_app.dependency_overrides.clear()


def test_status_endpoint(client) -> None:
    """Test that /status endpoint returns active status."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "syntraflow"
    assert data["status"] == "active"
    assert data["inference_connected"] is True


@pytest.mark.asyncio
async def test_ingest_duplicate_document(client, mock_db) -> None:
    """Test duplication checking on file upload."""
    mock_doc = MagicMock()
    mock_doc.id = uuid.uuid4()
    mock_doc.filename = "test.txt"

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = mock_doc

    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    mock_db.execute = AsyncMock(return_value=mock_result)

    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"hello world")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["skipped"] is True
    assert data["document_id"] == str(mock_doc.id)
    assert "Duplicate document detected" in data["message"]


@pytest.mark.asyncio
async def test_ingest_new_document(client, mock_db) -> None:
    """Test queuing ingestion for a new document."""
    # Mock no duplicate document
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)

    with patch("projects.syntraflow.api.publish_ingestion_job_to_kafka", return_value=True) as mock_kafka:
        response = client.post(
            "/ingest",
            files={"file": ("new.txt", b"unique content")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert "job_id" in data
        assert data["filename"] == "new.txt"
        assert mock_kafka.called


@pytest.mark.asyncio
async def test_ingest_new_document_kafka_offline_fallback(client, mock_db) -> None:
    """Test fallback to local background execution when Kafka broker is offline."""
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)

    # Mock publish failing (returns False)
    with patch("projects.syntraflow.api.publish_ingestion_job_to_kafka", return_value=False) as mock_kafka:
        with patch("projects.syntraflow.src.worker.process_ingestion_job") as mock_local_worker:
            response = client.post(
                "/ingest",
                files={"file": ("fallback.txt", b"fallback content")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert mock_kafka.called
            assert mock_local_worker.called


@pytest.mark.asyncio
async def test_get_job_status(client, mock_db) -> None:
    """Test retrieval of job status."""
    job_id = uuid.uuid4()
    mock_job = MagicMock()
    mock_job.id = job_id
    mock_job.document_id = uuid.uuid4()
    mock_job.status = "processing"
    mock_job.progress = 0.5
    mock_job.error_msg = None
    mock_job.created_at = None
    mock_job.updated_at = None

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = mock_job
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)

    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == str(job_id)
    assert data["status"] == "processing"
    assert data["progress"] == 0.5

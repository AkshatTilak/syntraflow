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


@pytest.mark.asyncio
async def test_delete_document_success(client, mock_db) -> None:
    """Test successful cascade deletion of a document."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock()
    mock_doc.id = doc_id

    # Configure db.execute to return mock values for the 4 queries
    # Query 1: Find document
    mock_scalars_doc = MagicMock()
    mock_scalars_doc.first.return_value = mock_doc
    mock_res_doc = MagicMock()
    mock_res_doc.scalars.return_value = mock_scalars_doc

    # Query 2: Chunk count
    mock_scalars_chunks = MagicMock()
    mock_scalars_chunks.all.return_value = [MagicMock(), MagicMock()]  # 2 chunks
    mock_res_chunks = MagicMock()
    mock_res_chunks.scalars.return_value = mock_scalars_chunks

    # Query 3: Video segments count
    mock_scalars_segs = MagicMock()
    mock_scalars_segs.all.return_value = []
    mock_res_segs = MagicMock()
    mock_res_segs.scalars.return_value = mock_scalars_segs

    # Query 4: Jobs count
    mock_scalars_jobs = MagicMock()
    mock_scalars_jobs.all.return_value = []
    mock_res_jobs = MagicMock()
    mock_res_jobs.scalars.return_value = mock_scalars_jobs

    mock_db.execute.side_effect = [
        mock_res_doc,
        mock_res_chunks,
        mock_res_segs,
        mock_res_jobs
    ]

    # Mock Qdrant and Neo4j clients
    with patch("projects.syntraflow.api.VectorClient") as mock_qdrant_cls, \
         patch("common.clients.neo4j.get_neo4j_driver") as mock_neo4j_driver:
         
         # Qdrant client mocks
         mock_qdrant_instance = MagicMock()
         mock_qdrant_cls.return_value = mock_qdrant_instance
         
         # Neo4j driver mocks
         mock_driver = MagicMock()
         mock_session = AsyncMock()
         
         # Mock execute_read_query / run summaries
         mock_summary = MagicMock()
         mock_summary.counters.relationships_deleted = 1
         mock_summary.counters.nodes_deleted = 2
         
         mock_result_cursor = AsyncMock()
         mock_result_cursor.consume.return_value = mock_summary
         mock_session.run.return_value = mock_result_cursor
         
         mock_driver.session.return_value.__aenter__.return_value = mock_session
         mock_neo4j_driver.return_value = mock_driver

         response = client.delete(f"/documents/{doc_id}")
         assert response.status_code == 200
         data = response.json()
         assert data["status"] == "success"
         assert data["document_id"] == str(doc_id)
         assert data["deleted_counts"]["postgres_chunks"] == 2
         assert data["deleted_counts"]["qdrant_vectors"] == 2
         assert data["deleted_counts"]["neo4j_nodes"] == 2
         assert data["deleted_counts"]["neo4j_edges"] == 1
         
         # Verify Postgres delete was called
         assert mock_db.delete.called
         assert mock_db.commit.called


@pytest.mark.asyncio
async def test_delete_document_not_found(client, mock_db) -> None:
    """Test 404 response when deleting a non-existent document."""
    doc_id = uuid.uuid4()
    
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute.return_value = mock_result

    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]

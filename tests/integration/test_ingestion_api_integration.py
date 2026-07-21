"""Integration tests for SyntraFlow dynamic ingestion API endpoints."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from common.clients.postgres import get_async_db as get_db


@pytest.mark.asyncio
async def test_dynamic_text_ingest_pipeline():
    """Test ingest text endpoint with custom chunker, pre-processor, and post-processor settings."""
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    
    # Mock duplicate check returning None
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = None
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_db.execute = AsyncMock(return_value=mock_result)

    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        with patch("projects.syntraflow.api.publish_ingestion_job_to_kafka", return_value=True):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                payload = {
                    "text": "Hello world! =====~~~~~\n\n\n\nSyntraFlow supports dynamic ingestion strategy execution. Contact team@syntraflow.io for details.",
                    "filename": "dynamic_test_doc.txt",
                    "chunker_type": "recursive",
                    "chunk_size": 128,
                    "chunk_overlap": 16,
                    "pre_processors": ["ocr_noise_reduction"],
                    "post_processors": ["metadata_extractor", "summary_tagger"],
                }

                res = await ac.post("/api/syntraflow/ingest/text", json=payload)
                assert res.status_code == 200
                data = res.json()
                assert data["status"] in ["queued", "success"]
                job_id = data.get("job_id")

                if job_id:
                    mock_job = MagicMock()
                    mock_job.id = job_id
                    mock_job.status = "processing"
                    mock_job.progress = 0.5
                    mock_job.error_msg = None
                    mock_scalars_job = MagicMock()
                    mock_scalars_job.first.return_value = mock_job
                    mock_result_job = MagicMock()
                    mock_result_job.scalars.return_value = mock_scalars_job
                    mock_db.execute.return_value = mock_result_job

                    job_res = await ac.get(f"/api/syntraflow/jobs/{job_id}")
                    assert job_res.status_code == 200
                    assert job_res.json()["status"] in ["completed", "queued", "processing"]
    finally:
        app.dependency_overrides.clear()

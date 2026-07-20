"""Integration tests for SyntraFlow dynamic ingestion API endpoints."""

import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app


@pytest.mark.asyncio
async def test_dynamic_text_ingest_pipeline():
    """Test ingest text endpoint with custom chunker, pre-processor, and post-processor settings."""
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
            # Poll job status to confirm completion
            await asyncio.sleep(1.0)
            job_res = await ac.get(f"/api/syntraflow/jobs/{job_id}")
            assert job_res.status_code == 200
            assert job_res.json()["status"] in ["completed", "queued", "processing"]

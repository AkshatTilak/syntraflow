"""SyntraFlow API routes.

Mounted at /api/syntraflow/* by the gateway's dynamic route loader.
Provides document ingestion and hybrid retrieval search.
"""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from common.config.settings import settings
from common.clients.postgres import get_async_db as get_db
from common.clients.qdrant import VectorClient
from projects.syntraflow.src.ingestion import ingest_document_pipeline, ingest_video_pipeline
from projects.syntraflow.src.retrieval import RetrievalEngine
from projects.syntraflow.src.database.models import SyntraFlowDocument, SyntraFlowJob

router = APIRouter(tags=["syntraflow"])
logger = logging.getLogger("syntraflow.api")


class SearchRequest(BaseModel):
    query: str
    strategy: Optional[str] = "hybrid"  # vector, graph, hybrid
    limit: Optional[int] = 5


class TextIngestRequest(BaseModel):
    text: str
    filename: str


@router.get("/status")
async def syntraflow_status(request: Request) -> dict:
    """SyntraFlow service status."""
    inference = getattr(request.app.state, "syntraflow_inference", None)
    return {
        "project": "syntraflow",
        "status": "active",
        "inference_connected": inference is not None,
    }


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def publish_ingestion_job_to_kafka(job_data: dict) -> bool:
    """Publish ingestion job details to Kafka topic 'syntraflow-ingestion-jobs'.

    Returns True if successfully published, False if Kafka is offline or fails.
    """
    try:
        from confluent_kafka import Producer
        conf = {"bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS}
        producer = Producer(conf)

        producer.produce(
            "syntraflow-ingestion-jobs",
            key=str(job_data.get("job_id")),
            value=json.dumps(job_data),
        )
        producer.flush(timeout=3.0)  # Wait up to 3 seconds for delivery report
        logger.info(
            "Ingestion job successfully published to Kafka: %s",
            job_data.get("job_id"),
        )
        return True
    except Exception as e:
        logger.warning(
            "Kafka broker unavailable: %s. Falling back to local background execution.",
            e,
        )
        return False


@router.post("/ingest")
async def ingest_file(
    request: Request,
    file: UploadFile = File(None),
    filepath: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Trigger document or video ingestion pipeline asynchronously.

    Supports either a direct multipart file upload, or a filepath on the local filesystem.
    """
    logger.info(
        "Received ingestion request (file=%s, filepath=%s)",
        file.filename if file else None,
        filepath,
    )

    # 1. Fetch bytes and filename
    if file:
        file_bytes = await file.read()
        filename = file.filename
    elif filepath:
        try:
            if not os.path.exists(filepath):
                raise HTTPException(
                    status_code=400, detail=f"File not found: {filepath}"
                )
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            filename = os.path.basename(filepath)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file from filepath: {str(e)}",
            )
    else:
        raise HTTPException(
            status_code=400, detail="Must provide either 'file' upload or 'filepath'"
        )

    # 2. Compute SHA-256 hash for duplicate checking
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # 3. Duplicate detection check (skip for videos, only for documents)
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    is_video_audio = ext in ["mp4", "avi", "mov", "mkv", "wav", "mp3", "flac"]

    if not is_video_audio:
        stmt = select(SyntraFlowDocument).where(
            SyntraFlowDocument.file_hash == file_hash
        )
        result = await db.execute(stmt)
        existing_doc = result.scalars().first()
        if existing_doc:
            logger.info(
                "Duplicate document detected for hash %s (ID: %s). Skipping ingestion.",
                file_hash,
                existing_doc.id,
            )
            return {
                "status": "success",
                "message": "Duplicate document detected. Skipping ingestion.",
                "document_id": str(existing_doc.id),
                "filename": existing_doc.filename,
                "skipped": True,
            }

    # 4. Create and persist Ingestion Job
    job = SyntraFlowJob(id=uuid.uuid4(), status="queued", progress=0.0)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # 5. Save uploaded file to temp directory (unless local file path was provided)
    if filepath and not file:
        temp_filepath = filepath
    else:
        temp_filename = f"{job.id}_{filename}"
        temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
        try:
            with open(temp_filepath, "wb") as f:
                f.write(file_bytes)
        except Exception as write_err:
            logger.error("Failed to save temporary upload file: %s", write_err)
            job.status = "failed"
            job.error_msg = f"Failed to save uploaded file: {str(write_err)}"
            await db.commit()
            raise HTTPException(
                status_code=500, detail="Failed to save uploaded file on Gateway."
            )

    # 6. Prepare Job payload
    job_payload = {
        "job_id": str(job.id),
        "file_hash": file_hash,
        "filename": filename,
        "temp_filepath": temp_filepath,
        "is_video_audio": is_video_audio,
    }

    # 7. Publish to Kafka or run locally in background task on error
    success = publish_ingestion_job_to_kafka(job_payload)
    if not success:
        from projects.syntraflow.src.worker import process_ingestion_job

        logger.info(
            "Kafka is offline. Launching local background task for job: %s", job.id
        )
        asyncio.create_task(
            process_ingestion_job(
                job_id=str(job.id),
                file_hash=file_hash,
                filename=filename,
                temp_filepath=temp_filepath,
                is_video_audio=is_video_audio,
            )
        )

    return {"status": "queued", "job_id": str(job.id), "filename": filename}


@router.post("/ingest/text")
async def ingest_raw_text(
    request: Request, req: TextIngestRequest, db: AsyncSession = Depends(get_db)
) -> dict:
    """Ingest raw text content directly (useful for testing/benchmarks)."""
    file_bytes = req.text.encode("utf-8")

    # 2. Compute SHA-256 hash for duplicate checking
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    stmt = select(SyntraFlowDocument).where(SyntraFlowDocument.file_hash == file_hash)
    result = await db.execute(stmt)
    existing_doc = result.scalars().first()
    if existing_doc:
        logger.info(
            "Duplicate document detected for hash %s (ID: %s). Skipping ingestion.",
            file_hash,
            existing_doc.id,
        )
        return {
            "status": "success",
            "message": "Duplicate document detected. Skipping ingestion.",
            "document_id": str(existing_doc.id),
            "filename": existing_doc.filename,
            "skipped": True,
        }

    # Create Ingestion Job
    job = SyntraFlowJob(id=uuid.uuid4(), status="queued", progress=0.0)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Save to temp uploads
    temp_filename = f"{job.id}_{req.filename}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
    try:
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)
    except Exception as write_err:
        logger.error("Failed to save temporary upload file: %s", write_err)
        job.status = "failed"
        job.error_msg = f"Failed to save uploaded file: {str(write_err)}"
        await db.commit()
        raise HTTPException(status_code=500, detail="Failed to save text on Gateway.")

    # Prepare Job payload
    job_payload = {
        "job_id": str(job.id),
        "file_hash": file_hash,
        "filename": req.filename,
        "temp_filepath": temp_filepath,
        "is_video_audio": False,
    }

    # Publish to Kafka or run locally in background task on error
    success = publish_ingestion_job_to_kafka(job_payload)
    if not success:
        from projects.syntraflow.src.worker import process_ingestion_job

        logger.info(
            "Kafka is offline. Launching local background task for job: %s", job.id
        )
        asyncio.create_task(
            process_ingestion_job(
                job_id=str(job.id),
                file_hash=file_hash,
                filename=req.filename,
                temp_filepath=temp_filepath,
                is_video_audio=False,
            )
        )

    return {"status": "queued", "job_id": str(job.id), "filename": req.filename}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Fetch status and progress details of a SyntraFlow ingestion job."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

    stmt = select(SyntraFlowJob).where(SyntraFlowJob.id == job_uuid)
    result = await db.execute(stmt)
    job = result.scalars().first()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": str(job.id),
        "document_id": str(job.document_id) if job.document_id else None,
        "status": job.status,
        "progress": job.progress,
        "error_msg": job.error_msg,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }


@router.post("/search")
async def search_documents(request: Request, req: SearchRequest) -> dict:
    """Search indexed documents via Vector, Graph, or Hybrid retrieval."""
    vector_client = VectorClient()
    engine = RetrievalEngine(vector_client)
    inference_client = request.app.state.syntraflow_inference

    try:
        # Embed query text
        try:
            embeds = await inference_client.embed(texts=[req.query])
            query_vector = embeds[0]
        except Exception:
            logger.warning("Embedding generation failed, utilizing zero-vector.")
            query_vector = [0.0] * 768

        if req.strategy == "vector":
            results = await engine.search_vector(query_vector, limit=req.limit)
        elif req.strategy == "graph":
            results = await engine.search_graph(req.query, limit=req.limit)
        else:
            results = await engine.search_hybrid(req.query, query_vector, limit=req.limit)

        return {
            "status": "success",
            "query": req.query,
            "strategy": req.strategy,
            "results": results
        }
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval query failed: {str(e)}")

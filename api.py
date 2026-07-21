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
    chunker_type: Optional[str] = "recursive"
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 64
    pre_processors: Optional[list[str]] = None
    post_processors: Optional[list[str]] = None


@router.get("/status")
async def syntraflow_status(request: Request) -> dict:
    """SyntraFlow service status."""
    inference = getattr(request.app.state, "syntraflow_inference", None)
    return {
        "project": "syntraflow",
        "status": "active",
        "inference_connected": inference is not None,
    }


# File Size & Format Limits Constraints
MAX_DOC_SIZE = 100 * 1024 * 1024   # 100 MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024 # 500 MB
MAX_AUDIO_SIZE = 200 * 1024 * 1024 # 200 MB

DOC_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".pptx"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

import re

ALLOWED_MIME_TYPES = {
    # Documents
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # pptx
    # Images
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/gif",
    # Videos
    "video/mp4",
    "video/quicktime", # mov
    "video/webm",
    "video/x-matroska", # mkv
    # Audio
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg", # mp3
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/x-flac",
    "application/octet-stream"
}


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename to prevent path traversal and ensure safe characters."""
    base = os.path.basename(filename)
    # Remove path traversal sequences
    base = base.replace("..", "").replace("/", "").replace("\\", "")
    # Keep only alphanumeric, dots, dashes, underscores
    base = re.sub(r"[^\w\.\-_]", "_", base)
    return base


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
    chunk_strategy: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    ocr_cleanup: Optional[bool] = Form(None),
    lang_filter: Optional[bool] = Form(None),
    extract_metadata: Optional[bool] = Form(None),
    generate_summary: Optional[bool] = Form(None),
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
        content_type = file.content_type
        if content_type and content_type not in ALLOWED_MIME_TYPES:
            is_valid_prefix = any(
                content_type.startswith(prefix)
                for prefix in ["image/", "video/", "audio/", "application/pdf"]
            )
            if not is_valid_prefix:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unauthorized/Unsupported Content-Type header: '{content_type}'"
                )

        filename = sanitize_filename(file.filename)
        _, ext = os.path.splitext(filename.lower())

        if ext in DOC_EXTENSIONS:
            max_size = MAX_DOC_SIZE
            category = "Document"
        elif ext in VIDEO_EXTENSIONS:
            max_size = MAX_VIDEO_SIZE
            category = "Video"
        elif ext in AUDIO_EXTENSIONS:
            max_size = MAX_AUDIO_SIZE
            category = "Audio"
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file format: {ext}. Supported formats are: "
                    f"Documents ({', '.join(sorted(DOC_EXTENSIONS))}), "
                    f"Videos ({', '.join(sorted(VIDEO_EXTENSIONS))}), "
                    f"Audio ({', '.join(sorted(AUDIO_EXTENSIONS))})."
                ),
            )

        file_size = file.size if file.size is not None else 0
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit for {category} ({max_size / (1024 * 1024):.0f} MB). Uploaded size: {file_size / (1024 * 1024):.2f} MB.",
            )

        file_bytes = await file.read()
        if len(file_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds limit for {category} ({max_size / (1024 * 1024):.0f} MB). Uploaded size: {len(file_bytes) / (1024 * 1024):.2f} MB.",
            )
    elif filepath:
        try:
            if not os.path.exists(filepath):
                raise HTTPException(
                    status_code=400, detail=f"File not found: {filepath}"
                )
            filename = sanitize_filename(os.path.basename(filepath))
            _, ext = os.path.splitext(filename.lower())

            if ext in DOC_EXTENSIONS:
                max_size = MAX_DOC_SIZE
                category = "Document"
            elif ext in VIDEO_EXTENSIONS:
                max_size = MAX_VIDEO_SIZE
                category = "Video"
            elif ext in AUDIO_EXTENSIONS:
                max_size = MAX_AUDIO_SIZE
                category = "Audio"
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file format: {ext}. Supported formats are: "
                        f"Documents ({', '.join(sorted(DOC_EXTENSIONS))}), "
                        f"Videos ({', '.join(sorted(VIDEO_EXTENSIONS))}), "
                        f"Audio ({', '.join(sorted(AUDIO_EXTENSIONS))})."
                    ),
                )

            file_size = os.path.getsize(filepath)
            if file_size > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds limit for {category} ({max_size / (1024 * 1024):.0f} MB). Uploaded size: {file_size / (1024 * 1024):.2f} MB.",
                )

            with open(filepath, "rb") as f:
                file_bytes = f.read()
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

    # 6. Prepare pre/post processor lists
    pre_procs = []
    if ocr_cleanup:
        pre_procs.append("ocr_cleanup")
    if lang_filter:
        pre_procs.append("language_filter")

    post_procs = []
    if extract_metadata:
        post_procs.append("metadata_extractor")
    if generate_summary:
        post_procs.append("summary_gen")

    job_payload = {
        "job_id": str(job.id),
        "file_hash": file_hash,
        "filename": filename,
        "temp_filepath": temp_filepath,
        "is_video_audio": is_video_audio,
        "chunker_type": chunk_strategy,
        "chunk_size": chunk_size or 512,
        "chunk_overlap": chunk_overlap or 64,
        "pre_processors": pre_procs,
        "post_processors": post_procs,
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
                chunker_type=chunk_strategy,
                chunk_size=chunk_size or 512,
                chunk_overlap=chunk_overlap or 64,
                pre_processors=pre_procs,
                post_processors=post_procs,
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
                chunker_type=req.chunker_type,
                chunk_size=req.chunk_size or 512,
                chunk_overlap=req.chunk_overlap or 64,
                pre_processors=req.pre_processors,
                post_processors=req.post_processors,
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


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Cascade delete document and related chunks / graph nodes / vectors."""
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")

    # 1. Fetch document from PostgreSQL to ensure it exists
    stmt = select(SyntraFlowDocument).where(SyntraFlowDocument.id == doc_uuid)
    result = await db.execute(stmt)
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    deleted_counts = {
        "postgres_chunks": 0,
        "postgres_video_segments": 0,
        "postgres_jobs": 0,
        "qdrant_vectors": 0,
        "neo4j_nodes": 0,
        "neo4j_edges": 0,
    }

    # Count database items for reporting
    from projects.syntraflow.src.database.models import SyntraFlowChunk, SyntraFlowVideoSegment, SyntraFlowJob
    
    chunk_count_stmt = select(SyntraFlowChunk).where(SyntraFlowChunk.document_id == doc_uuid)
    chunk_result = await db.execute(chunk_count_stmt)
    deleted_counts["postgres_chunks"] = len(chunk_result.scalars().all())

    video_seg_count_stmt = select(SyntraFlowVideoSegment).where(SyntraFlowVideoSegment.document_id == doc_uuid)
    video_seg_result = await db.execute(video_seg_count_stmt)
    deleted_counts["postgres_video_segments"] = len(video_seg_result.scalars().all())

    job_count_stmt = select(SyntraFlowJob).where(SyntraFlowJob.document_id == doc_uuid)
    job_result = await db.execute(job_count_stmt)
    deleted_counts["postgres_jobs"] = len(job_result.scalars().all())

    # 2. Delete vectors from Qdrant by document ID filter
    try:
        from qdrant_client.http import models
        vector_client = VectorClient()
        vector_client.get_client().delete(
            collection_name="syntraflow_chunks_v1",
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(doc_uuid)),
                        )
                    ]
                )
            )
        )
        deleted_counts["qdrant_vectors"] = deleted_counts["postgres_chunks"] + deleted_counts["postgres_video_segments"]
    except Exception as qdrant_err:
        logger.error("Failed to delete vectors from Qdrant: %s", qdrant_err)

    # 3. Delete graph nodes/edges from Neo4j by document reference
    try:
        from common.clients.neo4j import get_neo4j_driver
        driver = get_neo4j_driver()
        async with driver.session() as session:
            # Delete relationships (edges) with document_id matching doc_id
            edge_res = await session.run(
                "MATCH ()-[r:SyntraFlow_RELATION {document_id: $doc_id}]->() "
                "DELETE r",
                doc_id=str(doc_uuid)
            )
            edge_summary = await edge_res.consume()
            deleted_counts["neo4j_edges"] = edge_summary.counters.relationships_deleted

            # Delete entities (nodes) with document_id matching doc_id
            node_res = await session.run(
                "MATCH (e:SyntraFlow_Entity {document_id: $doc_id}) "
                "DETACH DELETE e",
                doc_id=str(doc_uuid)
            )
            node_summary = await node_res.consume()
            deleted_counts["neo4j_nodes"] = node_summary.counters.nodes_deleted
    except Exception as neo4j_err:
        logger.warning("Failed to delete graph nodes/edges from Neo4j: %s", neo4j_err)

    # 4. Cascade delete the document from PostgreSQL (deletes related chunks/video segments/jobs)
    await db.delete(doc)
    await db.commit()

    return {
        "status": "success",
        "message": f"Document {doc_id} successfully deleted cascade-wide.",
        "document_id": doc_id,
        "deleted_counts": deleted_counts,
    }

"""SyntraFlow API routes.

Mounted at /api/syntraflow/* by the gateway's dynamic route loader.
Provides document ingestion and hybrid retrieval search.
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from projects.syntraflow.src.database.client import get_db
from projects.syntraflow.src.vectors.client import VectorClient
from projects.syntraflow.src.ingestion import ingest_document_pipeline, ingest_video_pipeline
from projects.syntraflow.src.retrieval import RetrievalEngine

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


@router.post("/ingest")
async def ingest_file(
    request: Request,
    file: UploadFile = File(None),
    filepath: Optional[str] = Form(None)
) -> dict:
    """Trigger document or video ingestion pipeline.

    Supports either a direct multipart file upload, or a filepath on the local filesystem.
    """
    logger.info("Received ingestion request (file=%s, filepath=%s)", 
                file.filename if file else None, filepath)
    
    # 1. Fetch bytes and filename
    if file:
        file_bytes = await file.read()
        filename = file.filename
    elif filepath:
        try:
            import os
            if not os.path.exists(filepath):
                raise HTTPException(status_code=400, detail=f"File not found: {filepath}")
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            filename = os.path.basename(filepath)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file from filepath: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'file' upload or 'filepath'")

    # 2. Get DB, Inference, Vector clients
    db_session = next(get_db())
    inference_client = request.app.state.syntraflow_inference
    vector_client = VectorClient()

    # 3. Route based on file type
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    is_video_audio = ext in ["mp4", "avi", "mov", "mkv", "wav", "mp3", "flac"]

    try:
        if is_video_audio:
            segment_ids = await ingest_video_pipeline(
                video_bytes=file_bytes,
                video_name=filename,
                db=db_session,
                inference_client=inference_client,
                vector_client=vector_client
            )
            return {
                "status": "success",
                "type": "video_audio",
                "filename": filename,
                "segments_count": len(segment_ids),
                "segment_ids": segment_ids
            }
        else:
            doc_id = await ingest_document_pipeline(
                file_bytes=file_bytes,
                filename=filename,
                db=db_session,
                inference_client=inference_client,
                vector_client=vector_client
            )
            return {
                "status": "success",
                "type": "document",
                "filename": filename,
                "document_id": doc_id
            }
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")


@router.post("/ingest/text")
async def ingest_raw_text(request: Request, req: TextIngestRequest) -> dict:
    """Ingest raw text content directly (useful for testing/benchmarks)."""
    db_session = next(get_db())
    inference_client = request.app.state.syntraflow_inference
    vector_client = VectorClient()

    try:
        doc_id = await ingest_document_pipeline(
            file_bytes=req.text.encode("utf-8"),
            filename=req.filename,
            db=db_session,
            inference_client=inference_client,
            vector_client=vector_client
        )
        return {
            "status": "success",
            "document_id": doc_id,
            "filename": req.filename
        }
    except Exception as e:
        logger.error("Text ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")


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

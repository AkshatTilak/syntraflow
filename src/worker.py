"""Asynchronous ingestion worker loop for SyntraFlow.

Supports consuming jobs from Kafka or executing them locally if Kafka is offline.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.postgres import get_sessionmaker
from projects.syntraflow.src.ingestion import (
    ingest_document_pipeline,
    ingest_video_pipeline,
    update_job,
)

logger = logging.getLogger("syntraflow.worker")


async def process_ingestion_job(
    job_id: str,
    file_hash: str,
    filename: str,
    temp_filepath: str,
    is_video_audio: bool = False,
    *,
    hub_id: str,
    collection_id: str,
    chunker_type: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    pre_processors: Optional[list[str]] = None,
    post_processors: Optional[list[str]] = None,
    pipeline_config: Optional[dict] = None,
    db_session: Optional[Any] = None,
) -> None:
    """Load file bytes from disk, execute the hub-scoped ingestion pipeline, and clean up."""
    logger.info(
        "Starting ingestion processing for Job ID: %s (%s) in hub %s collection %s",
        job_id,
        filename,
        hub_id,
        collection_id,
    )

    if db_session is not None:
        await _process_ingestion_job_inner(
            db=db_session,
            job_id=job_id,
            filename=filename,
            temp_filepath=temp_filepath,
            is_video_audio=is_video_audio,
            hub_id=hub_id,
            collection_id=collection_id,
            chunker_type=chunker_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pre_processors=pre_processors,
            post_processors=post_processors,
            pipeline_config=pipeline_config,
        )
    else:
        SessionLocal = get_sessionmaker()
        async with SessionLocal() as db:
            await _process_ingestion_job_inner(
                db=db,
                job_id=job_id,
                filename=filename,
                temp_filepath=temp_filepath,
                is_video_audio=is_video_audio,
                hub_id=hub_id,
                collection_id=collection_id,
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pre_processors=pre_processors,
                post_processors=post_processors,
                pipeline_config=pipeline_config,
            )


async def _process_ingestion_job_inner(
    db: Any,
    job_id: str,
    filename: str,
    temp_filepath: str,
    is_video_audio: bool,
    hub_id: str,
    collection_id: str,
    chunker_type: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    pre_processors: Optional[list[str]],
    post_processors: Optional[list[str]],
    pipeline_config: Optional[dict],
) -> None:
    inference_client = None
    try:
        # 1. Read file bytes
        if not os.path.exists(temp_filepath):
            raise FileNotFoundError(f"Source file not found at: {temp_filepath}")

        with open(temp_filepath, "rb") as f:
            file_bytes = f.read()

        # 2. Setup inference client
        inference_client = InferenceClient(base_url=settings.INFERENCE_SERVER_URL)

        # 3. Route to proper pipeline with hub_id, collection_id, and pipeline_config
        if is_video_audio:
            await ingest_video_pipeline(
                video_bytes=file_bytes,
                video_name=filename,
                db=db,
                inference_client=inference_client,
                hub_id=hub_id,
                collection_id=collection_id,
                job_id=job_id,
                pipeline_config=pipeline_config,
            )
        else:
            await ingest_document_pipeline(
                file_bytes=file_bytes,
                filename=filename,
                db=db,
                inference_client=inference_client,
                hub_id=hub_id,
                collection_id=collection_id,
                job_id=job_id,
                chunker_type=chunker_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pre_processors=pre_processors,
                post_processors=post_processors,
                pipeline_config=pipeline_config,
            )
        logger.info("Successfully completed Ingestion Job: %s", job_id)

    except Exception as e:
        logger.error("Failed to execute Ingestion Job %s: %s", job_id, e)
        await update_job(
            db=db,
            job_id=job_id,
            progress=1.0,
            status="failed",
            error_msg=str(e),
        )
    finally:
        if inference_client:
            await inference_client.close()

        # Clean up temp upload file if applicable
        if "temp_uploads" in temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logger.debug("Cleaned up temporary upload file: %s", temp_filepath)
            except Exception as cleanup_err:
                logger.warning("Failed to remove temp file %s: %s", temp_filepath, cleanup_err)


async def run_ingestion_consumer(app) -> None:
    """Run Kafka consumer loop for syntraflow-ingestion-jobs."""
    try:
        import socket
        host, port = settings.KAFKA_BOOTSTRAP_SERVERS.split(":")[0], int(settings.KAFKA_BOOTSTRAP_SERVERS.split(":")[1]) if ":" in settings.KAFKA_BOOTSTRAP_SERVERS else 9092
        with socket.create_connection((host, port), timeout=0.2):
            pass
    except Exception:
        logger.info("Kafka broker (%s) is offline. SyntraFlow will use local background execution.", settings.KAFKA_BOOTSTRAP_SERVERS)
        return

    logger.info("Initializing SyntraFlow Kafka Ingestion Consumer...")
    try:
        from confluent_kafka import Consumer, KafkaError
        conf = {
            "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "syntraflow-ingestion-group",
            "auto.offset.reset": "earliest",
        }
        consumer = Consumer(conf)
        consumer.subscribe(["syntraflow-ingestion-jobs"])
    except Exception as e:
        logger.warning(
            "Kafka consumer initialization failed: %s. Kafka consumer loop will not start.",
            e,
        )
        return

    logger.info("SyntraFlow Kafka Consumer started.")

    try:
        while True:
            # Poll for messages asynchronously using run_in_executor to avoid blocking the event loop
            msg = await asyncio.to_thread(consumer.poll, 1.0)
            if msg is None:
                continue
            if msg.error():
                err_code = msg.error().code()
                if err_code in (
                    KafkaError._PARTITION_EOF,
                    KafkaError._UNKNOWN_TOPIC,
                    KafkaError._UNKNOWN_PARTITION,
                    KafkaError.UNKNOWN_TOPIC_OR_PART,
                ):
                    await asyncio.sleep(1.0)
                    continue
                else:
                    logger.error("Kafka consumer error: %s", msg.error())
                    await asyncio.sleep(2.0)
                    continue

            # Parse message and run job
            try:
                job_data = json.loads(msg.value().decode("utf-8"))
                job_id = job_data["job_id"]
                file_hash = job_data["file_hash"]
                filename = job_data["filename"]
                temp_filepath = job_data["temp_filepath"]
                is_video_audio = job_data["is_video_audio"]
                hub_id = job_data.get("hub_id")
                collection_id = job_data.get("collection_id")

                if not hub_id or not collection_id:
                    logger.error("Dead-lettering Kafka message for job %s: missing required 'hub_id' or 'collection_id'", job_id)
                    continue

                # Process job in background
                asyncio.create_task(
                    process_ingestion_job(
                        job_id=job_id,
                        file_hash=file_hash,
                        filename=filename,
                        temp_filepath=temp_filepath,
                        is_video_audio=is_video_audio,
                        hub_id=hub_id,
                        collection_id=collection_id,
                        chunker_type=job_data.get("chunker_type"),
                        chunk_size=job_data.get("chunk_size", 512),
                        chunk_overlap=job_data.get("chunk_overlap", 64),
                        pre_processors=job_data.get("pre_processors"),
                        post_processors=job_data.get("post_processors"),
                    )
                )
            except Exception as pe:
                logger.error("Failed to parse or trigger job from Kafka message: %s", pe)
    except asyncio.CancelledError:
        logger.info("Kafka consumer loop cancelled.")
    except Exception as run_err:
        logger.error("Kafka consumer run encountered error: %s", run_err)
    finally:
        try:
            consumer.close()
        except Exception:
            pass

"""Ingestion Pipeline Trace Logging & Execution Metrics (sub_07_03)."""

import time
import logging
from contextlib import contextmanager
from typing import Generator, Optional

logger = logging.getLogger("syntraflow.ingestion")


@contextmanager
def log_pipeline_stage(
    stage_name: str,
    document_id: Optional[str] = None,
    hub_id: Optional[str] = None,
) -> Generator[None, None, None]:
    """Context manager to measure and log execution timing and trace context for ingestion stages."""
    start_time = time.perf_counter()
    doc_str = f" [doc_id={document_id}]" if document_id else ""
    hub_str = f" [hub_id={hub_id}]" if hub_id else ""
    
    logger.info("Starting ingestion stage '%s'%s%s...", stage_name, doc_str, hub_str)
    try:
        yield
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "Completed ingestion stage '%s'%s%s in %.2f ms",
            stage_name, doc_str, hub_str, elapsed_ms
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            "Failed ingestion stage '%s'%s%s after %.2f ms: %s",
            stage_name, doc_str, hub_str, elapsed_ms, e, exc_info=True
        )
        raise

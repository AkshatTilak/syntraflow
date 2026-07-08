"""SyntraFlow project setup hook.

Called by the gateway's lifespan factory during startup/shutdown.
Initializes PostgreSQL database tables, Qdrant collections, and the inference client.
"""

from fastapi import FastAPI
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from common.clients.inference import InferenceClient
from common.clients.qdrant import VectorClient
from common.observability.logger import get_logger

logger = get_logger("syntraflow")


async def init_app_state(app: FastAPI, settings) -> None:
    """Initialize SyntraFlow database schemas, collections, and state on gateway startup."""
    # 1. Ensure Qdrant collection exists with correct vector dimension
    try:
        v_client = VectorClient()
        qc = v_client.get_client()
        collection_name = "syntraflow_chunks_v1"

        # Dynamically resolve vector dimension from model registry
        vector_dim = await v_client.get_vector_dimension()

        # Check if collection exists
        collections = qc.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            qc.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection: %s (dim=%d)", collection_name, vector_dim
            )
        else:
            # Verify existing collection dimension matches active embedding model
            collection_info = qc.get_collection(collection_name)
            existing_dim = collection_info.config.params.vectors.size
            if existing_dim != vector_dim:
                logger.warning(
                    "Qdrant collection '%s' dimension mismatch: existing=%d, active_model=%d. "
                    "Re-indexing may be required if you switch embedding models.",
                    collection_name,
                    existing_dim,
                    vector_dim,
                )
            else:
                logger.info(
                    "Qdrant collection '%s' verified (dim=%d).",
                    collection_name,
                    vector_dim,
                )
    except Exception as e:
        logger.error("Failed to verify/create Qdrant collection: %s", e)

    # 2. Setup Inference client
    app.state.syntraflow_inference = InferenceClient(
        base_url=settings.INFERENCE_SERVER_URL,
    )
    logger.info(
        "SyntraFlow initialized — inference client connected to %s",
        settings.INFERENCE_SERVER_URL,
    )


async def shutdown_app_state(app: FastAPI, settings) -> None:
    """Clean up SyntraFlow state on gateway shutdown."""
    if hasattr(app.state, "syntraflow_inference"):
        await app.state.syntraflow_inference.close()
    logger.info("SyntraFlow shut down")

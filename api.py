"""SyntraFlow service routes.

Flat legacy routes have been decommissioned in V6 in favor of hub-scoped API endpoints
mounted at /api/hubs/{hub_id}/... (S6-04e & S6-04f).
Only GET /status remains as the microservice health probe.
"""

import logging
from fastapi import APIRouter, Request

router = APIRouter(tags=["syntraflow"])
logger = logging.getLogger("syntraflow.api")


@router.get("/status")
async def syntraflow_status(request: Request) -> dict:
    """SyntraFlow service status."""
    inference = getattr(request.app.state, "syntraflow_inference", None)
    return {
        "project": "syntraflow",
        "status": "active",
        "inference_connected": inference is not None,
    }

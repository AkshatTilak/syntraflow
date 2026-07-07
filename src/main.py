"""Main entry point for the SyntraFlow FastAPI application."""

from fastapi import FastAPI
from common.config import settings

app = FastAPI(
    title="SyntraFlow Engine",
    description="Multi-Agent Document Extraction & Synthesis Engine",
    version="0.1.0",
)


@app.get("/")
async def read_root() -> dict[str, str]:
    """Root endpoint to check API name and status.

    Returns:
        A dictionary indicating the API status.
    """
    return {"app": "SyntraFlow Engine", "status": "running"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        A dictionary containing environment status and service health.
    """
    return {
        "status": "healthy",
        "environment": settings.app_env,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )

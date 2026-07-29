"""Pydantic schemas for SyntraFlow collections and retrieval configuration (S6-04a)."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CollectionRetrievalConfig(BaseModel):
    """Configuration for per-collection retrieval strategy."""

    strategy: str = Field(default="hybrid", description="dense | sparse | hybrid | graph")
    top_k: int = Field(default=5, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    graph_depth: int = Field(default=2, ge=1, le=5)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"dense", "sparse", "hybrid", "graph"}
        lowered = v.strip().lower()
        if lowered not in valid:
            raise ValueError(f"Invalid retrieval strategy '{v}'. Must be one of {valid}")
        return lowered

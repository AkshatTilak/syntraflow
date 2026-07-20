"""SyntraFlow Abstract Ingestion Strategy Interfaces.

Defines the core Strategy Pattern contracts for customizable text chunking,
raw data pre-processing, and metadata post-processing enrichment.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    """Configuration parameters for text chunking strategies."""

    strategy: str = Field(default="recursive", description="Strategy identifier (recursive, semantic, fixed)")
    chunk_size: int = Field(default=512, gt=0, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=64, ge=0, description="Overlap between consecutive chunks")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional strategy-specific settings")


class PreProcessorConfig(BaseModel):
    """Configuration parameters for pre-processing strategies."""

    strategy: str = Field(..., description="Pre-processor strategy identifier (ocr_cleanup, noise_filter, format_normalize)")
    options: Dict[str, Any] = Field(default_factory=dict, description="Pre-processor specific options")


class PostProcessorConfig(BaseModel):
    """Configuration parameters for post-processing strategies."""

    strategy: str = Field(..., description="Post-processor strategy identifier (entity_extraction, summary_gen, metadata_enricher)")
    options: Dict[str, Any] = Field(default_factory=dict, description="Post-processor specific options")


class BaseChunker(ABC):
    """Abstract base class for document text chunking strategies."""

    def __init__(self, config: Optional[ChunkerConfig] = None) -> None:
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Split raw text into structured chunk dictionaries.

        Args:
            text: Raw document text string.

        Returns:
            List of chunk dicts containing 'text', 'index', and optional metadata.
        """
        pass


class BasePreProcessor(ABC):
    """Abstract base class for data pre-processing strategies."""

    def __init__(self, config: Optional[PreProcessorConfig] = None) -> None:
        self.config = config

    @abstractmethod
    def process(self, data: bytes) -> bytes:
        """Pre-process raw binary data or text before ingestion.

        Args:
            data: Raw input bytes.

        Returns:
            Cleaned/processed bytes.
        """
        pass


class BasePostProcessor(ABC):
    """Abstract base class for chunk metadata post-processing strategies."""

    def __init__(self, config: Optional[PostProcessorConfig] = None) -> None:
        self.config = config

    @abstractmethod
    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a chunk dictionary with additional metadata or annotations.

        Args:
            chunk: Input chunk dict containing 'text' and 'metadata'.

        Returns:
            Enriched chunk dict.
        """
        pass

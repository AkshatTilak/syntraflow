"""Unit tests for SyntraFlow Ingestion Strategy interfaces."""

import pytest
from typing import Any, Dict, List
from projects.syntraflow.src.ingestion.strategies import (
    BaseChunker,
    BasePreProcessor,
    BasePostProcessor,
    ChunkerConfig,
    PreProcessorConfig,
    PostProcessorConfig,
)


def test_abstract_class_instantiation_raises_type_error():
    """Verify abstract strategy classes cannot be instantiated without abstract methods."""
    with pytest.raises(TypeError):
        BaseChunker()

    with pytest.raises(TypeError):
        BasePreProcessor()

    with pytest.raises(TypeError):
        BasePostProcessor()


class ConcreteChunker(BaseChunker):
    """Dummy concrete chunker for testing."""

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        return [{"text": text, "index": 0}]


class ConcretePreProcessor(BasePreProcessor):
    """Dummy concrete pre-processor for testing."""

    def process(self, data: bytes) -> bytes:
        return data.strip()


class ConcretePostProcessor(BasePostProcessor):
    """Dummy concrete post-processor for testing."""

    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        chunk["enriched"] = True
        return chunk


def test_concrete_implementations_work():
    """Verify concrete strategy subclasses instantiate and run correctly."""
    config = ChunkerConfig(chunk_size=256, chunk_overlap=32)
    chunker = ConcreteChunker(config=config)
    res = chunker.chunk("Hello world")
    assert len(res) == 1
    assert res[0]["text"] == "Hello world"
    assert chunker.config.chunk_size == 256

    pre_processor = ConcretePreProcessor(PreProcessorConfig(strategy="trim"))
    assert pre_processor.process(b"  hello  ") == b"hello"

    post_processor = ConcretePostProcessor(PostProcessorConfig(strategy="annotate"))
    enriched = post_processor.enrich({"text": "sample", "metadata": {}})
    assert enriched["enriched"] is True

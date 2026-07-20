"""Unit tests for SyntraFlow chunker implementations."""

import pytest
from projects.syntraflow.src.ingestion.strategies import ChunkerConfig
from projects.syntraflow.src.ingestion.chunkers import (
    FixedSizeChunking,
    RecursiveCharacterChunking,
    SemanticChunking,
    get_chunker,
)


def test_fixed_size_chunking():
    """Verify FixedSizeChunking creates character windows with overlap."""
    config = ChunkerConfig(chunk_size=10, chunk_overlap=2)
    chunker = FixedSizeChunking(config)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text)
    
    assert len(chunks) > 1
    assert chunks[0]["text"] == "0123456789"
    assert chunks[0]["strategy"] == "fixed"
    assert chunks[1]["char_start"] == 8


def test_recursive_character_chunking():
    """Verify RecursiveCharacterChunking splits on paragraph and sentence boundaries."""
    config = ChunkerConfig(chunk_size=30, chunk_overlap=0)
    chunker = RecursiveCharacterChunking(config)
    text = "Paragraph one is here.\n\nParagraph two is here.\n\nParagraph three."
    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0]["text"] == "Paragraph one is here."
    assert chunks[1]["text"] == "Paragraph two is here."
    assert chunks[2]["text"] == "Paragraph three."
    assert chunks[0]["strategy"] == "recursive"


def test_semantic_chunking_with_mock_embedding():
    """Verify SemanticChunking groups sentences by vector similarity."""
    def dummy_embeddings(sentences):
        # Return identical vector for sentence 0 & 1, different vector for sentence 2
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    config = ChunkerConfig(chunk_size=100)
    chunker = SemanticChunking(config=config, embedding_fn=dummy_embeddings, similarity_threshold=0.8)
    text = "Sentence A is first. Sentence B is second. Sentence C is about something else."
    chunks = chunker.chunk(text)

    assert len(chunks) == 2
    assert "Sentence A is first. Sentence B is second." in chunks[0]["text"]
    assert chunks[1]["text"] == "Sentence C is about something else."
    assert chunks[0]["strategy"] == "semantic"


def test_semantic_chunking_error_fallback():
    """Verify SemanticChunking falls back to recursive chunking on embedding failure."""
    def failing_embeddings(sentences):
        raise RuntimeError("Embedding service unavailable")

    config = ChunkerConfig(chunk_size=40)
    chunker = SemanticChunking(config=config, embedding_fn=failing_embeddings)
    text = "This is sentence one. This is sentence two.\n\nThis is paragraph two."
    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert chunks[0]["strategy"] == "recursive_fallback"


def test_get_chunker_factory():
    """Verify get_chunker constructs appropriate strategy instances."""
    fixed = get_chunker(ChunkerConfig(strategy="fixed"))
    assert isinstance(fixed, FixedSizeChunking)

    semantic = get_chunker(ChunkerConfig(strategy="semantic"))
    assert isinstance(semantic, SemanticChunking)

    recursive = get_chunker(ChunkerConfig(strategy="recursive"))
    assert isinstance(recursive, RecursiveCharacterChunking)

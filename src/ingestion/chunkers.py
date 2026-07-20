"""Concrete Chunking Implementations for SyntraFlow.

Includes FixedSizeChunking, RecursiveCharacterChunking, and SemanticChunking
with automatic error-recovery fallback.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from projects.syntraflow.src.ingestion.strategies import BaseChunker, ChunkerConfig

logger = logging.getLogger("syntraflow.chunkers")


class FixedSizeChunking(BaseChunker):
    """Fixed-length character window chunker with overlapping boundaries."""

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        step = max(1, chunk_size - chunk_overlap)

        chunks = []
        index = 0
        for i in range(0, len(text), step):
            chunk_str = text[i : i + chunk_size]
            if chunk_str:
                chunks.append({
                    "text": chunk_str,
                    "index": index,
                    "strategy": "fixed",
                    "char_start": i,
                    "char_end": i + len(chunk_str),
                })
                index += 1
        return chunks


class RecursiveCharacterChunking(BaseChunker):
    """Recursive chunker that splits on hierarchical boundaries (paragraphs, lines, sentences)."""

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        raw_chunks = self._recursive_split(text, self.SEPARATORS)
        chunk_objs = []
        for idx, c_text in enumerate(raw_chunks):
            if c_text.strip():
                chunk_objs.append({
                    "text": c_text.strip(),
                    "index": idx,
                    "strategy": "recursive",
                })
        return chunk_objs

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        chunk_size = self.config.chunk_size

        if len(text) <= chunk_size or not separators:
            return [text]

        separator = separators[0]
        next_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        final_chunks = []
        current_chunk = ""

        for s in splits:
            item = s + separator if separator else s
            if len(current_chunk) + len(item) <= chunk_size:
                current_chunk += item
            else:
                if current_chunk:
                    final_chunks.append(current_chunk)
                if len(item) > chunk_size and next_separators:
                    final_chunks.extend(self._recursive_split(item, next_separators))
                    current_chunk = ""
                else:
                    current_chunk = item

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks


class SemanticChunking(BaseChunker):
    """Semantic boundary chunker using sentence distance with fallback to Recursive chunking."""

    def __init__(
        self,
        config: Optional[ChunkerConfig] = None,
        embedding_fn: Optional[Any] = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        super().__init__(config)
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self._fallback_chunker = RecursiveCharacterChunking(config)

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        try:
            # 1. Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                return self._fallback_chunker.chunk(text)

            # 2. Compute embeddings if embedding_fn provided
            if self.embedding_fn is not None:
                embeddings = self.embedding_fn(sentences)
                if len(embeddings) != len(sentences):
                    raise ValueError("Embedding count mismatch")

                # Compute cosine similarity between adjacent sentences
                chunks = []
                current_group = [sentences[0]]
                
                for i in range(1, len(sentences)):
                    sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
                    current_text = " ".join(current_group)
                    
                    if sim < self.similarity_threshold or len(current_text) >= self.config.chunk_size:
                        chunks.append(current_text)
                        current_group = [sentences[i]]
                    else:
                        current_group.append(sentences[i])
                
                if current_group:
                    chunks.append(" ".join(current_group))

                return [
                    {
                        "text": c_text,
                        "index": idx,
                        "strategy": "semantic",
                    }
                    for idx, c_text in enumerate(chunks)
                ]

            # If no embedding_fn is attached, fallback to recursive chunking gracefully
            logger.info("No embedding_fn provided for SemanticChunking. Falling back to RecursiveCharacterChunking.")
            return self._fallback_chunker.chunk(text)

        except Exception as e:
            logger.warning(
                "SemanticChunking failed due to error: %s. Executing error-recovery fallback to RecursiveCharacterChunking.",
                e,
            )
            fallback_res = self._fallback_chunker.chunk(text)
            for item in fallback_res:
                item["strategy"] = "recursive_fallback"
            return fallback_res

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = (sum(a * a for a in vec_a)) ** 0.5
        norm_b = (sum(b * b for b in vec_b)) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def get_chunker(config: Optional[ChunkerConfig] = None) -> BaseChunker:
    """Factory function to build chunker instance from ChunkerConfig."""
    config = config or ChunkerConfig()
    strategy = config.strategy.lower()

    if strategy == "fixed":
        return FixedSizeChunking(config)
    elif strategy == "semantic":
        return SemanticChunking(config)
    else:
        return RecursiveCharacterChunking(config)

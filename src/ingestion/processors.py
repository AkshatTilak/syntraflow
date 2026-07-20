"""Concrete Pre-Processor and Post-Processor Implementations for SyntraFlow.

Includes OCRNoiseReduction, LanguageFilter, MetadataExtractor, and SummaryTagger.
"""

import logging
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional

from projects.syntraflow.src.ingestion.strategies import (
    BasePostProcessor,
    BasePreProcessor,
    PostProcessorConfig,
    PreProcessorConfig,
)

logger = logging.getLogger("syntraflow.processors")


class OCRNoiseReduction(BasePreProcessor):
    """Pre-processor that cleans visual artifacts, repeated symbols, and noise from raw OCR data."""

    def process(self, data: bytes) -> bytes:
        if not data:
            return b""

        text = data.decode("utf-8", errors="replace")

        # 1. Remove repeated noise characters (e.g. "=====", "~~~~~", "*****")
        text = re.sub(r"[=~*_\-]{4,}", " ", text)

        # 2. Collapse 3+ consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. Strip non-printable ASCII characters while preserving newlines & tabs
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (32 <= ord(ch) <= 126) or ord(ch) > 127)

        return text.encode("utf-8")


class LanguageFilter(BasePreProcessor):
    """Pre-processor that normalizes character encodings and strips null bytes."""

    def process(self, data: bytes) -> bytes:
        if not data:
            return b""

        text = data.decode("utf-8", errors="ignore")
        # Unicode normalization (NFKC)
        normalized = unicodedata.normalize("NFKC", text)
        # Remove null bytes
        cleaned = normalized.replace("\x00", "")
        return cleaned.encode("utf-8")


class MetadataExtractor(BasePostProcessor):
    """Post-processor that extracts entities (dates, emails, keywords) from chunk text."""

    def __init__(
        self,
        config: Optional[PostProcessorConfig] = None,
        completion_fn: Optional[Callable[[str], dict]] = None,
    ) -> None:
        super().__init__(config)
        self.completion_fn = completion_fn

    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        if self.completion_fn is not None:
            try:
                llm_meta = self.completion_fn(text)
                metadata["entities"] = llm_meta.get("entities", [])
                metadata["author"] = llm_meta.get("author")
                chunk["metadata"] = metadata
                return chunk
            except Exception as e:
                logger.warning("MetadataExtractor LLM call failed: %s. Falling back to regex extraction.", e)

        # Rule-based fallback entity extraction
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
        title_words = re.findall(r"\b[A-Z][a-z]{3,}\b", text)
        unique_keywords = list(set(title_words[:5]))

        metadata["entities"] = {
            "emails": emails,
            "dates": dates,
            "keywords": unique_keywords,
        }
        chunk["metadata"] = metadata
        return chunk


class SummaryTagger(BasePostProcessor):
    """Post-processor that generates short summary tags for each chunk."""

    def enrich(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})

        # Generate a concise summary preview (first sentence or first 100 chars)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary = sentences[0] if sentences and len(sentences[0]) > 5 else text[:100]

        # Generate tags from longest unique words
        words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
        unique_words = sorted(list(set(words)), key=len, reverse=True)[:4]

        metadata["summary"] = summary
        metadata["tags"] = unique_words
        chunk["metadata"] = metadata
        return chunk


def get_pre_processor(
    name: str, config: Optional[PreProcessorConfig] = None
) -> BasePreProcessor:
    """Factory function for PreProcessor instances."""
    name_lower = name.lower()
    if "ocr" in name_lower or "noise" in name_lower:
        return OCRNoiseReduction(config)
    else:
        return LanguageFilter(config)


def get_post_processor(
    name: str, config: Optional[PostProcessorConfig] = None
) -> BasePostProcessor:
    """Factory function for PostProcessor instances."""
    name_lower = name.lower()
    if "summary" in name_lower or "tag" in name_lower:
        return SummaryTagger(config)
    else:
        return MetadataExtractor(config)

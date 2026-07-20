"""Unit tests for SyntraFlow PreProcessors and PostProcessors."""

import pytest
from projects.syntraflow.src.ingestion.processors import (
    OCRNoiseReduction,
    LanguageFilter,
    MetadataExtractor,
    SummaryTagger,
    get_pre_processor,
    get_post_processor,
)


def test_ocr_noise_reduction():
    """Verify OCRNoiseReduction removes repetitive symbols and extra newlines."""
    processor = OCRNoiseReduction()
    raw_data = b"Hello world! =====~~~~~*****\n\n\n\nThis is a test line."
    cleaned_bytes = processor.process(raw_data)
    cleaned_text = cleaned_bytes.decode("utf-8")

    assert "=====" not in cleaned_text
    assert "\n\n\n\n" not in cleaned_text
    assert "Hello world!" in cleaned_text


def test_language_filter():
    """Verify LanguageFilter removes null bytes and normalizes text."""
    processor = LanguageFilter()
    raw_data = "Clean text \x00 with unicode \u2147".encode("utf-8")
    cleaned_bytes = processor.process(raw_data)
    cleaned_text = cleaned_bytes.decode("utf-8")

    assert "\x00" not in cleaned_text
    assert "Clean text" in cleaned_text


def test_metadata_extractor_rule_based():
    """Verify MetadataExtractor extracts emails, dates, and keywords."""
    extractor = MetadataExtractor()
    chunk = {
        "text": "Contact user@example.com on 2026-07-20 regarding Architecture Refactor.",
        "metadata": {},
    }
    enriched = extractor.enrich(chunk)
    entities = enriched["metadata"]["entities"]

    assert "user@example.com" in entities["emails"]
    assert "2026-07-20" in entities["dates"]
    assert "Architecture" in entities["keywords"] or "Refactor" in entities["keywords"]


def test_metadata_extractor_completion_fn():
    """Verify MetadataExtractor uses completion_fn when provided."""
    def mock_llm(text):
        return {"entities": ["CustomEntity"], "author": "Alice"}

    extractor = MetadataExtractor(completion_fn=mock_llm)
    chunk = {"text": "Some text", "metadata": {}}
    enriched = extractor.enrich(chunk)

    assert enriched["metadata"]["entities"] == ["CustomEntity"]
    assert enriched["metadata"]["author"] == "Alice"


def test_summary_tagger():
    """Verify SummaryTagger attaches summary preview and tags."""
    tagger = SummaryTagger()
    chunk = {"text": "SyntraFlow provides modular RAG ingestion. It handles documents easily.", "metadata": {}}
    enriched = tagger.enrich(chunk)

    assert "summary" in enriched["metadata"]
    assert "tags" in enriched["metadata"]
    assert len(enriched["metadata"]["tags"]) > 0


def test_processor_factories():
    """Verify factory functions return correct pre and post processors."""
    ocr = get_pre_processor("ocr_noise_reduction")
    assert isinstance(ocr, OCRNoiseReduction)

    lang = get_pre_processor("language_filter")
    assert isinstance(lang, LanguageFilter)

    summary = get_post_processor("summary_tagger")
    assert isinstance(summary, SummaryTagger)

    meta = get_post_processor("metadata_extractor")
    assert isinstance(meta, MetadataExtractor)

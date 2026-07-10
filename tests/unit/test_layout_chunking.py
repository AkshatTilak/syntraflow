import pytest
from projects.syntraflow.src.ingestion import chunk_document_layout_aware, count_tokens

def test_count_tokens():
    text = "hello world this is a test"
    assert count_tokens(text) > 0

def test_chunk_document_layout_aware_with_blocks():
    mock_ocr = {
        "blocks": [
            {"type": "header", "content": "# Overview", "bbox": [0,0,10,10]},
            {"type": "paragraph", "content": "This is the first section of the document talking about overview details.", "bbox": [0,10,10,20]},
            {"type": "header", "content": "## Subsection", "bbox": [0,20,10,30]},
            {"type": "paragraph", "content": "This paragraph is in a subsection.", "bbox": [0,30,10,40]},
            {"type": "paragraph", "content": "Another paragraph in the subsection.", "bbox": [0,40,10,50]},
        ]
    }
    
    chunks = chunk_document_layout_aware(mock_ocr, max_tokens=100, overlap=10, min_tokens=5)
    
    assert len(chunks) == 2
    
    # First chunk should contain header and paragraph under # Overview
    assert "Overview" in chunks[0]["text"]
    assert "Subsection" not in chunks[0]["text"]
    assert "Overview" in chunks[0]["metadata"]["hierarchy"]
    
    # Second chunk should contain Subsection header and subsection paragraphs
    assert "Subsection" in chunks[1]["text"]
    assert "Overview > Subsection" in chunks[1]["text"]
    assert "Subsection" in chunks[1]["metadata"]["hierarchy"]


def test_chunk_document_layout_aware_with_markdown_text():
    mock_ocr = {
        "text": "# Overview\nThis is paragraph one.\n## Detail Section\nThis is paragraph two."
    }
    
    chunks = chunk_document_layout_aware(mock_ocr, max_tokens=100, overlap=10, min_tokens=5)
    
    assert len(chunks) == 2
    assert "Overview" in chunks[0]["metadata"]["hierarchy"]
    assert "Detail Section" in chunks[1]["metadata"]["hierarchy"]

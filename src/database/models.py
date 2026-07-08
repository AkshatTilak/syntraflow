"""SQLAlchemy models for SyntraFlow data schemas."""

import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, ForeignKey, Uuid
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class SyntraFlowDocument(Base):
    """Stores document metadata and layout-preserving Markdown content."""

    __tablename__ = "syntraflow_documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    layout_json = Column(Text, nullable=True)  # Stores serialized layout structure
    created_at = Column(DateTime, default=datetime.utcnow)


class SyntraFlowChunk(Base):
    """Stores individual text/markdown chunks, image references, and structural JSON."""

    __tablename__ = "syntraflow_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    image_path = Column(String(512), nullable=True)
    metadata_json = Column(Text, nullable=True)  # Stores chunk layout metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class SyntraFlowVideoSegment(Base):
    """Stores timestamped transcribed video segments, visual descriptions, and audio tags."""

    __tablename__ = "syntraflow_video_segments"

    id = Column(Integer, primary_key=True, index=True)
    video_name = Column(String(255), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    transcript = Column(Text, nullable=False)
    visual_summary = Column(Text, nullable=True)
    emotion_tags = Column(String(255), nullable=True)
    audio_events = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SyntraFlowJob(Base):
    """Stores status tracking details for SyntraFlow ingestion jobs."""

    __tablename__ = "syntraflow_jobs"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id = Column(Integer, ForeignKey("syntraflow_documents.id"), nullable=True)
    status = Column(String(20), nullable=False, default="queued")  # queued, processing, completed, failed
    progress = Column(Float, default=0.0)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

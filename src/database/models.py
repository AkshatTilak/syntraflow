"""SQLAlchemy models for SyntraFlow data schemas.

All models use the shared Base from common.models.database for unified
Alembic migration support across the monorepo.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, JSON, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import relationship

from common.models.database import Base, HubScopedMixin


def build_physical_name(hub_slug: str, name: str) -> str:
    """Build canonical global Qdrant physical collection name from hub_slug and collection name."""
    import re
    slug_clean = re.sub(r"[^a-z0-9_]+", "_", hub_slug.lower())
    name_clean = re.sub(r"[^a-z0-9_]+", "_", name.lower())
    return f"{slug_clean}__{name_clean}"


class SyntraFlowDocument(HubScopedMixin, Base):
    """Stores document metadata and layout-preserving Markdown content.
    Hub-scoped: every query MUST filter by hub_id. Note: hub_id is authoritative for scoping.
    """

    __tablename__ = "syntraflow_documents"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id = Column(String(36), ForeignKey("syntraflow_collections.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)  # SHA-256 for duplicate detection
    content = Column(Text, nullable=False)
    layout_json = Column(Text, nullable=True)  # Stores serialized layout structure
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_syntraflow_documents_hub_col", "hub_id", "collection_id"),
    )

    # Relationships
    chunks = relationship("SyntraFlowChunk", back_populates="document", cascade="all, delete-orphan")
    video_segments = relationship("SyntraFlowVideoSegment", back_populates="document", cascade="all, delete-orphan")
    jobs = relationship("SyntraFlowJob", back_populates="document", cascade="all, delete-orphan")


class SyntraFlowChunk(HubScopedMixin, Base):
    """Stores individual text/markdown chunks, image references, and structural JSON.
    Hub-scoped: every query MUST filter by hub_id. Denormalised from parent document for performance.
    """

    __tablename__ = "syntraflow_chunks"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id = Column(Uuid, ForeignKey("syntraflow_documents.id", ondelete="CASCADE"), nullable=True, index=True)
    chunk_index = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    image_path = Column(String(512), nullable=True)
    metadata_json = Column(Text, nullable=True)  # Stores chunk layout metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("SyntraFlowDocument", back_populates="chunks")


class SyntraFlowVideoSegment(HubScopedMixin, Base):
    """Stores timestamped transcribed video segments, visual descriptions, and audio tags.
    Hub-scoped: every query MUST filter by hub_id. Denormalised from parent document for performance.
    """

    __tablename__ = "syntraflow_video_segments"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id = Column(Uuid, ForeignKey("syntraflow_documents.id", ondelete="CASCADE"), nullable=True, index=True)
    video_name = Column(String(255), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    transcript = Column(Text, nullable=False)
    visual_summary = Column(Text, nullable=True)
    emotion_tags = Column(String(255), nullable=True)
    audio_events = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("SyntraFlowDocument", back_populates="video_segments")


class SyntraFlowJob(HubScopedMixin, Base):
    """Stores status tracking details for SyntraFlow ingestion jobs.
    Hub-scoped: every query MUST filter by hub_id. Denormalised from parent document for performance.
    """

    __tablename__ = "syntraflow_jobs"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id = Column(String(36), ForeignKey("syntraflow_collections.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(Uuid, ForeignKey("syntraflow_documents.id", ondelete="CASCADE"), nullable=True)
    status = Column(String(20), nullable=False, default="queued")  # queued, processing, completed, failed
    progress = Column(Float, default=0.0)
    error_msg = Column(Text, nullable=True)
    pipeline_config_json = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_syntraflow_jobs_hub_status", "hub_id", "status"),
    )

    # Relationships
    document = relationship("SyntraFlowDocument", back_populates="jobs")


class SyntraFlowCollection(HubScopedMixin, Base):
    """Stores metadata for dynamic Qdrant vector collections.
    Hub-scoped: every query MUST filter by hub_id (hubs.md §5.3).
    """

    __tablename__ = "syntraflow_collections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    physical_name = Column(String(320), nullable=False, index=True)  # "{hub_slug}__{name}", globally unique
    embedding_model = Column(String(255), nullable=False, default="jina-clip-v2")
    vector_dimension = Column(Float, nullable=False, default=1024)
    description = Column(Text, nullable=True)
    retrieval_config_json = Column(JSON, nullable=False, default=dict)
    pipeline_config_json = Column(JSON, nullable=True, default=dict)
    datastore_binding_id = Column(String(36), ForeignKey("datastore_bindings.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("hub_id", "name", name="uq_syntraflow_collections_hub_name"),
        UniqueConstraint("physical_name", name="uq_syntraflow_collections_physical_name"),
    )


"""Unit tests for SyntraFlowCollection model CRUD operations."""

import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from common.models.database import Base
from projects.syntraflow.src.database.models import SyntraFlowCollection


def test_syntraflow_collection_crud():
    """Verify SyntraFlowCollection model creation, querying, and deletion in SQLite memory session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()


    collection_id = uuid.uuid4()
    col = SyntraFlowCollection(
        id=collection_id,
        name="test_collection_docs",
        tenant_id="tenant_alpha",
        embedding_model="jina-clip-v2",
        vector_dimension=1024,
        description="Collection for test documents",
    )
    session.add(col)
    session.commit()

    # Query back
    queried = session.query(SyntraFlowCollection).filter_by(name="test_collection_docs").first()
    assert queried is not None
    assert queried.id == collection_id
    assert queried.tenant_id == "tenant_alpha"
    assert queried.embedding_model == "jina-clip-v2"
    assert queried.vector_dimension == 1024
    assert queried.description == "Collection for test documents"

    # Delete
    session.delete(queried)
    session.commit()

    deleted = session.query(SyntraFlowCollection).filter_by(name="test_collection_docs").first()
    assert deleted is None

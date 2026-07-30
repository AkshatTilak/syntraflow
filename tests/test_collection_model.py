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
        id=str(collection_id),
        hub_id="hub_alpha",
        name="test_collection_docs",
        physical_name="hub_alpha__test_collection_docs",
        embedding_model="jina-clip-v2",
        vector_dimension=1024,
        description="Collection for test documents",
    )
    session.add(col)
    session.commit()

    # Query back
    queried = session.query(SyntraFlowCollection).filter_by(name="test_collection_docs").first()
    assert queried is not None
    assert queried.id == str(collection_id)
    assert queried.hub_id == "hub_alpha"
    assert queried.embedding_model == "jina-clip-v2"
    assert queried.vector_dimension == 1024
    assert queried.description == "Collection for test documents"

    # Delete
    session.delete(queried)
    session.commit()

    deleted = session.query(SyntraFlowCollection).filter_by(name="test_collection_docs").first()
    assert deleted is None

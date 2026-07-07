"""Database client wrapper for PostgreSQL using SQLAlchemy."""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from common.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception class for database errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised when database connection fails."""

    pass


try:
    # Initialize engine with robust connection pooling configurations
    engine = create_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except SQLAlchemyError as e:
    logger.error("Failed to initialize database engine: %s", e)
    raise DatabaseConnectionError("Database initialization failed") from e


def get_db() -> Generator[Session, None, None]:
    """Dependency generator to retrieve a database session.

    Yields:
        An active SQLAlchemy Session.

    Raises:
        DatabaseError: If session operations fail or connection is lost.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error("Database session error occurred: %s", e)
        db.rollback()
        raise DatabaseError("Database transaction failed") from e
    finally:
        db.close()

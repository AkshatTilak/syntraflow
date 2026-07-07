"""Vector database client wrapper for Qdrant."""

import logging
from typing import Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from common.config import settings

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception class for vector store errors."""

    pass


class VectorClient:
    """Wrapper class for Qdrant client interactions.

    Encapsulates QdrantClient connections and maps exceptions to VectorStoreError.
    """

    def __init__(self) -> None:
        """Initializes the QdrantClient using settings configuration."""
        try:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e)
            raise VectorStoreError("Vector store client initialization failed") from e

    def get_client(self) -> QdrantClient:
        """Access the underlying QdrantClient directly.

        Returns:
            The raw QdrantClient instance.
        """
        return self._client

    def search_similarity(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Any]:
        """Perform similarity search on Qdrant.

        Args:
            collection_name: The name of the target vector collection.
            query_vector: The query embedding vector.
            limit: Maximum number of records to return.

        Returns:
            A list of search results.

        Raises:
            VectorStoreError: If query execution or API response fails.
        """
        try:
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
            )
            return results
        except (UnexpectedResponse, ValueError) as e:
            logger.error("Similarity search failed in Qdrant: %s", e)
            raise VectorStoreError("Query failed inside Vector DB") from e

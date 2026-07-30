"""Model Context Protocol (MCP) server for SyntraFlow retrieval tools."""

import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy import text

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.qdrant import VectorClient
from projects.syntraflow.src.retrieval import RetrievalEngine

# Initialize FastMCP Server
mcp = FastMCP("SyntraFlow")
logger = logging.getLogger("syntraflow.mcp_server")


def get_inference_client() -> InferenceClient:
    return InferenceClient(base_url=settings.INFERENCE_SERVER_URL)


def get_vector_client() -> VectorClient:
    return VectorClient()


@mcp.tool()
async def retrieve_documents(hub_id: str, query: str, strategy: str = "hybrid", limit: int = 5, collection_ids: Optional[List[str]] = None) -> str:
    """Retrieve document contents from vector database and/or graph database for a given hub_id.

    Args:
        hub_id: ID of the target hub.
        query: Search query text.
        strategy: Retrieval strategy - 'dense', 'sparse', 'hybrid', or 'graph'.
        limit: Max results to return.
        collection_ids: Optional list of collection IDs within the hub.
    """
    logger.info("MCP Tool [retrieve_documents]: hub_id=%s, query=%s, strategy=%s", hub_id, query, strategy)
    from common.clients.postgres import get_sessionmaker
    SessionLocal = get_sessionmaker()

    async with SessionLocal() as db:
        engine = RetrievalEngine(db, hub_id)
        hits = await engine.search(
            query=query,
            collection_ids=collection_ids,
            strategy=strategy,
            limit=limit,
        )
        return json.dumps(hits, indent=2)


@mcp.tool()
async def retrieve_video_segments(hub_id: str, query: str, limit: int = 5, collection_ids: Optional[List[str]] = None) -> str:
    """Retrieve timestamped segments of video transcripts and aligned visual summaries for a given hub_id.

    Args:
        hub_id: ID of the target hub.
        query: Search query text.
        limit: Max segments to return.
        collection_ids: Optional list of collection IDs within the hub.
    """
    logger.info("MCP Tool [retrieve_video_segments]: hub_id=%s, query=%s", hub_id, query)
    from common.clients.postgres import get_sessionmaker
    SessionLocal = get_sessionmaker()

    async with SessionLocal() as db:
        engine = RetrievalEngine(db, hub_id)
        hits = await engine.search(
            query=query,
            collection_ids=collection_ids,
            strategy="dense",
            limit=limit,
        )
        return json.dumps(hits, indent=2)


@mcp.tool()
async def query_database(hub_id: str, table: str, filters: Dict[str, Any], columns: List[str]) -> str:
    """Query local PostgreSQL relational tables using parameterized criteria bounded by hub_id.

    Args:
        hub_id: ID of the target hub.
        table: Target table name. MUST begin with 'syntraflow_'.
        filters: Dictionary of column names and values to filter by.
        columns: List of column names to retrieve.
    """
    logger.info("MCP Tool [query_database]: hub_id=%s, table=%s, filters=%s", hub_id, table, filters)
    
    # 1. Table name sanitization & isolation boundary check
    if not table.startswith("syntraflow_"):
        return json.dumps({"error": "Unauthorized: Access limited to syntraflow_ prefix tables."})

    # Enforce hub_id in filters
    filters["hub_id"] = hub_id

    # Basic alphanumeric checks for columns and table to prevent injection
    safe_columns = [col for col in columns if col.replace("_", "").isalnum()]
    if not safe_columns:
        safe_columns = ["*"]
        
    if not table.replace("_", "").isalnum():
        return json.dumps({"error": "Invalid table name format."})

    # 2. Build parameterized query
    select_clause = ", ".join(safe_columns)
    where_clauses = []
    params = {}
    
    for idx, (col, val) in enumerate(filters.items()):
        if col.replace("_", "").isalnum():
            param_name = f"val_{idx}"
            where_clauses.append(f"{col} = :{param_name}")
            params[param_name] = val
            
    sql_str = f"SELECT {select_clause} FROM {table}"
    if where_clauses:
        sql_str += " WHERE " + " AND ".join(where_clauses)

    try:
        from common.clients.postgres import get_sessionmaker
        SessionLocal = get_sessionmaker()
        async with SessionLocal() as db:
            result = await db.execute(text(sql_str), params)
            rows = [dict(r._mapping) for r in result]
            return json.dumps(rows, default=str)
    except Exception as e:
        logger.error("DB Query error: %s", e)
        return json.dumps({"error": f"Database query failed: {str(e)}"})


@mcp.tool()
async def query_graph(hub_id: str, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """Execute Cypher query against Neo4j in a safe, read-only parameterized way for a target hub_id.

    Args:
        hub_id: ID of the target hub.
        cypher_query: Cypher statement. Must start with MATCH or MATCH/RETURN only.
        parameters: Optional dictionary of parameters for the query.
    """
    logger.info("MCP Tool [query_graph]: hub_id=%s, query=%s, parameters=%s", hub_id, cypher_query, parameters)
    
    # Simple query check to prevent write operations (CREATE, MERGE, DELETE, SET)
    query_upper = cypher_query.upper()
    forbidden_keywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]
    for kw in forbidden_keywords:
        if kw in query_upper:
            return json.dumps({"error": f"Unauthorized statement: write operations like {kw} are blocked."})

    # Verify prefix boundaries in query
    if "SYNTRAFLOW_" not in query_upper:
        return json.dumps({"error": "Unauthorized: Cypher queries must only query SyntraFlow_ prefixes."})

    params = parameters.copy() if parameters else {}
    params["hub_id"] = hub_id

    try:
        from projects.syntraflow.src.datastores import resolve_graph_client
        from common.clients.postgres import get_sessionmaker
        SessionLocal = get_sessionmaker()
        async with SessionLocal() as db:
            driver = await resolve_graph_client(db, hub_id)
            async with driver.session() as session:
                res = await session.run(cypher_query, params)
                records = await res.data()
            return json.dumps(records, default=str)
    except Exception as e:
        logger.error("Graph Query error: %s", e)
        return json.dumps({"error": f"Graph query failed: {str(e)}"})



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        import uvicorn
        host = "0.0.0.0"
        port = 8012
        logger.info("Starting FastMCP server over SSE at http://%s:%d", host, port)
        app = mcp.sse_app()
        uvicorn.run(app, host=host, port=port)
    else:
        mcp.run()

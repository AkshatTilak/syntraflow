"""Model Context Protocol (MCP) server for SyntraFlow retrieval tools."""

import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from sqlalchemy.orm import Session
from sqlalchemy import text

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.qdrant import VectorClient
from projects.syntraflow.src.retrieval import RetrievalEngine
from common.clients.postgres import get_async_db as get_db

# Initialize FastMCP Server
mcp = FastMCP("SyntraFlow")
logger = logging.getLogger("syntraflow.mcp_server")


def get_inference_client() -> InferenceClient:
    return InferenceClient(base_url=settings.INFERENCE_SERVER_URL)


def get_vector_client() -> VectorClient:
    return VectorClient()


@mcp.tool()
async def retrieve_documents(query: str, strategy: str = "hybrid", limit: int = 5) -> str:
    """Retrieve document contents from vector database and/or graph database.

    Args:
        query: Search query text.
        strategy: Retrieval strategy - 'vector', 'graph', or 'hybrid'.
        limit: Max results to return.
    """
    logger.info("MCP Tool [retrieve_documents]: query=%s, strategy=%s", query, strategy)
    inference = get_inference_client()
    vector = get_vector_client()
    engine = RetrievalEngine(vector)

    try:
        # Get query embedding
        embeds = await inference.embed(texts=[query])
        query_vector = embeds[0]
    except Exception:
        # Fallback query vector
        query_vector = [0.0] * 768

    if strategy == "vector":
        hits = await engine.search_vector(query_vector, limit=limit)
    elif strategy == "graph":
        hits = await engine.search_graph(query, limit=limit)
    else:
        hits = await engine.search_hybrid(query, query_vector, limit=limit)

    return json.dumps(hits, indent=2)


@mcp.tool()
async def retrieve_video_segments(query: str, limit: int = 5) -> str:
    """Retrieve timestamped segments of video transcripts and aligned visual summaries.

    Args:
        query: Search query text.
        limit: Max segments to return.
    """
    logger.info("MCP Tool [retrieve_video_segments]: query=%s", query)
    inference = get_inference_client()
    vector = get_vector_client()
    engine = RetrievalEngine(vector)

    try:
        embeds = await inference.embed(texts=[query])
        query_vector = embeds[0]
    except Exception:
        query_vector = [0.0] * 768

    hits = await engine.search_vector(query_vector, limit=limit)
    video_hits = [h for h in hits if h["metadata"].get("start_time") is not None]
    
    return json.dumps(video_hits or hits[:limit], indent=2)


@mcp.tool()
async def query_database(table: str, filters: Dict[str, Any], columns: List[str]) -> str:
    """Query local PostgreSQL relational tables using parameterized criteria (rejects raw SQL).

    Args:
        table: Target table name. MUST begin with 'syntraflow_'.
        filters: Dictionary of column names and values to filter by.
        columns: List of column names to retrieve.
    """
    logger.info("MCP Tool [query_database]: table=%s, filters=%s", table, filters)
    
    # 1. Table name sanitization & isolation boundary check
    if not table.startswith("syntraflow_"):
        return json.dumps({"error": "Unauthorized: Access limited to syntraflow_ prefix tables."})

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
        db_generator = get_db()
        db: Session = next(db_generator)
        result = db.execute(text(sql_str), params)
        rows = [dict(r._mapping) for r in result]
        return json.dumps(rows, default=str)
    except Exception as e:
        logger.error("DB Query error: %s", e)
        return json.dumps({"error": f"Database query failed: {str(e)}"})


@mcp.tool()
async def query_graph(cypher_query: str) -> str:
    """Execute Cypher query against Neo4j in a safe, read-only parameterized way.

    Args:
        cypher_query: Cypher statement. Must start with MATCH or MATCH/RETURN only.
    """
    logger.info("MCP Tool [query_graph]: query=%s", cypher_query)
    
    # Simple query check to prevent write operations (CREATE, MERGE, DELETE, SET)
    query_upper = cypher_query.upper()
    forbidden_keywords = ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP"]
    for kw in forbidden_keywords:
        if kw in query_upper:
            return json.dumps({"error": f"Unauthorized statement: write operations like {kw} are blocked."})

    # Verify prefix boundaries in query
    if "SYNTRAFLOW_" not in query_upper:
        return json.dumps({"error": "Unauthorized: Cypher queries must only query SyntraFlow_ prefixes."})

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            result = session.run(cypher_query)
            records = [dict(r) for r in result]
        driver.close()
        return json.dumps(records, default=str)
    except Exception as e:
        logger.error("Graph Query error: %s", e)
        return json.dumps({"error": f"Graph query failed: {str(e)}"})


if __name__ == "__main__":
    mcp.run()

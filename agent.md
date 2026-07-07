# SyntraFlow Developer Agent Guidelines

This document details coding standards and requirements specific to the **SyntraFlow** submodule within the monorepo architecture. For general platform standards, refer to the [Root Monorepo Guidelines](../../agent.md).

---

## 1. Submodule Boundary & Interfaces

SyntraFlow is a plug-and-play submodule designed to run inside the central API gateway. It exposes its capabilities via two entrypoints:
1. **API Router (`api.py`)**: Mounts FastAPI endpoints (e.g., `/api/syntraflow/ingest`, `/api/syntraflow/search`) into the gateway.
2. **Setup Hook (`setup.py`)**: Runs on gateway startup to register the `InferenceClient` and start/stop the MCP retrieval server.

---

## 2. Ingestion & Retrieval Model Stack

SyntraFlow relies on local GPU models served by the `inference` process. Ingestion pipelines must call these models asynchronously:

| Action | Local GPU Target | API Target (Required/Fallback) |
|---|---|---|
| **Document OCR & Layout** | Baidu Unlimited-OCR (when `OCR_PROVIDER=local`) | Gemini Flash API (when `OCR_PROVIDER=api` or local fallback) |
| **JSON Schema Mapping** | N/A (CPU-only) | Gemini Flash API |
| **ASR Transcription** | SenseVoice-Small / Moonshine | Gemini Flash API (audio mode) |
| **Multimodal Embeddings** | jina-clip-v2 | gemini-embedding-2 API |

### Implementation Rules
- **No Direct Model Loads**: SyntraFlow source code *must never* call `from transformers import AutoModelForCausalLM` or load torch weights into the gateway's memory. All calls go through `InferenceClient` (HTTP).
- **Customizable OCR**: Read `settings.OCR_PROVIDER` to choose the layout extraction route. If `local`, perform HTTP call to inference server `/infer/ocr` (Baidu). If `api`, run Gemini Flash layout extraction.
- **Hybrid RAG Strategy**: Implement standard retrieval methods for Vector RAG (Qdrant), GraphRAG (Neo4j), and Hybrid RAG (combines both using Reciprocal Rank Fusion).
- **Concurrency**: Replace any legacy PySpark dependencies with Python `asyncio` and `concurrent.futures` for file preprocessing.

---

## 3. Database Schema Isolation

To prevent conflicts with other monorepo submodules sharing the same database:
1. **Table Namespacing**: Prefix all PostgreSQL tables with `syntraflow_` (e.g., `syntraflow_documents`, `syntraflow_video_segments`).
2. **Qdrant Collections**: Prefix vector collections with `syntraflow_` (e.g., `syntraflow_chunks_v1`).
3. **Neo4j Labels**: Prefix all graph labels and relation types with `SyntraFlow_` (e.g., `SyntraFlow_Entity`, `SyntraFlow_RELATION`) to isolate Neo4j workspace.
4. **Safe MCP Queries**: The `query_database` and `query_graph` MCP tools must reject raw query strings. Implement parameterized Cypher/SQL parameters or structured dictionaries to protect against injection.

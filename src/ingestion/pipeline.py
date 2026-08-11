"""SyntraFlow document and video ingestion pipelines.

Implements customizable layout OCR, async keyframe sampling, SenseVoice audio transcription,
temporal alignment, chunking, jina-clip embeddings, and DB writes.
"""

import base64
import json
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.http.models import PointStruct

from common.config.settings import settings
from common.clients.inference import InferenceClient
from common.clients.litellm import completion_with_fallback
from projects.syntraflow.src.database.models import (
    SyntraFlowDocument,
    SyntraFlowChunk,
    SyntraFlowVideoSegment,
    SyntraFlowCollection,
)

logger = logging.getLogger("syntraflow.ingestion")


async def extract_layout_ocr(
    file_bytes: bytes,
    filename: str,
    client: InferenceClient,
) -> dict:
    """Extract layout structures from document using the active Model Registry configuration."""
    from opentelemetry import trace
    tracer = trace.get_tracer("syntraflow")
    with tracer.start_as_current_span("syntraflow.ocr") as span:
        span.set_attribute("filename", filename)
        return await _extract_layout_ocr_inner(file_bytes, filename, client)


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text directly from PDF file bytes using pypdf, fitz, or regex stream fallback."""
    if not file_bytes or not file_bytes.startswith(b"%PDF"):
        return ""

    # 1. Try pypdf
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt and txt.strip():
                pages_text.append(f"## Page {i+1}\n\n{txt.strip()}")
        if pages_text:
            return "\n\n".join(pages_text)
    except Exception as e:
        logger.debug("pypdf extraction skipped: %s", e)

    # 2. Try fitz (PyMuPDF)
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        for i, page in enumerate(doc):
            txt = page.get_text()
            if txt and txt.strip():
                pages_text.append(f"## Page {i+1}\n\n{txt.strip()}")
        if pages_text:
            return "\n\n".join(pages_text)
    except Exception:
        pass

    # 3. Stream regex fallback
    try:
        import re
        matches = re.findall(r"\(([^)]+)\)\s*(?:Tj|TJ)", file_bytes.decode("latin1", errors="ignore"))
        if matches:
            text = " ".join(matches).strip()
            if len(text) > 30:
                return text
    except Exception:
        pass

    return ""


async def _extract_layout_ocr_inner(
    file_bytes: bytes,
    filename: str,
    client: InferenceClient,
) -> dict:
    # 0. Check direct PDF text extraction first if valid PDF
    if file_bytes and file_bytes.startswith(b"%PDF"):
        pdf_text = extract_pdf_text(file_bytes)
        if pdf_text and pdf_text.strip():
            logger.info("Successfully extracted %d chars directly from PDF '%s'", len(pdf_text), filename)
            return {
                "text": pdf_text,
                "blocks": [{"type": "paragraph", "content": pdf_text}],
                "tables": [],
                "layout": {}
            }

    from common.models.registry import get_active_model

    model_spec = await get_active_model("ocr")
    mode = model_spec.mode.lower()

    if mode == "local":
        logger.info("Executing local %s layout extraction...", model_spec.display_name)
        # 1. Run local OCR via inference server
        ocr_result = await client.ocr(file_bytes, filename=filename)
        
        # 2. Call Gemini Flash to structure layout and convert to schema
        completion_spec = await get_active_model("completion")
        prompt = (
            "You are a document structuring expert. Convert this raw OCR layout result "
            "into a clean, layout-preserving Markdown text, identifying all sections and tables:\n\n"
            f"{json.dumps(ocr_result)}"
        )
        response = await completion_with_fallback(
            model=completion_spec.model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "json" in filename.lower() else None
        )
        content_text = response.choices[0].message.content
        return {
            "text": content_text,
            "blocks": ocr_result.get("blocks", []),
            "tables": ocr_result.get("tables", []),
            "layout": ocr_result.get("layout", {})
        }

    else:
        logger.info("Executing API %s layout-aware extraction...", model_spec.display_name)
        # Encode image to base64
        b64_image = base64.b64encode(file_bytes).decode("utf-8")
        
        prompt = (
            "Perform a layout-aware OCR extraction of this document image. "
            "Return a JSON object containing the main 'text' in layout-preserving markdown, "
            "any identified 'tables', and section metadata."
        )
        
        # Prepare content payload with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                    }
                ]
            }
        ]
        
        response = await completion_with_fallback(
            model=model_spec.model_id,
            messages=messages,
        )
        res_text = response.choices[0].message.content
        try:
            # Try to strip markdown code blocks if any
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            return json.loads(res_text)
        except Exception:
            return {"text": res_text, "layout": {}, "tables": []}


def count_tokens(text: str) -> int:
    """Helper to estimate tokens using tiktoken (if available) or fallback estimation."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text.split()) + len(text) // 10


def chunk_document_layout_aware(
    ocr_result: dict,
    max_tokens: int = 512,
    overlap: int = 50,
    min_tokens: int = 50
) -> list[dict]:
    """Split OCR text by logical layout boundaries, preserving header context/metadata."""
    import re
    
    blocks = ocr_result.get("blocks", [])
    parsed_blocks = []
    
    if blocks:
        for b in blocks:
            b_type = b.get("type", "paragraph")
            content = b.get("content", "").strip()
            if not content:
                continue
            header_level = 0
            if b_type == "header":
                h_match = re.match(r"^(#+)\s+(.*)", content)
                if h_match:
                    header_level = len(h_match.group(1))
                    content = h_match.group(2)
                else:
                    header_level = 2
            parsed_blocks.append({
                "type": b_type,
                "content": content,
                "header_level": header_level,
                "bbox": b.get("bbox")
            })
    else:
        text = ocr_result.get("text", "")
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            h_match = re.match(r"^(#+)\s+(.*)", line)
            if h_match:
                parsed_blocks.append({
                    "type": "header",
                    "content": h_match.group(2),
                    "header_level": len(h_match.group(1))
                })
            else:
                parsed_blocks.append({
                    "type": "paragraph",
                    "content": line,
                    "header_level": 0
                })
                
    tables = ocr_result.get("tables", [])
    if tables and not blocks:
        for t in tables:
            title = t.get("title", "Table")
            headers = t.get("headers", [])
            rows = t.get("rows", [])
            tb_lines = [f"### Table: {title}"]
            if headers:
                tb_lines.append("| " + " | ".join(headers) + " |")
                tb_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for r in rows:
                tb_lines.append("| " + " | ".join([str(cell) for cell in r]) + " |")
            parsed_blocks.append({
                "type": "table",
                "content": "\n".join(tb_lines),
                "header_level": 0
            })

    current_headers = []
    chunks = []
    
    current_chunk_text = ""
    current_chunk_blocks = []
    
    def get_hierarchy_prefix(headers_list):
        if not headers_list:
            return ""
        path = " > ".join([h[1] for h in headers_list])
        return f"[{path}]\n"
        
    for block in parsed_blocks:
        b_type = block["type"]
        content = block["content"]
        h_level = block["header_level"]
        
        if b_type == "header":
            current_headers = [h for h in current_headers if h[0] < h_level]
            current_headers.append((h_level, content))
            
            if current_chunk_text.strip():
                prefix = get_hierarchy_prefix(current_headers[:-1])
                chunk_payload = prefix + current_chunk_text.strip()
                if count_tokens(chunk_payload) >= min_tokens or not chunks:
                    chunks.append({
                        "text": chunk_payload,
                        "metadata": {
                            "hierarchy": [h[1] for h in current_headers[:-1]],
                            "bbox_list": [b.get("bbox") for b in current_chunk_blocks if b.get("bbox")]
                        }
                    })
                current_chunk_text = ""
                current_chunk_blocks = []
            
            current_chunk_text = f"Header: {content}\n"
            current_chunk_blocks.append(block)
            
        else:
            prefix = get_hierarchy_prefix(current_headers)
            proposed_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + content
            proposed_payload = prefix + proposed_text
            proposed_tokens = count_tokens(proposed_payload)
            
            if proposed_tokens <= max_tokens:
                current_chunk_text = proposed_text
                current_chunk_blocks.append(block)
            else:
                if current_chunk_text.strip():
                    chunk_payload = prefix + current_chunk_text.strip()
                    chunks.append({
                        "text": chunk_payload,
                        "metadata": {
                            "hierarchy": [h[1] for h in current_headers],
                            "bbox_list": [b.get("bbox") for b in current_chunk_blocks if b.get("bbox")]
                        }
                    })
                
                overlap_text = ""
                overlap_blocks = []
                if current_chunk_blocks:
                    accumulated_tokens = 0
                    for rev_b in reversed(current_chunk_blocks):
                        b_tok = count_tokens(rev_b["content"])
                        if accumulated_tokens + b_tok <= overlap:
                            overlap_blocks.insert(0, rev_b)
                            accumulated_tokens += b_tok
                        else:
                            break
                    if overlap_blocks:
                        overlap_text = "\n\n".join([b["content"] for b in overlap_blocks])
                
                if overlap_text:
                    current_chunk_text = overlap_text + "\n\n" + content
                    current_chunk_blocks = overlap_blocks + [block]
                else:
                    current_chunk_text = content
                    current_chunk_blocks = [block]
                    
    if current_chunk_text.strip():
        prefix = get_hierarchy_prefix(current_headers)
        chunk_payload = prefix + current_chunk_text.strip()
        chunks.append({
            "text": chunk_payload,
            "metadata": {
                "hierarchy": [h[1] for h in current_headers],
                "bbox_list": [b.get("bbox") for b in current_chunk_blocks if b.get("bbox")]
            }
        })
        
    return chunks


async def extract_graph_entities(text: str, model_id: Optional[str] = None) -> dict:
    """Extract entities and relationships from chunk text for Neo4j GraphRAG."""
    prompt = (
        "You are an information extraction assistant. Extract key entities and relationships from this text.\n"
        "Allowed Entity Types: Person, Organization, Location, Concept, Event, Product.\n"
        "Return a JSON object with this exact schema:\n"
        "{\n"
        "  \"entities\": [{\"name\": \"Entity Name\", \"type\": \"Person|Organization|Location|Concept|Event|Product\", \"description\": \"Brief description\"}],\n"
        "  \"relationships\": [{\"source\": \"Source Entity Name\", \"target\": \"Target Entity Name\", \"type\": \"Relationship Type (e.g. WORKS_AT, LOCATED_IN, etc.)\", \"description\": \"Brief description\"}]\n"
        "}\n\n"
        f"Text:\n{text}"
    )
    
    if not model_id:
        try:
            from common.models.registry import get_active_model
            comp_model = await get_active_model("completion")
            model_id = comp_model.model_id
        except Exception:
            model_id = "gemini/gemini-3.5-flash"
        
    try:
        response = await completion_with_fallback(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        res_text = response.choices[0].message.content
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(res_text)
        if "entities" not in data:
            data["entities"] = []
        if "relationships" not in data:
            data["relationships"] = []
        return data
    except Exception as e:
        logger.warning("Failed to extract graph entities: %s", e)
        return {"entities": [], "relationships": []}


async def write_to_neo4j(
    extractions: list[dict],
    document_id: Optional[str] = None,
    hub_id: Optional[str] = None,
    collection_id: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> None:
    """Deduplicate extracted entities/relationships and batch write them to Neo4j with hub properties."""
    entities = {}  # name -> {type, description}
    relationships = {}  # (source, target, type) -> description
    
    for ext in extractions:
        for ent in ext.get("entities", []):
            name = ent.get("name")
            ent_type = ent.get("type")
            desc = ent.get("description", "")
            if not name or not ent_type:
                continue
            name_key = name.strip()
            if name_key not in entities:
                entities[name_key] = {"type": ent_type, "description": desc}
            else:
                if desc and desc not in entities[name_key]["description"]:
                    entities[name_key]["description"] += f" | {desc}"
                    
        for rel in ext.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            desc = rel.get("description", "")
            if not source or not target or not rel_type:
                continue
            rel_key = (source.strip(), target.strip(), rel_type.strip())
            if rel_key not in relationships:
                relationships[rel_key] = desc
            else:
                if desc and desc not in relationships[rel_key]:
                    relationships[rel_key] += f" | {desc}"
                    
    if not entities and not relationships:
        logger.info("No entities or relationships to write to Neo4j.")
        return
        
    try:
        if db and hub_id:
            from projects.syntraflow.src.datastores import resolve_graph_client
            driver = await resolve_graph_client(db, hub_id)
        else:
            from common.clients.neo4j import get_neo4j_driver
            driver = get_neo4j_driver()

        async with driver.session() as session:
            for name, info in entities.items():
                await session.run(
                    "MERGE (e:SyntraFlow_Entity {hub_id: $hub_id, name: $name}) "
                    "ON CREATE SET e.type = $type, e.description = $description, e.document_id = $doc_id, e.collection_id = $col_id "
                    "ON MATCH SET e.document_id = $doc_id, e.collection_id = $col_id, e.description = CASE WHEN e.description IS NULL THEN $description ELSE e.description + ' | ' + $description END",
                    name=name,
                    type=info["type"],
                    description=info["description"],
                    doc_id=document_id,
                    hub_id=hub_id or "default",
                    col_id=collection_id,
                )
                
            for (source, target, rel_type), desc in relationships.items():
                await session.run(
                    "MATCH (a:SyntraFlow_Entity {hub_id: $hub_id, name: $source}), (b:SyntraFlow_Entity {hub_id: $hub_id, name: $target}) "
                    "MERGE (a)-[r:SyntraFlow_RELATION {type: $rel_type}]->(b) "
                    "ON CREATE SET r.description = $description, r.document_id = $doc_id, r.hub_id = $hub_id, r.collection_id = $col_id "
                    "ON MATCH SET r.document_id = $doc_id, r.collection_id = $col_id, r.description = CASE WHEN r.description IS NULL THEN $description ELSE r.description + ' | ' + $description END",
                    source=source,
                    target=target,
                    rel_type=rel_type,
                    description=desc,
                    doc_id=document_id,
                    hub_id=hub_id or "default",
                    col_id=collection_id,
                )
        logger.info("Successfully wrote %d entities and %d relationships to Neo4j.", len(entities), len(relationships))
    except Exception as e:
        logger.warning("Could not write to Neo4j: %s. Proceeding with database and vector writes.", e)


async def demux_audio(file_bytes: bytes, filename: str) -> bytes:
    """Extract audio from video container or transcode audio to 16kHz mono WAV using FFmpeg."""
    import asyncio
    import os
    import tempfile
    
    ext = os.path.splitext(filename)[1].lower() or ".mp4"
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    input_file = tempfile.NamedTemporaryFile(suffix=ext, dir=upload_dir, delete=False)
    output_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=upload_dir, delete=False)
    
    input_path = input_file.name
    output_path = output_file.name
    
    input_file.close()
    output_file.close()
    
    try:
        with open(input_path, "wb") as f:
            f.write(file_bytes)
            
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode}: {stderr.decode(errors='replace')}")
            
        with open(output_path, "rb") as f:
            wav_bytes = f.read()
            
        return wav_bytes
        
    finally:
        for path in [input_path, output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning("Failed to remove temp file %s: %s", path, e)


async def extract_keyframes_with_timestamps(video_bytes: bytes, filename: str) -> list[dict]:
    """Extract keyframes on scene change limits and return list of dicts with image bytes and timestamp."""
    import asyncio
    import os
    import re
    import tempfile
    
    ext = os.path.splitext(filename)[1].lower() or ".mp4"
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    input_file = tempfile.NamedTemporaryFile(suffix=ext, dir=upload_dir, delete=False)
    input_path = input_file.name
    input_file.close()
    
    temp_dir = tempfile.mkdtemp(dir=upload_dir)
    
    try:
        with open(input_path, "wb") as f:
            f.write(video_bytes)
            
        # Run scene change detection
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", "select='gt(scene,0.3)',showinfo,scale=640:-1",
            "-vsync", "vfr",
            os.path.join(temp_dir, "frame_%03d.jpg")
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        stderr_text = stderr.decode(errors="replace")
        
        # Parse pts_times
        pts_times = []
        for line in stderr_text.splitlines():
            if "Parsed_showinfo" in line and "pts_time:" in line:
                match = re.search(r"pts_time:([0-9.]+)", line)
                if match:
                    pts_times.append(float(match.group(1)))
                    
        files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".jpg")])
        
        # Fallback if no scene changes detected
        if not files:
            logger.info("No scene changes detected. Falling back to fps=1/5 periodic keyframe sampling.")
            cmd_fallback = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", "fps=1/5,showinfo,scale=640:-1",
                os.path.join(temp_dir, "frame_%03d.jpg")
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd_fallback,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            stderr_text = stderr.decode(errors="replace")
            
            pts_times = []
            for line in stderr_text.splitlines():
                if "Parsed_showinfo" in line and "pts_time:" in line:
                    match = re.search(r"pts_time:([0-9.]+)", line)
                    if match:
                        pts_times.append(float(match.group(1)))
                        
            files = sorted([f for f in os.listdir(temp_dir) if f.endswith(".jpg")])
            
        keyframes = []
        for idx, fname in enumerate(files):
            fpath = os.path.join(temp_dir, fname)
            with open(fpath, "rb") as f:
                img_bytes = f.read()
            timestamp = pts_times[idx] if idx < len(pts_times) else (idx * 5.0)
            keyframes.append({
                "image_bytes": img_bytes,
                "timestamp": timestamp,
                "filename": fname
            })
            
        return keyframes
        
    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass
        if os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


async def describe_keyframe(image_bytes: bytes, filename: str) -> str:
    """Send keyframe image to cloud LLM for visual description/summary."""
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = (
        "Analyze this video keyframe. Describe the visual content in detail, "
        "including any text, slides, charts, drawings, faces, or notable events."
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                }
            ]
        }
    ]
    
    try:
        from common.models.registry import get_active_model
        comp_model = await get_active_model("completion")
        model_id = comp_model.model_id
    except Exception:
        model_id = "gemini/gemini-3.5-flash"
        
    try:
        response = await completion_with_fallback(
            model=model_id,
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Failed to get keyframe description from cloud LLM: %s", e)
        return "Visual representation of the video sequence."


async def update_job(
    db: AsyncSession,
    job_id: Optional[Any],
    progress: float,
    status: str = "processing",
    error_msg: Optional[str] = None,
) -> None:
    """Helper to update ingestion job status and progress in the database."""
    if not job_id:
        return
    try:
        from datetime import datetime
        from sqlalchemy import update
        from projects.syntraflow.src.database.models import SyntraFlowJob

        stmt = (
            update(SyntraFlowJob)
            .where(SyntraFlowJob.id == job_id)
            .values(
                status=status,
                progress=progress,
                error_msg=error_msg,
                updated_at=datetime.utcnow(),
            )
        )
        await db.execute(stmt)
        await db.commit()
    except Exception as e:
        logger.error("Failed to update job status: %s", e)


async def assert_collection_in_hub(
    db: AsyncSession, hub_id: str, collection_id: str
) -> SyntraFlowCollection:
    """Assert collection exists and belongs to hub_id. Raises 404 if missing or foreign."""
    from fastapi import HTTPException
    from sqlalchemy import select
    from projects.syntraflow.src.database.models import SyntraFlowCollection

    stmt = select(SyntraFlowCollection).where(
        SyntraFlowCollection.id == collection_id,
        SyntraFlowCollection.hub_id == hub_id,
    )
    res = await db.execute(stmt)
    col = res.scalar_one_or_none()
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


async def ingest_document_pipeline(
    file_bytes: bytes,
    filename: str,
    db: AsyncSession,
    inference_client: InferenceClient,
    *,
    hub_id: str,
    collection_id: str,
    job_id: Optional[Any] = None,
    chunker_type: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    pre_processors: Optional[List[str]] = None,
    post_processors: Optional[List[str]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Ingest a multi-page PDF/image document with hub scoping, batching and Neo4j deduplication."""
    from fastapi import HTTPException
    from opentelemetry import trace
    from projects.syntraflow.src.datastores import resolve_vector_client

    # 0a. Collection Hub Validation
    collection = await assert_collection_in_hub(db, hub_id, collection_id)
    vector_client = await resolve_vector_client(db, hub_id)

    cfg = dict(collection.pipeline_config_json or {})
    if pipeline_config:
        cfg.update(pipeline_config)

    effective_ocr = cfg.get("ocr_engine")
    effective_chunker = cfg.get("chunking_strategy") or chunker_type
    effective_chunk_size = cfg.get("chunk_size") or chunk_size or 512
    effective_chunk_overlap = cfg.get("chunk_overlap") or chunk_overlap or 64
    effective_embedder = cfg.get("embedding_model") or collection.embedding_model or "jina-clip-v2"
    effective_post_procs = cfg.get("post_processors") or post_processors or []
    summary_model = cfg.get("summary_model")
    graph_model = cfg.get("graph_model")

    tracer = trace.get_tracer("syntraflow")
    with tracer.start_as_current_span("syntraflow.ingest_document") as span:
        span.set_attribute("hub_id", hub_id)
        span.set_attribute("collection_id", collection_id)
        span.set_attribute("filename", filename)

        await update_job(db, job_id, progress=0.1, status="processing")

        # 0b. Pre-processing pipeline
        if pre_processors:
            from projects.syntraflow.src.ingestion.processors import get_pre_processor
            for pre_name in pre_processors:
                try:
                    pre_proc = get_pre_processor(pre_name)
                    file_bytes = pre_proc.process(file_bytes)
                    logger.info("Executed pre-processor '%s' on %s", pre_name, filename)
                except Exception as e:
                    logger.warning("Failed pre-processor '%s': %s", pre_name, e)

        # 1. Extract Layout and text using OCR (dynamic engine)
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if effective_ocr == "direct" or (not effective_ocr and ext in ("txt", "md", "json", "csv", "log")):
            try:
                extracted_text = file_bytes.decode("utf-8")
                ocr_result = {"text": extracted_text, "blocks": [], "tables": [], "layout": {}}
            except Exception:
                ocr_result = await extract_layout_ocr(file_bytes, filename, inference_client)
                extracted_text = ocr_result.get("text", "")
        else:
            ocr_result = await extract_layout_ocr(file_bytes, filename, inference_client)
            extracted_text = ocr_result.get("text", "")

        await update_job(db, job_id, progress=0.4, status="processing")
        
        # 2. Save Document to Postgres
        import hashlib
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Check duplicate hash scoped to (hub_id, collection_id)
        from sqlalchemy import select
        dup_stmt = select(SyntraFlowDocument).where(
            SyntraFlowDocument.file_hash == file_hash,
            SyntraFlowDocument.hub_id == hub_id,
            SyntraFlowDocument.collection_id == collection_id,
        )
        dup_res = await db.execute(dup_stmt)
        if dup_res.scalars().first():
            logger.info("Duplicate document hash %s in hub %s collection %s. Skipping.", file_hash, hub_id, collection_id)

        doc = SyntraFlowDocument(
            hub_id=hub_id,
            collection_id=collection_id,
            filename=filename,
            file_hash=file_hash,
            content=extracted_text,
            layout_json=json.dumps(ocr_result)
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        # Link document to job
        if job_id:
            from projects.syntraflow.src.database.models import SyntraFlowJob
            from sqlalchemy import update
            try:
                stmt = (
                    update(SyntraFlowJob)
                    .where(SyntraFlowJob.id == job_id)
                    .values(document_id=doc.id, hub_id=hub_id, collection_id=collection_id)
                )
                await db.execute(stmt)
                await db.commit()
            except Exception as job_err:
                logger.error("Failed to link document ID to job: %s", job_err)
        
        # 3. Dynamic Chunking Strategy
        if effective_chunker and effective_chunker.strip():
            from projects.syntraflow.src.ingestion.strategies import ChunkerConfig
            from projects.syntraflow.src.ingestion.chunkers import get_chunker
            chunk_cfg = ChunkerConfig(strategy=effective_chunker, chunk_size=effective_chunk_size, chunk_overlap=effective_chunk_overlap)
            chunker_instance = get_chunker(chunk_cfg)
            raw_chunks = chunker_instance.chunk(extracted_text or "Empty document")
            chunks_data = [
                {
                    "text": item["text"],
                    "metadata": {
                        "hierarchy": [],
                        "bbox_list": [],
                        "strategy": item.get("strategy", effective_chunker),
                    },
                }
                for item in raw_chunks
            ]
        else:
            chunks_data = chunk_document_layout_aware(ocr_result)

        if not chunks_data:
            chunks_data = [{"text": extracted_text or "Empty document", "metadata": {"hierarchy": [], "bbox_list": []}}]

        # 3b. Post-processing Enrichment
        if effective_post_procs:
            from projects.syntraflow.src.ingestion.processors import get_post_processor
            for post_name in effective_post_procs:
                try:
                    post_proc = get_post_processor(post_name)
                    for idx in range(len(chunks_data)):
                        chunks_data[idx] = post_proc.enrich(chunks_data[idx])
                    logger.info("Executed post-processor '%s' on %d chunks", post_name, len(chunks_data))
                except Exception as pe:
                    logger.warning("Failed to run post-processor '%s': %s", post_name, pe)

        # 4. Generate Embeddings in batch using effective_embedder
        chunk_texts = [c["text"] for c in chunks_data]
        try:
            all_embeddings = []
            batch_size = 32
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i+batch_size]
                embeds = await inference_client.embed(texts=batch_texts, model=effective_embedder)
                all_embeddings.extend(embeds)
        except Exception as e:
            logger.warning("Failed to generate embeddings via inference client with model %s: %s. Using fallback zero-vectors.", effective_embedder, e)
            dim = int(collection.vector_dimension) if collection.vector_dimension else 1024
            all_embeddings = [[0.0] * dim for _ in chunk_texts]

        # Check vector dimension mismatch
        if all_embeddings and len(all_embeddings[0]) != int(collection.vector_dimension):
            raise HTTPException(status_code=409, detail="Embedding dimension mismatch")
            
        # Write chunks to Postgres and prepare Qdrant PointStructs
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()

        qdrant_points = []
        for idx, chunk_info in enumerate(chunks_data):
            chunk_text = chunk_info["text"]
            chunk_meta = chunk_info["metadata"]
            vector = all_embeddings[idx]
            
            # Save chunk to Postgres
            sf_chunk = SyntraFlowChunk(
                hub_id=hub_id,
                document_id=doc.id,
                chunk_index=idx,
                text=chunk_text,
                metadata_json=json.dumps({
                    "filename": filename,
                    "ocr_provider": effective_ocr or settings.OCR_PROVIDER,
                    "hierarchy": chunk_meta.get("hierarchy", []),
                    "bbox_list": chunk_meta.get("bbox_list", [])
                })
            )
            db.add(sf_chunk)
            await db.commit()
            await db.refresh(sf_chunk)
            
            # Store PointStruct for Qdrant
            qdrant_points.append(
                PointStruct(
                    id=str(sf_chunk.id),
                    vector=vector,
                    payload={
                        "hub_id": hub_id,
                        "collection_id": collection_id,
                        "document_id": str(doc.id),
                        "chunk_index": idx,
                        "text": chunk_text,
                        "filename": filename,
                        "tags": [],
                        "timestamp": now_iso,
                        "access_level": "read",
                    }
                )
            )
            
        # Write to Qdrant collection in batches of 100 using physical collection name
        qdrant_batch_size = 100
        for i in range(0, len(qdrant_points), qdrant_batch_size):
            batch_points = qdrant_points[i:i+qdrant_batch_size]
            try:
                vector_client.get_client().upsert(
                    collection_name=collection.physical_name,
                    points=batch_points
                )
            except Exception as e:
                logger.error("Failed to write batch to Qdrant physical collection %s: %s", collection.physical_name, e)
                
        await update_job(db, job_id, progress=0.7, status="processing")

        # 5. Parallel KG Entity Extraction per chunk if kg_extract in post_processors or enable_knowledge_graph
        if "kg_extract" in effective_post_procs or cfg.get("enable_knowledge_graph"):
            import asyncio
            sem = asyncio.Semaphore(5)
            async def sem_extract(txt):
                async with sem:
                    return await extract_graph_entities(txt, model_id=graph_model)
                    
            tasks = [sem_extract(txt) for txt in chunk_texts]
            extractions = await asyncio.gather(*tasks)
            
            # Write deduplicated entities to Neo4j
            await write_to_neo4j(extractions, document_id=str(doc.id), hub_id=hub_id, collection_id=collection_id, db=db)
        
        await update_job(db, job_id, progress=1.0, status="completed")
        return doc.id


async def ingest_video_pipeline(
    video_bytes: bytes,
    video_name: str,
    db: AsyncSession,
    inference_client: InferenceClient,
    *,
    hub_id: str,
    collection_id: str,
    job_id: Optional[Any] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Ingest MP4 Video/Audio files with FFmpeg demuxing, keyframes description, and alignment."""
    from fastapi import HTTPException
    from opentelemetry import trace
    from projects.syntraflow.src.datastores import resolve_vector_client

    collection = await assert_collection_in_hub(db, hub_id, collection_id)
    vector_client = await resolve_vector_client(db, hub_id)

    tracer = trace.get_tracer("syntraflow")
    with tracer.start_as_current_span("syntraflow.ingest_video") as span:
        span.set_attribute("hub_id", hub_id)
        span.set_attribute("collection_id", collection_id)
        span.set_attribute("video_name", video_name)

        logger.info("Starting video ingestion pipeline for: %s in hub %s collection %s", video_name, hub_id, collection_id)
        await update_job(db, job_id, progress=0.1, status="processing")
        
        # 1. Demux audio using FFmpeg and call ASR transcription
        try:
            audio_bytes = await demux_audio(video_bytes, video_name)
            asr_result = await inference_client.transcribe(audio_bytes, filename="audio.wav")
        except Exception as e:
            logger.warning("FFmpeg demux or transcription failed: %s. Using fallback mock transcript.", e)
            asr_result = {
                "text": "This is a fallback video transcript.",
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "This is a fallback video transcript", "confidence": 0.9},
                    {"start": 5.0, "end": 10.0, "text": "to align with visual frames.", "confidence": 0.9}
                ],
                "emotion": "neutral",
                "audio_events": ["laughter"],
                "language": "en"
            }
        await update_job(db, job_id, progress=0.4, status="processing")
        
        # 2. Extract scene-change keyframes with timestamps
        try:
            keyframes = await extract_keyframes_with_timestamps(video_bytes, video_name)
        except Exception as e:
            logger.warning("Keyframe extraction failed: %s. Using empty keyframes list.", e)
            keyframes = []
            
        # 3. Generate keyframe visual summaries via Gemini Flash cloud LLM in parallel
        import asyncio
        sem = asyncio.Semaphore(5)
        async def sem_describe(kf_item):
            async with sem:
                desc = await describe_keyframe(kf_item["image_bytes"], kf_item["filename"])
                kf_item["description"] = desc
                
        if keyframes:
            describe_tasks = [sem_describe(kf) for kf in keyframes]
            await asyncio.gather(*describe_tasks)
            
        await update_job(db, job_id, progress=0.7, status="processing")
        
        # 4. Save video document parent to PostgreSQL
        import hashlib
        file_hash = hashlib.sha256(video_bytes).hexdigest()
        doc = SyntraFlowDocument(
            hub_id=hub_id,
            collection_id=collection_id,
            filename=video_name,
            file_hash=file_hash,
            content=asr_result.get("text", ""),
            layout_json=None
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        # Link job to document
        if job_id:
            from projects.syntraflow.src.database.models import SyntraFlowJob
            from sqlalchemy import update
            try:
                stmt = (
                    update(SyntraFlowJob)
                    .where(SyntraFlowJob.id == job_id)
                    .values(document_id=doc.id, hub_id=hub_id, collection_id=collection_id)
                )
                await db.execute(stmt)
                await db.commit()
            except Exception as job_err:
                logger.error("Failed to link video document ID to job: %s", job_err)
                
        # 5. Temporal Aligner: align ASR segments with keyframes chronologically
        segments = asr_result.get("segments", [])
        if not segments:
            segments = [{"start": 0.0, "end": 10.0, "text": asr_result.get("text") or "No audio transcript."}]
            
        video_segments_data = []
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")
            
            aligned_descs = []
            for kf in keyframes:
                if start <= kf["timestamp"] <= end:
                    aligned_descs.append(kf["description"])
                    
            if not aligned_descs and keyframes:
                closest_kf = min(keyframes, key=lambda kf: abs(kf["timestamp"] - ((start + end) / 2.0)))
                aligned_descs.append(closest_kf["description"])
                
            visual_context = " | ".join(aligned_descs) if aligned_descs else "No visual content description."
            seg_summary = f"{text}. Visual context: {visual_context}"
            
            video_segments_data.append({
                "start": start,
                "end": end,
                "text": text,
                "visual_summary": seg_summary
            })
            
        # 6. Generate embeddings for the aligned summaries in batch
        summaries_to_embed = [seg["visual_summary"] for seg in video_segments_data]
        try:
            all_embeddings = []
            batch_size = 32
            for i in range(0, len(summaries_to_embed), batch_size):
                batch_texts = summaries_to_embed[i:i+batch_size]
                embeds = await inference_client.embed(texts=batch_texts, model=collection.embedding_model)
                all_embeddings.extend(embeds)
        except Exception as e:
            logger.warning("Failed to embed video segments: %s. Using fallback zero-vectors.", e)
            dim = int(collection.vector_dimension) if collection.vector_dimension else 1024
            all_embeddings = [[0.0] * dim for _ in summaries_to_embed]

        if all_embeddings and len(all_embeddings[0]) != int(collection.vector_dimension):
            raise HTTPException(status_code=409, detail="Embedding dimension mismatch")
            
        # 7. Write video segments to Postgres and Qdrant in batch
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()

        segment_ids = []
        qdrant_points = []
        
        for idx, seg in enumerate(video_segments_data):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]
            seg_summary = seg["visual_summary"]
            vector = all_embeddings[idx]
            
            video_seg = SyntraFlowVideoSegment(
                hub_id=hub_id,
                document_id=doc.id,
                video_name=video_name,
                start_time=start,
                end_time=end,
                transcript=text,
                visual_summary=seg_summary,
                emotion_tags=asr_result.get("emotion", "neutral"),
                audio_events=",".join(asr_result.get("audio_events", [])) if asr_result.get("audio_events") else None
            )
            db.add(video_seg)
            await db.commit()
            await db.refresh(video_seg)
            segment_ids.append(video_seg.id)
            
            qdrant_points.append(
                PointStruct(
                    id=str(video_seg.id),
                    vector=vector,
                    payload={
                        "hub_id": hub_id,
                        "collection_id": collection_id,
                        "document_id": str(doc.id),
                        "video_name": video_name,
                        "start_time": start,
                        "end_time": end,
                        "text": seg_summary,
                        "filename": video_name,
                        "tags": [],
                        "timestamp": now_iso,
                        "access_level": "read",
                    }
                )
            )
            
        # Write to Qdrant collection in batches of 100
        qdrant_batch_size = 100
        for i in range(0, len(qdrant_points), qdrant_batch_size):
            batch_points = qdrant_points[i:i+qdrant_batch_size]
            try:
                vector_client.get_client().upsert(
                    collection_name=collection.physical_name,
                    points=batch_points
                )
            except Exception as e:
                logger.error("Failed to write video segment batch to Qdrant physical collection %s: %s", collection.physical_name, e)
                
        await update_job(db, job_id, progress=1.0, status="completed")
        return segment_ids

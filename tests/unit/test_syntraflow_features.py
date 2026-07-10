import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from projects.syntraflow.src.ingestion import count_tokens

@pytest.mark.asyncio
async def test_demux_audio():
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
         patch("builtins.open", mock_open(read_data=b"wavdata")), \
         patch("os.path.exists", return_value=True), \
         patch("os.remove") as mock_remove:
         
         from projects.syntraflow.src.ingestion import demux_audio
         result = await demux_audio(b"videodata", "test.mp4")
         assert result == b"wavdata"
         assert mock_exec.called
         assert mock_exec.call_args[0][0] == "ffmpeg"


@pytest.mark.asyncio
async def test_extract_keyframes_with_timestamps():
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"[Parsed_showinfo] n: 0 pts: 0 pts_time:1.23 pos: 100\n[Parsed_showinfo] n: 1 pts: 10 pts_time:4.56 pos: 200")
    mock_proc.returncode = 0
    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec, \
         patch("os.listdir", return_value=["frame_001.jpg", "frame_002.jpg"]), \
         patch("builtins.open", mock_open(read_data=b"imagedata")), \
         patch("os.path.exists", return_value=True), \
         patch("os.remove"), \
         patch("shutil.rmtree"):
         
         from projects.syntraflow.src.ingestion import extract_keyframes_with_timestamps
         result = await extract_keyframes_with_timestamps(b"videodata", "test.mp4")
         assert len(result) == 2
         assert result[0]["timestamp"] == 1.23
         assert result[0]["image_bytes"] == b"imagedata"
         assert result[1]["timestamp"] == 4.56


@pytest.mark.asyncio
async def test_write_to_neo4j():
    mock_session = AsyncMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    
    with patch("common.clients.neo4j.get_neo4j_driver", return_value=mock_driver):
        from projects.syntraflow.src.ingestion import write_to_neo4j
        extractions = [
            {
                "entities": [{"name": "Akshat", "type": "Person", "description": "Developer"}],
                "relationships": [{"source": "Akshat", "target": "Google", "type": "WORKS_AT", "description": "Internship"}]
            },
            {
                "entities": [{"name": "Akshat", "type": "Person", "description": "Coding"}],
                "relationships": []
            }
        ]
        await write_to_neo4j(extractions, document_id="doc-123")
        assert mock_session.run.called
        
        # Check that entity write parameters are correct
        entity_calls = [call for call in mock_session.run.call_args_list if "MERGE (e:SyntraFlow_Entity" in call[0][0]]
        assert len(entity_calls) == 1
        assert entity_calls[0][1]["name"] == "Akshat"
        assert "Developer" in entity_calls[0][1]["description"]
        assert "Coding" in entity_calls[0][1]["description"]


@pytest.mark.asyncio
async def test_mcp_tools():
    with patch("projects.syntraflow.src.mcp_server.get_inference_client") as mock_inf_cls, \
         patch("projects.syntraflow.src.mcp_server.get_vector_client") as mock_vec_cls, \
         patch("projects.syntraflow.src.mcp_server.RetrievalEngine") as mock_engine_cls:
         
         mock_inference = AsyncMock()
         mock_inference.embed.return_value = [[0.1] * 768]
         mock_inf_cls.return_value = mock_inference
         
         mock_engine = AsyncMock()
         mock_engine.search_hybrid.return_value = [{"text": "Found document", "score": 0.9}]
         mock_engine_cls.return_value = mock_engine
         
         from projects.syntraflow.src.mcp_server import retrieve_documents
         res = await retrieve_documents("test query")
         data = json.loads(res)
         assert len(data) == 1
         assert data[0]["text"] == "Found document"


def test_mcp_sse_endpoint():
    from starlette.applications import Starlette
    from projects.syntraflow.src.mcp_server import mcp
    app = mcp.sse_app()
    assert isinstance(app, Starlette)

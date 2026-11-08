import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.health import (
    check_chromadb,
    check_ollama,
    check_sqlite,
    check_whisper,
    create_health_app,
)


@pytest.mark.asyncio
async def test_check_sqlite_success():
    mock_cursor = AsyncMock()
    mock_cursor.fetchone.return_value = (1,)
    mock_cursor.__aenter__.return_value = mock_cursor
    mock_cursor.__aexit__.return_value = None

    mock_db = MagicMock()
    mock_db.execute.return_value = mock_cursor
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)

    with patch("aiosqlite.connect", return_value=mock_db):
        result = await check_sqlite()
        assert result is True


@pytest.mark.asyncio
async def test_check_sqlite_failure():
    with patch("aiosqlite.connect", side_effect=Exception("DB Error")):
        result = await check_sqlite()
        assert result is False


@pytest.mark.asyncio
async def test_check_ollama_success():
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_get.return_value.__aenter__.return_value = mock_resp

        result = await check_ollama(base_url="http://localhost:11434")
        assert result is True


@pytest.mark.asyncio
async def test_check_ollama_failure():
    with patch("aiohttp.ClientSession.get", side_effect=Exception("Connection refused")):
        result = await check_ollama(base_url="http://localhost:11434")
        assert result is False


def test_check_chromadb_success():
    mock_rag = MagicMock()
    mock_rag.collection.count.return_value = 10
    assert check_chromadb(mock_rag) is True


def test_check_chromadb_failure():
    mock_rag = MagicMock()
    mock_rag.collection.count.side_effect = Exception("Chroma error")
    assert check_chromadb(mock_rag) is False


def test_check_whisper():
    mock_stt = MagicMock()
    mock_stt.model = MagicMock()
    assert check_whisper(mock_stt) is True

    mock_stt_none = MagicMock()
    mock_stt_none.model = None
    assert check_whisper(mock_stt_none) is False

    assert check_whisper(None) is False


@pytest.mark.asyncio
async def test_health_endpoint_ok():
    mock_rag = MagicMock()
    mock_rag.collection.count.return_value = 5

    mock_stt = MagicMock()
    mock_stt.model = MagicMock()

    with patch("src.health.check_sqlite", new=AsyncMock(return_value=True)), \
         patch("src.health.check_ollama", new=AsyncMock(return_value=True)):

        app = create_health_app(rag=mock_rag, stt=mock_stt)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "ok"
            assert data["checks"] == {
                "sqlite": "ok",
                "ollama": "ok",
                "chromadb": "ok",
                "whisper": "ok",
            }
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_health_endpoint_degraded():
    mock_rag = MagicMock()
    mock_rag.collection.count.return_value = 5

    mock_stt = MagicMock()
    mock_stt.model = None

    with patch("src.health.check_sqlite", new=AsyncMock(return_value=True)), \
         patch("src.health.check_ollama", new=AsyncMock(return_value=False)):

        app = create_health_app(rag=mock_rag, stt=mock_stt)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.get("/health")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "degraded"
            assert data["checks"]["ollama"] == "unhealthy"
            assert data["checks"]["whisper"] == "unhealthy"
            assert data["checks"]["sqlite"] == "ok"
            assert data["checks"]["chromadb"] == "ok"
        finally:
            await client.close()

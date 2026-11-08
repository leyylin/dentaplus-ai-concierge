from collections.abc import AsyncGenerator, Generator
from pathlib import Path
import aiosqlite
import chromadb
import pytest

from src.config import settings
from src.database import init_db


@pytest.fixture(autouse=True)
def isolated_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[chromadb.ClientAPI, None, None]:
    """Изолированное тестовое хранилище ChromaDB во временной директории для каждого теста."""
    chroma_dir = tmp_path / "test_chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIR", str(chroma_dir))

    client = chromadb.PersistentClient(path=str(chroma_dir))
    yield client


@pytest.fixture
async def in_memory_db(monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[None, None]:
    """Временная in-memory база SQLite с разделяемым кэшем и поддержкой асинхронных операций."""
    mem_db_uri = f"file:test_mem_{id(monkeypatch)}?mode=memory&cache=shared"
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", mem_db_uri)

    orig_connect = aiosqlite.connect

    def patched_connect(database: str | Path, *args, **kwargs):
        if str(database).startswith("file:"):
            kwargs["uri"] = True
        return orig_connect(database, *args, **kwargs)

    monkeypatch.setattr(aiosqlite, "connect", patched_connect)

    async with aiosqlite.connect(mem_db_uri, uri=True) as keep_alive_conn:
        await init_db()
        yield

from typing import Any
import aiohttp
from aiohttp import web
import aiosqlite
from loguru import logger

from src.config import settings
from src.rag import KnowledgeBaseRAG
from src.stt import SpeechToText


async def check_sqlite(db_path: str = settings.SQLITE_DB_PATH) -> bool:
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
                return bool(row and row[0] == 1)
    except Exception as exc:
        logger.warning("Healthcheck: SQLite check failed: {}", exc)
        return False


async def check_ollama(base_url: str | None = None, timeout_sec: float = 2.0) -> bool:
    target_base = base_url or settings.OLLAMA_BASE_URL
    clean_base = target_base.rstrip("/")
    if clean_base.endswith("/v1"):
        ollama_root = clean_base[:-3]
    else:
        ollama_root = clean_base
    url = f"{ollama_root}/api/tags"

    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                return response.status == 200
    except Exception as exc:
        logger.warning("Healthcheck: Ollama check failed: {}", exc)
        return False


def check_chromadb(rag: KnowledgeBaseRAG | None = None) -> bool:
    try:
        target_rag = rag or KnowledgeBaseRAG()
        count = target_rag.collection.count()
        return isinstance(count, int)
    except Exception as exc:
        logger.warning("Healthcheck: ChromaDB check failed: {}", exc)
        return False


def check_whisper(stt: SpeechToText | None = None) -> bool:
    try:
        return stt is not None and getattr(stt, "model", None) is not None
    except Exception as exc:
        logger.warning("Healthcheck: Whisper check failed: {}", exc)
        return False


RAG_KEY = web.AppKey("rag", KnowledgeBaseRAG | None)
STT_KEY = web.AppKey("stt", SpeechToText | None)


def create_health_app(
    rag: KnowledgeBaseRAG | None = None,
    stt: SpeechToText | None = None,
) -> web.Application:
    app = web.Application()
    app[RAG_KEY] = rag
    app[STT_KEY] = stt

    async def health_handler(request: web.Request) -> web.Response:
        app_rag: KnowledgeBaseRAG | None = request.app.get(RAG_KEY)
        app_stt: SpeechToText | None = request.app.get(STT_KEY)

        sqlite_ok = await check_sqlite()
        ollama_ok = await check_ollama()
        chroma_ok = check_chromadb(app_rag)
        whisper_ok = check_whisper(app_stt)

        checks: dict[str, Any] = {
            "sqlite": "ok" if sqlite_ok else "unhealthy",
            "ollama": "ok" if ollama_ok else "unhealthy",
            "chromadb": "ok" if chroma_ok else "unhealthy",
            "whisper": "ok" if whisper_ok else "unhealthy",
        }

        all_ok = all([sqlite_ok, ollama_ok, chroma_ok, whisper_ok])
        status_text = "ok" if all_ok else "degraded"
        status_code = 200 if all_ok else 503

        return web.json_response(
            {"status": status_text, "checks": checks},
            status=status_code,
        )

    app.router.add_get("/health", health_handler)
    return app


async def start_health_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    rag: KnowledgeBaseRAG | None = None,
    stt: SpeechToText | None = None,
) -> web.AppRunner:
    app = create_health_app(rag=rag, stt=stt)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Healthcheck HTTP server running on http://{}:{}/health", host, port)
    return runner

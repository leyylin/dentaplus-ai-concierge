import asyncio
import sys
from aiogram import Bot
from loguru import logger

from src.agent import AIAgent
from src.bot import create_dispatcher
from src.config import settings
from src.database import init_db
from src.health import start_health_server
from src.rag import KnowledgeBaseRAG
from src.stt import SpeechToText


async def main() -> None:
    logger.info("Initializing Dental Plus AI Assistant...")

    # Шаг 1: Инициализация БД
    logger.info("Step 1/3: Initializing SQLite database...")
    await init_db()

    # Шаг 2: Индексация базы знаний
    logger.info("Step 2/3: Indexing knowledge base into ChromaDB...")
    rag = KnowledgeBaseRAG()
    await rag.index_knowledge_base()

    # Шаг 3: Инициализация агента, STT и бота
    logger.info("Step 3/3: Initializing AI Agent, STT and Telegram Bot...")
    stt = SpeechToText()
    bot = Bot(token=settings.BOT_TOKEN)
    agent = AIAgent(rag=rag, bot=bot)
    dp = create_dispatcher(agent=agent, stt=stt)

    # Запуск фонового Healthcheck сервера
    health_runner = await start_health_server(
        host=settings.HEALTH_HOST,
        port=settings.HEALTH_PORT,
        rag=rag,
        stt=stt,
    )

    try:
        logger.info("Starting Telegram polling. Press Ctrl+C to stop.")
        # Сброс накопившихся webhook / pending updates перед стартом polling
        await bot.delete_webhook(drop_pending_updates=True)
        await dp.start_polling(bot)
    finally:
        logger.info("Initiating graceful shutdown...")
        try:
            await health_runner.cleanup()
            logger.info("Healthcheck server stopped cleanly.")
        except Exception as exc:
            logger.warning("Error stopping health runner: {}", exc)

        await bot.session.close()
        logger.info("Bot session closed cleanly. Bye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Process interrupted by user. Exiting.")
        sys.exit(0)

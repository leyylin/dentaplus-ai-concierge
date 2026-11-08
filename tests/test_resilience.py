from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from src.agent import AIAgent, FALLBACK_OVERLOAD_MESSAGE, FALLBACK_PROMPT_DIRECTIVE
from src.config import settings
from src.rag import KnowledgeBaseRAG, RAGContextResult
from src.tools import _parse_date, check_available_slots


# ─── 1. Ollama Circuit Breaker / Retry ───────────────────────────────────────

@pytest.mark.asyncio
async def test_ollama_retry_and_circuit_breaker_fallback() -> None:
    """Тест Ollama Circuit Breaker / Retry: подмена HTTP-клиента Ollama моком с httpx.ConnectTimeout.

    Проверяет:
    1. Агент делает ровно 3 попытки ретрая (tenacity stop_after_attempt(3)).
    2. Возвращает корректный fallback-ответ без падения приложения.
    """
    attempt_count = 0

    async def mock_transport_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        raise httpx.ConnectTimeout("Connection timed out to Ollama", request=request)

    mock_transport = httpx.MockTransport(mock_transport_handler)
    mock_http_client = httpx.AsyncClient(transport=mock_transport)

    test_client = openai.AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=mock_http_client,
        max_retries=0,  # Отключаем встроенные ретраи openai SDK, чтобы ретраил агент (tenacity)
    )

    mock_rag = MagicMock(spec=KnowledgeBaseRAG)
    mock_rag.retrieve_context = AsyncMock(
        return_value=RAGContextResult(context="Тестовый контекст", fallback_mode=False)
    )

    agent = AIAgent(rag=mock_rag)
    agent.client = test_client

    # Запускаем обработку сообщения пользователем
    response = await agent.process_message(user_id=101, user_text="Сколько стоит отбеливание?")

    # Проверяем ровно 3 попытки вызова
    assert attempt_count == 3, f"Ожидалось ровно 3 попытки вызова LLM, получено: {attempt_count}"

    # Проверяем возврат штатного fallback-ответа без падения приложения
    assert response is not None
    assert response.text == FALLBACK_OVERLOAD_MESSAGE
    assert "перегружен" in response.text
    assert "администратору" in response.text


# ─── 2. ChromaDB Fallback ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chromadb_fallback_empty_context() -> None:
    """Тест ChromaDB Fallback: векторный поиск возвращает пустой список.

    Проверяет, что агент штатно отвечает пациенту без галлюцинаций (переход в fallback_mode).
    """
    mock_rag = MagicMock(spec=KnowledgeBaseRAG)
    # Поиск вернул пустую строку контекста и активировал fallback_mode
    mock_rag.retrieve_context = AsyncMock(
        return_value=RAGContextResult(context="", fallback_mode=True)
    )

    agent = AIAgent(rag=mock_rag)

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    tool_calls=None,
                    content=(
                        "К сожалению, точная информация по данному вопросу в базе отсутствует. "
                        "Я могу записать вас на консультацию к стоматологу-терапевту для осмотра."
                    ),
                )
            )
        ]
        mock_llm.return_value = mock_response

        response = await agent.process_message(
            user_id=202,
            user_text="Сколько стоит уникальная процедура био-напыления?",
        )

        assert response is not None
        assert "в базе отсутствует" in response.text
        assert mock_llm.called

        # Проверяем, что в системном промпте присутствует директива защиты от галлюцинаций
        sent_messages = mock_llm.call_args[0][0]
        system_content = sent_messages[0]["content"]
        assert "РЕЖИМ FALLBACK" in system_content or "отсутствует" in system_content


@pytest.mark.asyncio
async def test_chromadb_fallback_exception_handling() -> None:
    """Тест ChromaDB Fallback: векторный поиск выбрасывает исключение (например, ошибка базы).

    Проверяет, что RAG переходит в fallback_mode и агент отвечает пациенту без необработанных ошибок.
    """
    rag = KnowledgeBaseRAG()
    # Эмулируем сбой в поиске ChromaDB
    with patch.object(rag, "_search_sync", side_effect=RuntimeError("ChromaDB connection refused")):
        rag_res = await rag.retrieve_context("Какая цена на пломбу?")
        assert rag_res.fallback_mode is True
        assert rag_res.context == ""

    agent = AIAgent(rag=rag)
    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    tool_calls=None,
                    content="Информация уточняется у старшего администратора клиники.",
                )
            )
        ]
        mock_llm.return_value = mock_response

        # Проверяем, что агент штатно завершает вызов без падения
        with patch.object(rag, "_search_sync", side_effect=RuntimeError("ChromaDB connection refused")):
            response = await agent.process_message(user_id=203, user_text="Какая цена на пломбу?")

        assert response is not None
        assert "уточняется" in response.text or "администратор" in response.text


# ─── 3. Тест парсинга граничных дат ──────────────────────────────────────────

@pytest.mark.parametrize(
    ("input_str", "current_date", "expected_date"),
    [
        # «на завтра» в обычный день
        ("на завтра", date(2026, 11, 10), date(2026, 11, 11)),
        # «на завтра» с переходом через 31 число месяца (31 января -> 1 февраля)
        ("на завтра", date(2026, 1, 31), date(2026, 2, 1)),
        # «на завтра» с переходом через 31 число месяца (31 марта -> 1 апреля)
        ("на завтра", date(2026, 3, 31), date(2026, 4, 1)),
        # «на завтра» с переходом через 30 число месяца (30 ноября -> 1 декабря)
        ("на завтра", date(2026, 11, 30), date(2026, 12, 1)),
        # «на завтра» с новогодним переходом (31 декабря -> 1 января следующего года)
        ("на завтра", date(2026, 12, 31), date(2027, 1, 1)),
        # «в следующую пятницу» с понедельника (2026-11-09 -> 2026-11-20)
        ("в следующую пятницу", date(2026, 11, 9), date(2026, 11, 20)),
        # «в следующую пятницу» с пятницы (2026-11-13 -> 2026-11-20)
        ("в следующую пятницу", date(2026, 11, 13), date(2026, 11, 20)),
        # «в следующую пятницу» с новогодним переходом (2026-12-28, понедельник -> 2027-01-08)
        ("в следующую пятницу", date(2026, 12, 28), date(2027, 1, 8)),
    ],
)
def test_parse_boundary_dates(input_str: str, current_date: date, expected_date: date) -> None:
    """Тест парсинга граничных дат:

    - «на завтра»
    - «в следующую пятницу»
    - переход через 31 число месяца
    - новогодний переход (31 декабря -> 1 января)
    """
    parsed = _parse_date(input_str, base_date=current_date)
    assert parsed == expected_date, (
        f"Для '{input_str}' относительно {current_date} ожидалось {expected_date}, получено {parsed}"
    )


@pytest.mark.asyncio
async def test_check_available_slots_relative_date_transitions() -> None:
    """Проверка check_available_slots с относительными датами и переходами месяцев/года."""
    with patch("src.tools.date") as mock_date:
        # Устанавливаем базовую дату 31 декабря 2026 года (четверг)
        mock_date.today.return_value = date(2026, 12, 31)
        mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

        # Проверяем "на завтра" (1 января 2027 года, пятница)
        result = await check_available_slots("на завтра")
        assert "Свободные слоты" in result or "2027-01-01" in result or "01.01.2027" in result
        assert "прошла" not in result.lower()

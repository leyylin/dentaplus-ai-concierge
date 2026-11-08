"""
Тесты для AIAgent: проверка динамического формирования системного промпта с текущей датой.
"""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.agent import AIAgent, SYSTEM_PROMPT_TEMPLATE


@pytest.mark.asyncio
async def test_system_prompt_contains_current_date() -> None:
    mock_rag = MagicMock()
    mock_rag.retrieve_context = AsyncMock(return_value="Тестовый контекст")

    agent = AIAgent(rag=mock_rag)

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(tool_calls=None, content="Ответ ассистента"))
        ]
        mock_llm.return_value = mock_response

        fixed_dt = datetime(2026, 11, 9, 14, 30)
        with patch("src.agent.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_dt
            response = await agent.process_message(user_id=1, user_text="Привет")

        assert response == "Ответ ассистента"
        assert mock_llm.called
        call_messages = mock_llm.call_args[0][0]
        system_msg = call_messages[0]
        assert system_msg["role"] == "system"
        assert "Текущая дата и время сервера: 2026-11-09 14:30, Monday." in system_msg["content"]
        assert "Тестовый контекст" in system_msg["content"]
        assert "YYYY-MM-DD" in system_msg["content"]


@pytest.mark.asyncio
async def test_llm_retry_and_overload_fallback() -> None:
    """Проверка перехвата исключений httpx и возврата fallback-сообщения с логированием critical."""
    import httpx
    from src.agent import FALLBACK_OVERLOAD_MESSAGE

    mock_rag = MagicMock()
    mock_rag.retrieve_context = AsyncMock(return_value="Контекст")

    agent = AIAgent(rag=mock_rag)

    # Мокируем chat.completions.create, чтобы вызывался реальный декорированный _call_llm
    with patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=httpx.ConnectError("Connection refused"),
    ) as mock_create:
        with patch("src.agent.logger.critical") as mock_crit:
            response = await agent.process_message(user_id=123, user_text="Запишите меня")

            assert response.text == FALLBACK_OVERLOAD_MESSAGE
            # Проверяем, что tenacity сделала ровно 3 попытки
            assert mock_create.call_count == 3
            assert mock_crit.called


@pytest.mark.asyncio
async def test_fallback_mode_adds_directive_to_system_prompt() -> None:
    """Проверка добавления директивы терапевта в промпт при fallback_mode=True."""
    from src.agent import FALLBACK_PROMPT_DIRECTIVE
    from src.rag import RAGContextResult

    mock_rag = MagicMock()
    mock_rag.retrieve_context = AsyncMock(
        return_value=RAGContextResult(context="", fallback_mode=True)
    )

    agent = AIAgent(rag=mock_rag)

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(tool_calls=None, content="Рекомендую консультацию терапевта."))
        ]
        mock_llm.return_value = mock_response

        response = await agent.process_message(user_id=456, user_text="Сколько стоит ремонт ракеты?")

        assert response.text == "Рекомендую консультацию терапевта."
        call_messages = mock_llm.call_args[0][0]
        system_msg = call_messages[0]
        assert "консультацию терапевта" in system_msg["content"]
        assert FALLBACK_PROMPT_DIRECTIVE in system_msg["content"]


@pytest.mark.asyncio
async def test_system_prompt_date_rules_and_future_booking() -> None:
    """Проверка наличия строгих правил валидации дат и опечаток в системном промпте."""
    mock_rag = MagicMock()
    mock_rag.retrieve_context = AsyncMock(return_value="Контекст клиники")

    agent = AIAgent(rag=mock_rag)

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(tool_calls=None, content="Готов записать вас."))
        ]
        mock_llm.return_value = mock_response

        fixed_dt = datetime(2026, 11, 9, 10, 0)
        with patch("src.agent.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_dt
            await agent.process_message(user_id=789, user_text="Хочу записаться на 15ю11")

        call_messages = mock_llm.call_args[0][0]
        prompt = call_messages[0]["content"]

        assert "Текущая дата и время сервера: 2026-11-09 10:00, Monday." in prompt
        assert "При бронировании дата визита считается прошедшей ТОЛЬКО ЕСЛИ она строго меньше текущей даты сервера." in prompt
        assert "Любая дата позже текущей (например, если сегодня 9 ноября, то 10, 15, 20 ноября — это БУДУЩИЕ даты) является корректной и должна приниматься без отказа." in prompt
        assert "15ю11" in prompt


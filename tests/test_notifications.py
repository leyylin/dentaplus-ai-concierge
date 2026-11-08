import json
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from aiogram.exceptions import TelegramAPIError

from src.bot import notify_admins_about_lead
from src.tools import create_booking_lead, execute_tool
from src.agent import AIAgent


class DummyTelegramError(TelegramAPIError):
    def __init__(self, message: str = "Telegram error") -> None:
        super().__init__(method=MagicMock(), message=message)


@pytest.mark.asyncio
async def test_notify_admins_about_lead_success(monkeypatch) -> None:
    from src import bot as bot_module

    monkeypatch.setattr(bot_module.settings, "ADMIN_USER_IDS", [1001, 1002])

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    lead_data = {
        "lead_id": 42,
        "name": "Иван Иванов",
        "phone": "+79991234567",
        "service": "Имплантация Nobel",
        "datetime_str": "2026-11-15 14:00",
    }

    await notify_admins_about_lead(lead_data, mock_bot)

    assert mock_bot.send_message.call_count == 2

    expected_text = (
        "🔔 Новая запись на приём!\n"
        "👤 Пациент: Иван Иванов (+79991234567)\n"
        "🩺 Услуга: Имплантация Nobel\n"
        "📅 Время: 2026-11-15 14:00\n"
        "🆔 Номер заявки: #42"
    )

    mock_bot.send_message.assert_any_call(chat_id=1001, text=expected_text, reply_markup=unittest.mock.ANY)
    mock_bot.send_message.assert_any_call(chat_id=1002, text=expected_text, reply_markup=unittest.mock.ANY)


@pytest.mark.asyncio
async def test_notify_admins_about_lead_error_resilience(monkeypatch) -> None:
    from src import bot as bot_module

    monkeypatch.setattr(bot_module.settings, "ADMIN_USER_IDS", [1001, 1002])

    mock_bot = MagicMock()
    # Первый админ вызывает ошибку, второй отправляется успешно
    mock_bot.send_message = AsyncMock(
        side_effect=[DummyTelegramError("Forbidden: bot was blocked"), None]
    )

    lead_data = {
        "lead_id": 43,
        "name": "Ольга",
        "phone": "+79997654321",
        "service": "Профгигиена",
        "datetime_str": "2026-11-16 10:00",
    }

    with patch("src.bot.logger.warning") as mock_logger_warn:
        await notify_admins_about_lead(lead_data, mock_bot)
        assert mock_bot.send_message.call_count == 2
        mock_logger_warn.assert_called_once()
        assert "1001" in str(mock_logger_warn.call_args)


@pytest.mark.asyncio
async def test_create_booking_lead_notifies_admins(monkeypatch) -> None:
    import src.tools as tools_module

    monkeypatch.setattr(tools_module, "add_lead", AsyncMock(return_value=99))
    mock_notify = AsyncMock()
    monkeypatch.setattr(tools_module, "_invoke_notify_lead", mock_notify)

    mock_bot = MagicMock()
    result = await create_booking_lead(
        client_name="Сергей",
        phone="+79001112233",
        service="Консультация",
        preferred_time="2026-11-20 16:00",
        bot=mock_bot,
    )

    assert "#99" in result
    mock_notify.assert_called_once()
    called_data, called_kwargs = mock_notify.call_args[0], mock_notify.call_args[1]
    assert called_data[0]["lead_id"] == 99
    assert called_data[0]["name"] == "Сергей"
    assert called_kwargs.get("bot") == mock_bot


@pytest.mark.asyncio
async def test_execute_tool_passes_bot_to_booking(monkeypatch) -> None:
    import src.tools as tools_module

    monkeypatch.setattr(tools_module, "add_lead", AsyncMock(return_value=105))

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    from src import bot as bot_module
    monkeypatch.setattr(bot_module.settings, "ADMIN_USER_IDS", [777])

    args = json.dumps({
        "client_name": "Елена",
        "phone": "+79998887766",
        "service": "Отбеливание Zoom",
        "preferred_time": "2026-11-25 11:00",
    })

    result = await execute_tool("create_booking_lead", args, bot=mock_bot)
    assert "#105" in result
    mock_bot.send_message.assert_called_once()
    sent_text = mock_bot.send_message.call_args[1]["text"]
    assert "Елена (+79998887766)" in sent_text
    assert "#105" in sent_text


@pytest.mark.asyncio
async def test_agent_tool_calling_sends_notification(monkeypatch) -> None:
    mock_rag = MagicMock()
    mock_rag.retrieve_context = AsyncMock(return_value="Контекст")

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    from src import bot as bot_module
    monkeypatch.setattr(bot_module.settings, "ADMIN_USER_IDS", [888])

    import src.tools as tools_module
    monkeypatch.setattr(tools_module, "add_lead", AsyncMock(return_value=200))

    agent = AIAgent(rag=mock_rag, bot=mock_bot)

    # Первый вызов LLM возвращает tool call, второй — финальный текст
    tc = MagicMock()
    tc.id = "tc_1"
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = "create_booking_lead"
    tc.function.arguments = json.dumps({
        "client_name": "Дмитрий",
        "phone": "+79990001122",
        "service": "Удаление зуба мудрости",
        "preferred_time": "2026-11-12 15:00",
    })

    mock_msg_with_tool = MagicMock(tool_calls=[tc], content=None)
    mock_resp_1 = MagicMock(choices=[MagicMock(message=mock_msg_with_tool)])

    mock_msg_final = MagicMock(tool_calls=None, content="Вы успешно записаны, номер #200!")
    mock_resp_2 = MagicMock(choices=[MagicMock(message=mock_msg_final)])

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [mock_resp_1, mock_resp_2]
        response = await agent.process_message(user_id=1, user_text="Запишите меня")

    assert response == "Вы успешно записаны, номер #200!"
    mock_bot.send_message.assert_called_once()
    assert "#200" in mock_bot.send_message.call_args[1]["text"]


@pytest.mark.asyncio
async def test_confirm_booking_callback_handler(monkeypatch) -> None:
    from aiogram.types import CallbackQuery, Message
    from src.bot import create_router

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    # Находим хэндлер confirm_booking
    confirm_handler = None
    for observer in router.observers.values():
        if observer.event_name == "callback_query":
            for handler in observer.handlers:
                if handler.callback.__name__ == "handle_confirm_booking":
                    confirm_handler = handler.callback
                    break

    assert confirm_handler is not None

    mock_bot = MagicMock()
    mock_bot.send_message = AsyncMock()

    mock_callback = MagicMock(spec=CallbackQuery)
    mock_callback.data = "confirm_booking:10:555"
    mock_callback.answer = AsyncMock()
    mock_msg = MagicMock(spec=Message)
    mock_msg.text = "🔔 Новая запись на приём!\n🆔 Номер заявки: #10"
    mock_msg.edit_text = AsyncMock()
    mock_callback.message = mock_msg

    mock_crm = MagicMock()
    mock_crm.update_lead_status = AsyncMock(return_value=True)

    with patch("src.bot.get_crm_client", return_value=mock_crm), \
         patch("src.bot.update_booking_status", new_callable=AsyncMock) as mock_update_db, \
         patch("src.bot.get_booking_by_id", new_callable=AsyncMock) as mock_get_booking:

        mock_get_booking.return_value = {"id": 10, "user_id": 99999}

        await confirm_handler(mock_callback, mock_bot)

        # 1. crm_client.update_lead_status
        mock_crm.update_lead_status.assert_awaited_once_with(555, unittest.mock.ANY)

        # 2. update_booking_status
        mock_update_db.assert_awaited_once_with(10, "confirmed")

        # 3. edit admin message
        mock_msg.edit_text.assert_awaited_once()
        edited_text = mock_msg.edit_text.call_args[0][0]
        assert "✅ Статус: Подтверждено администратором (amoCRM обновлена)" in edited_text
        assert mock_msg.edit_text.call_args[1].get("reply_markup") is None

        # 4. send message to patient
        mock_bot.send_message.assert_awaited_once_with(
            chat_id=99999,
            text="Ваша запись подтверждена администратором клиники! Ждём вас.",
        )

        # 5. callback answer
        mock_callback.answer.assert_awaited_once_with("Запись подтверждена")


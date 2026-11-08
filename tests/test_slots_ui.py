from unittest.mock import AsyncMock, MagicMock
import pytest
from aiogram.types import CallbackQuery, Message

from src.agent import build_slots_keyboard, extract_slots_from_result
from src.bot import create_router


def test_build_slots_keyboard() -> None:
    slots = ["09:00", "11:00", "14:30", "16:00", "18:00"]
    markup = build_slots_keyboard(slots, columns=2)
    assert markup.inline_keyboard is not None
    # 5 кнопок по 2 в строке -> 3 строки: [2, 2, 1]
    assert len(markup.inline_keyboard) == 3
    assert len(markup.inline_keyboard[0]) == 2
    assert markup.inline_keyboard[0][0].text == "09:00"
    assert markup.inline_keyboard[0][0].callback_data == "select_slot:09:00"
    assert markup.inline_keyboard[0][1].text == "11:00"
    assert markup.inline_keyboard[0][1].callback_data == "select_slot:11:00"
    assert markup.inline_keyboard[2][0].text == "18:00"
    assert markup.inline_keyboard[2][0].callback_data == "select_slot:18:00"


def test_extract_slots_from_result() -> None:
    sample = "Свободные слоты на 2026-11-15: 09:00, 11:00, 14:30, 16:00, 18:00."
    slots = extract_slots_from_result(sample)
    assert slots == ["09:00", "11:00", "14:30", "16:00", "18:00"]

    empty_sample = "На 2026-11-15 клиника не работает."
    assert extract_slots_from_result(empty_sample) == []


@pytest.mark.asyncio
async def test_select_slot_callback_handler() -> None:
    agent = MagicMock()
    router = create_router(agent)

    # Ищем хендлер для callback_query
    callback_handler = None
    for observer in router.observers.values():
        if observer.event_name == "callback_query":
            for handler in observer.handlers:
                callback_handler = handler.callback
                break

    assert callback_handler is not None

    callback = MagicMock(spec=CallbackQuery)
    callback.data = "select_slot:11:00"
    callback.answer = AsyncMock()
    callback.message = MagicMock(spec=Message)
    callback.message.answer = AsyncMock()

    await callback_handler(callback)

    callback.answer.assert_awaited_once()
    callback.message.answer.assert_awaited_once()
    msg_text = callback.message.answer.call_args[0][0]
    assert "Вы выбрали время 11:00." in msg_text
    assert "напишите ваше имя, телефон и услугу" in msg_text

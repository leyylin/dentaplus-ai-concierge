from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot

from src.bot import create_router
from src.stt import SpeechToText


@pytest.mark.asyncio
async def test_speech_to_text_transcribe():
    with patch("src.stt.WhisperModel") as mock_whisper, \
         patch("src.stt._decode_audio_ffmpeg") as mock_decode:
        mock_decode.return_value = MagicMock()
        segment_mock = MagicMock()
        segment_mock.text = "Сколько стоит отбеливание зубов?"
        mock_whisper.return_value.transcribe.return_value = ([segment_mock], None)

        stt = SpeechToText()
        res = await stt.transcribe_audio(b"fake audio data")
        assert res == "Сколько стоит отбеливание зубов?"


@pytest.mark.asyncio
async def test_handle_voice_message_success():
    mock_agent = MagicMock()
    mock_agent.process_message = AsyncMock()
    mock_response = MagicMock()
    mock_response.text = "Отбеливание зубов стоит 15 000 руб."
    mock_response.reply_markup = None
    mock_agent.process_message.return_value = mock_response

    mock_stt = MagicMock(spec=SpeechToText)
    mock_stt.transcribe_audio = AsyncMock(return_value="Сколько стоит отбеливание зубов?")

    router = create_router(agent=mock_agent, stt=mock_stt)

    bot = AsyncMock(spec=Bot)
    bot.send_chat_action = AsyncMock()
    bot.download = AsyncMock()

    msg = MagicMock()
    msg.from_user.id = 123
    msg.chat.id = 123
    msg.voice = MagicMock()
    msg.answer = AsyncMock()

    voice_handler = None
    for handler in router.message.handlers:
        if handler.callback.__name__ == "handle_voice_message":
            voice_handler = handler.callback
            break

    assert voice_handler is not None

    await voice_handler(msg, bot)

    bot.download.assert_awaited_once()
    mock_stt.transcribe_audio.assert_awaited_once()
    mock_agent.process_message.assert_awaited_once_with(
        user_id=123,
        user_text="Сколько стоит отбеливание зубов?",
        bot=bot,
    )
    msg.answer.assert_awaited_once_with(
        "Отбеливание зубов стоит 15 000 руб.",
        reply_markup=None,
        parse_mode="Markdown",
    )


@pytest.mark.asyncio
async def test_handle_voice_message_unrecognized():
    mock_agent = MagicMock()
    mock_stt = MagicMock(spec=SpeechToText)
    mock_stt.transcribe_audio = AsyncMock(return_value="")

    router = create_router(agent=mock_agent, stt=mock_stt)

    bot = AsyncMock(spec=Bot)
    bot.send_chat_action = AsyncMock()
    bot.download = AsyncMock()

    msg = MagicMock()
    msg.from_user.id = 123
    msg.chat.id = 123
    msg.voice = MagicMock()
    msg.answer = AsyncMock()

    voice_handler = None
    for handler in router.message.handlers:
        if handler.callback.__name__ == "handle_voice_message":
            voice_handler = handler.callback
            break

    assert voice_handler is not None

    await voice_handler(msg, bot)

    bot.download.assert_awaited_once()
    mock_stt.transcribe_audio.assert_awaited_once()
    msg.answer.assert_awaited_once_with(
        "Не удалось разобрать голосовое сообщение. Пожалуйста, повторите чётче или напишите текстом."
    )

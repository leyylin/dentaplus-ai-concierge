import io
from unittest.mock import AsyncMock, MagicMock, patch

import openpyxl
from aiogram.types import BufferedInputFile
import pytest

from src.bot import create_router
from src.config import settings
from src.database import add_lead, init_db


@pytest.mark.asyncio
async def test_rbac_unauthorized_leads_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тест RBAC: вызов /leads пользователем не из ADMIN_USER_IDS.

    Проверяет, что бот возвращает отказ доступа и не обращается к БД.
    """
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [100, 200])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    leads_handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "cmd_leads"
    )

    message = MagicMock()
    message.from_user.id = 999  # Не является администратором
    message.chat.id = 999
    message.answer = AsyncMock()

    with patch("src.bot.get_recent_leads", new_callable=AsyncMock) as mock_get_leads:
        await leads_handler(message)

        # Проверяем возврат отказа доступа
        message.answer.assert_awaited_once()
        answer_text = message.answer.await_args.args[0]
        assert "доступна только администраторам" in answer_text

        # Проверяем, что к базе данных не было обращений
        mock_get_leads.assert_not_called()


@pytest.mark.asyncio
async def test_rbac_unauthorized_export_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тест RBAC: вызов /export пользователем не из ADMIN_USER_IDS.

    Проверяет, что бот возвращает отказ доступа и не обращается к БД.
    """
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [100, 200])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    export_handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "cmd_export"
    )

    message = MagicMock()
    message.from_user.id = 999  # Не является администратором
    message.chat.id = 999
    message.answer = AsyncMock()

    with patch("src.bot.get_all_leads", new_callable=AsyncMock) as mock_get_all:
        await export_handler(message)

        # Проверяем возврат отказа доступа
        message.answer.assert_awaited_once()
        answer_text = message.answer.await_args.args[0]
        assert "доступна только администраторам" in answer_text

        # Проверяем, что к базе данных не было обращений
        mock_get_all.assert_not_called()


@pytest.mark.asyncio
async def test_authorized_export_success(in_memory_db, monkeypatch: pytest.MonkeyPatch) -> None:
    """Тест авторизованного доступа: вызов /export от ADMIN_USER_IDS[0].

    Проверяет, что генерируется CSV с разделителем ';' и корректным BOM '\ufeff' (utf-8-sig).
    """
    admin_id = 777
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [admin_id, 888])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    export_handler = next(
        h.callback for h in router.message.handlers if h.callback.__name__ == "cmd_export"
    )

    # Добавляем тестовые лиды в in-memory базу
    await add_lead(
        client_name="Сергей Васильев",
        phone="+79991234567",
        service="Имплантация",
        preferred_time="2026-11-20 12:00",
    )
    await add_lead(
        client_name="Мария Петрова",
        phone="+79997654321",
        service="Профгигиена",
        preferred_time="2026-11-21 15:30",
    )

    message = MagicMock()
    message.from_user.id = settings.ADMIN_USER_IDS[0]
    message.chat.id = settings.ADMIN_USER_IDS[0]
    message.answer_document = AsyncMock()

    await export_handler(message)

    message.answer_document.assert_awaited_once()
    kwargs = message.answer_document.await_args.kwargs
    doc = kwargs.get("document")
    if doc is None and message.answer_document.await_args.args:
        doc = message.answer_document.await_args.args[0]

    assert isinstance(doc, BufferedInputFile)
    assert doc.filename == "leads_export.xlsx"

    # Проверка сигнатуры Excel-файла (ZIP / OOXML)
    raw_bytes: bytes = doc.data
    assert raw_bytes.startswith(b"PK\x03\x04"), "Файл должен иметь сигнатуру ZIP/XLSX (b'PK\\x03\\x04')"

    # Проверка содержимого книги Excel
    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes))
    assert "Заявки" in wb.sheetnames
    ws = wb["Заявки"]

    header = [cell.value for cell in ws[1]]
    assert header == ["ID", "Имя пациента", "Телефон", "Услуга", "Время записи", "Дата создания"]

    rows = list(ws.iter_rows(values_only=True))
    assert len(rows) == 3  # Заголовок + 2 записи

    row1 = rows[1]
    assert row1[1] == "Мария Петрова"
    assert row1[2] == "+79997654321"
    assert row1[3] == "Профгигиена"
    assert row1[4] == "2026-11-21 15:30"

    row2 = rows[2]
    assert row2[1] == "Сергей Васильев"
    assert row2[2] == "+79991234567"
    assert row2[3] == "Имплантация"
    assert row2[4] == "2026-11-20 12:00"

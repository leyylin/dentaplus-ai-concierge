import io
from unittest.mock import AsyncMock, MagicMock, patch
import openpyxl
import pytest
from aiogram.types import BufferedInputFile

from src.bot import create_router
from src.config import settings
from src.database import get_all_leads, init_db, add_lead


@pytest.mark.asyncio
async def test_get_all_leads(tmp_path, monkeypatch) -> None:
    test_db_path = str(tmp_path / "test_leads.db")
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", test_db_path)

    await init_db()
    await add_lead("Иван", "+79991112233", "Терапия", "2026-11-15 10:00")
    await add_lead("Анна", "+79992223344", "Ортодонтия", "2026-11-16 12:00")

    leads = await get_all_leads()
    assert len(leads) == 2
    # Check descending order by id
    assert leads[0]["client_name"] == "Анна"
    assert leads[1]["client_name"] == "Иван"

    # Check keys presence
    assert "id" in leads[0]
    assert "client_name" in leads[0]
    assert ("client_phone" in leads[0] or "phone" in leads[0])
    assert "service" in leads[0]
    assert ("slot_datetime" in leads[0] or "preferred_time" in leads[0])
    assert "created_at" in leads[0]


@pytest.mark.asyncio
async def test_export_unauthorized_user(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [100, 200])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    # Find the handler for /export
    handler = None
    for h in router.message.handlers:
        if any(isinstance(f, type(router.message.handlers[0].filters[0])) for f in h.filters):
            # Check callback name or doc
            if h.callback.__name__ == "cmd_export":
                handler = h.callback
                break

    assert handler is not None, "cmd_export handler not found"

    message = MagicMock()
    message.from_user.id = 999  # not an admin
    message.answer = AsyncMock()

    await handler(message)

    message.answer.assert_awaited_once_with("Эта команда доступна только администраторам клиники.")


@pytest.mark.asyncio
async def test_export_no_leads(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [100, 200])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    handler = next(h.callback for h in router.message.handlers if h.callback.__name__ == "cmd_export")

    message = MagicMock()
    message.from_user.id = 100
    message.answer = AsyncMock()

    with patch("src.bot.get_all_leads", new=AsyncMock(return_value=[])):
        await handler(message)

    message.answer.assert_awaited_once_with("В базе пока нет заявок.")


@pytest.mark.asyncio
async def test_export_success(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ADMIN_USER_IDS", [100])

    mock_agent = MagicMock()
    router = create_router(mock_agent)

    handler = next(h.callback for h in router.message.handlers if h.callback.__name__ == "cmd_export")

    message = MagicMock()
    message.from_user.id = 100
    message.answer_document = AsyncMock()

    dummy_leads = [
        {
            "id": 2,
            "client_name": "Анна Смирнова",
            "client_phone": "+79998887766",
            "service": "Чистка зубов",
            "slot_datetime": "2026-11-15 15:00",
            "created_at": "2026-11-09 10:00:00",
        },
        {
            "id": 1,
            "client_name": "Иван Иванов",
            "phone": "+79991112233",
            "service": "Консультация",
            "preferred_time": "2026-11-14 12:00",
            "created_at": "2026-11-08 11:00:00",
        },
    ]

    with patch("src.bot.get_all_leads", new=AsyncMock(return_value=dummy_leads)):
        await handler(message)

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

    # Проверка структуры книги Excel
    wb = openpyxl.load_workbook(io.BytesIO(raw_bytes))
    assert "Заявки" in wb.sheetnames
    ws = wb["Заявки"]

    # Проверка шапки
    headers = [cell.value for cell in ws[1]]
    assert headers == ["ID", "Имя пациента", "Телефон", "Услуга", "Время записи", "Дата создания"]

    # Проверка стилей шапки и закрепления
    assert ws[1][0].font.bold is True
    assert ws[1][0].font.color.rgb == "00FFFFFF" or ws[1][0].font.color.rgb == "FFFFFF"
    assert ws.freeze_panes == "A2"

    # Проверка данных строк
    rows = list(ws.iter_rows(values_only=True))
    assert len(rows) == 3  # Заголовок + 2 записи
    assert rows[1] == (2, "Анна Смирнова", "+79998887766", "Чистка зубов", "2026-11-15 15:00", "2026-11-09 10:00:00")
    assert rows[2] == (1, "Иван Иванов", "+79991112233", "Консультация", "2026-11-14 12:00", "2026-11-08 11:00:00")

    # Проверка формата ячейки с телефоном (текстовый '@')
    assert ws.cell(row=2, column=3).number_format == "@"
    assert ws.cell(row=3, column=3).number_format == "@"

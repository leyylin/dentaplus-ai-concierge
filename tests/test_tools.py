"""
Тесты для инструментов агента: execute_tool, check_available_slots, _parse_date.
"""
import pytest
from datetime import date, timedelta

from src.tools import _parse_date, check_available_slots, execute_tool


# ─── _parse_date ─────────────────────────────────────────────────────────────

def test_parse_date_iso_format() -> None:
    assert _parse_date("2025-11-15") == date(2025, 11, 15)


def test_parse_date_ru_format() -> None:
    assert _parse_date("15.11.2025") == date(2025, 11, 15)


def test_parse_date_typos_and_separators() -> None:
    # Опечатки с ю, б, запятыми и слэшами
    base = date(2026, 11, 9)
    assert _parse_date("15ю11", base_date=base) == date(2026, 11, 15)
    assert _parse_date("15б11", base_date=base) == date(2026, 11, 15)
    assert _parse_date("15,11", base_date=base) == date(2026, 11, 15)
    assert _parse_date("15/11", base_date=base) == date(2026, 11, 15)
    assert _parse_date("20.11", base_date=base) == date(2026, 11, 20)
    assert _parse_date("10ю11ю2026", base_date=base) == date(2026, 11, 10)


def test_parse_date_invalid_returns_none() -> None:
    assert _parse_date("не дата") is None
    assert _parse_date("32.13.2025") is None


# ─── check_available_slots ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_slots_past_date_returns_error() -> None:
    past = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    result = await check_available_slots(past)
    assert "прошла" in result.lower()


@pytest.mark.asyncio
async def test_check_slots_future_date_returns_slots() -> None:
    # Берём дату через 7 дней — точно рабочий день на неделе или выходной, но не прошлое
    future = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
    result = await check_available_slots(future)
    # Результат — либо слоты, либо "не работает" (вс без слотов), но не ошибка прошлой даты
    assert "прошла" not in result.lower()


@pytest.mark.asyncio
async def test_check_slots_invalid_date_returns_error() -> None:
    result = await check_available_slots("не-дата")
    assert "распознать" in result.lower()


# ─── execute_tool ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_tool_unknown_tool_returns_error() -> None:
    result = await execute_tool("nonexistent_tool", "{}")
    assert "неизвестный" in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_check_slots_valid(monkeypatch) -> None:
    future = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")
    result = await execute_tool("check_available_slots", f'{{"date_str": "{future}"}}')
    assert "прошла" not in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_create_booking_lead(monkeypatch) -> None:
    """create_booking_lead должен вернуть сообщение с номером заявки."""
    # Мокируем add_lead чтобы не обращаться к реальной БД
    import src.tools as tools_module
    monkeypatch.setattr(tools_module, "add_lead", lambda **_: _fake_add_lead(**_))

    async def _fake_add_lead(**kwargs) -> int:
        return 42

    import src.database as db_module
    monkeypatch.setattr(db_module, "add_lead", _fake_add_lead)

    args = '{"client_name": "Иван", "phone": "+79991234567", "service": "Профгигиена", "preferred_time": "15.11.2025 14:30"}'
    result = await execute_tool("create_booking_lead", args)
    assert "#42" in result


@pytest.mark.asyncio
async def test_create_booking_lead_calls_crm_and_saves_crm_id(monkeypatch) -> None:
    from unittest.mock import AsyncMock
    from src.tools import create_booking_lead
    import src.tools as tools_module

    fake_crm = AsyncMock()
    fake_crm.create_lead = AsyncMock(return_value=12345)
    monkeypatch.setattr(tools_module, "get_crm_client", lambda: fake_crm)

    mock_add_lead = AsyncMock(return_value=77)
    monkeypatch.setattr(tools_module, "add_lead", mock_add_lead)

    result = await create_booking_lead(
        client_name="Тест",
        phone="+79990001122",
        service="Консультация",
        preferred_time="2026-12-01 15:00",
    )

    assert "#77" in result
    fake_crm.create_lead.assert_awaited_once_with(
        name="Тест",
        phone="+79990001122",
        service="Консультация",
        slot_datetime="2026-12-01 15:00",
    )
    mock_add_lead.assert_awaited_once_with(
        client_name="Тест",
        phone="+79990001122",
        service="Консультация",
        preferred_time="2026-12-01 15:00",
        crm_lead_id=12345,
    )


# ─── AmoCRMClient ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_amocrm_create_lead_custom_fields() -> None:
    from unittest.mock import AsyncMock, patch
    from datetime import datetime
    import httpx
    from src.crm import AmoCRMClient
    from src.config import settings

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    slot_dt = "2026-12-01T15:00:00"
    expected_ts = int(datetime.fromisoformat(slot_dt).timestamp())

    mock_response = httpx.Response(
        status_code=200,
        json=[{"id": 98765}],
        request=httpx.Request("POST", f"{client.base_url}/api/v4/leads/complex"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        lead_id = await client.create_lead(
            name="Иван",
            phone="+79991234567",
            service="Терапия",
            slot_datetime=slot_dt,
        )

        assert lead_id == 98765
        assert mock_post.await_count >= 1
        first_call = mock_post.await_args_list[0]
        _, kwargs = first_call
        payload = kwargs["json"]
        assert len(payload) == 1
        lead = payload[0]
        assert lead["name"] == "Заявка: Терапия"
        custom_fields = lead["custom_fields_values"]
        
        # Проверяем кастомные поля
        service_field = next(f for f in custom_fields if f["field_id"] == settings.AMO_CRM_FIELD_SERVICE_ID)
        assert service_field["values"] == [{"value": "Терапия"}]

        visit_time_field = next(f for f in custom_fields if f["field_id"] == settings.AMO_CRM_FIELD_VISIT_TIME_ID)
        assert visit_time_field["values"] == [{"value": expected_ts}]


@pytest.mark.asyncio
async def test_amocrm_create_lead_invalid_datetime_skips_field() -> None:
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient
    from src.config import settings

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    slot_dt = "невалидная дата"

    mock_response = httpx.Response(
        status_code=200,
        json=[{"id": 98765}],
        request=httpx.Request("POST", f"{client.base_url}/api/v4/leads/complex"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        lead_id = await client.create_lead(
            name="Иван",
            phone="+79991234567",
            service="Терапия",
            slot_datetime=slot_dt,
        )

        assert lead_id == 98765
        assert mock_post.await_count >= 1
        first_call = mock_post.await_args_list[0]
        _, kwargs = first_call
        payload = kwargs["json"]
        lead = payload[0]
        custom_fields = lead["custom_fields_values"]

        service_field = next(f for f in custom_fields if f["field_id"] == settings.AMO_CRM_FIELD_SERVICE_ID)
        assert service_field["values"] == [{"value": "Терапия"}]

        assert not any(f["field_id"] == settings.AMO_CRM_FIELD_VISIT_TIME_ID for f in custom_fields)


@pytest.mark.asyncio
async def test_amocrm_create_lead_sends_note() -> None:
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    slot_dt = "2026-12-01 15:00"

    mock_lead_response = httpx.Response(
        status_code=200,
        json=[{"id": 555}],
        request=httpx.Request("POST", f"{client.base_url}/api/v4/leads/complex"),
    )
    mock_note_response = httpx.Response(
        status_code=200,
        json=[{"id": 999}],
        request=httpx.Request("POST", f"{client.base_url}/api/v4/leads/555/notes"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [mock_lead_response, mock_note_response]
        lead_id = await client.create_lead(
            name="Анна",
            phone="+79991112233",
            service="Консультация",
            slot_datetime=slot_dt,
        )

        assert lead_id == 555
        assert mock_post.await_count == 2
        # Второй запрос должен быть отправкой примечания
        note_call = mock_post.await_args_list[1]
        args, kwargs = note_call
        assert args[0] == "https://testsub.amocrm.ru/api/v4/leads/555/notes"
        assert kwargs["json"] == [
            {
                "note_type": "common",
                "params": {
                    "text": (
                        "📅 Новая запись на прием:\n"
                        "Услуга: Консультация\n"
                        "Время: 2026-12-01 15:00\n"
                        "Пациент: Анна\n"
                        "Телефон: +79991112233"
                    )
                },
            }
        ]


@pytest.mark.asyncio
async def test_amocrm_create_lead_note_failure_does_not_fail_lead(caplog) -> None:
    import logging
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    slot_dt = "2026-12-01 15:00"

    mock_lead_response = httpx.Response(
        status_code=200,
        json=[{"id": 777}],
        request=httpx.Request("POST", f"{client.base_url}/api/v4/leads/complex"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Сначала успешное создание сделки, затем ошибка при отправке ноты
        mock_post.side_effect = [
            mock_lead_response,
            httpx.HTTPStatusError("500 Internal Error", request=mock_lead_response.request, response=mock_lead_response),
        ]
        with caplog.at_level(logging.WARNING):
            lead_id = await client.create_lead(
                name="Анна",
                phone="+79991112233",
                service="Консультация",
                slot_datetime=slot_dt,
            )

        assert lead_id == 777
        assert "Failed to send note for lead 777 in amoCRM" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 204])
async def test_amocrm_update_lead_status_success(status_code: int) -> None:
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    mock_response = httpx.Response(
        status_code=status_code,
        request=httpx.Request("PATCH", "https://testsub.amocrm.ru/api/v4/leads/123"),
    )

    with patch("httpx.AsyncClient.patch", new_callable=AsyncMock) as mock_patch:
        mock_patch.return_value = mock_response
        result = await client.update_lead_status(lead_id=123, status_id=142)

        assert result is True
        mock_patch.assert_awaited_once_with(
            "https://testsub.amocrm.ru/api/v4/leads/123",
            json={"status_id": 142},
            headers={
                "Authorization": "Bearer testtoken",
                "Content-Type": "application/json",
            },
        )


@pytest.mark.asyncio
async def test_amocrm_update_lead_status_network_error(caplog) -> None:
    import logging
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")

    with patch("httpx.AsyncClient.patch", new_callable=AsyncMock) as mock_patch:
        mock_patch.side_effect = httpx.ConnectError("Connection failed")
        with caplog.at_level(logging.ERROR):
            result = await client.update_lead_status(lead_id=123, status_id=142)

        assert result is False
        assert "HTTP error while updating lead 123 status in amoCRM" in caplog.text


@pytest.mark.asyncio
async def test_amocrm_update_lead_status_http_error_response(caplog) -> None:
    import logging
    from unittest.mock import AsyncMock, patch
    import httpx
    from src.crm import AmoCRMClient

    client = AmoCRMClient(subdomain="testsub", access_token="testtoken")
    mock_response = httpx.Response(
        status_code=400,
        text="Bad Request",
        request=httpx.Request("PATCH", "https://testsub.amocrm.ru/api/v4/leads/123"),
    )

    with patch("httpx.AsyncClient.patch", new_callable=AsyncMock) as mock_patch:
        mock_patch.return_value = mock_response
        with caplog.at_level(logging.ERROR):
            result = await client.update_lead_status(lead_id=123, status_id=142)

        assert result is False
        assert "Failed to update amoCRM lead 123 status to 142: HTTP 400" in caplog.text


@pytest.mark.asyncio
async def test_mock_crm_update_lead_status() -> None:
    from src.crm import MockCRMClient

    mock_client = MockCRMClient()
    result = await mock_client.update_lead_status(lead_id=123, status_id=142)
    assert result is True

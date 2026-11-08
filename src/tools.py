import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from typing import Any

from aiogram import Bot
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field

from src.crm import get_crm_client
from src.database import add_lead


class CreateBookingLeadArgs(BaseModel):
    client_name: str = Field(..., description="Имя клиента")
    phone: str = Field(..., description="Номер телефона клиента")
    service: str = Field(..., description="Название услуги или процедуры")
    preferred_time: str = Field(..., description="Желаемая дата и время записи")


class CheckAvailableSlotsArgs(BaseModel):
    date_str: str = Field(
        ...,
        description="Дата для проверки свободных слотов в формате YYYY-MM-DD или DD.MM.YYYY",
    )


TOOLS_SCHEMA: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "create_booking_lead",
            "description": "Записать клиента на прием и сохранить заявку в базу данных",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {
                        "type": "string",
                        "description": "Имя клиента",
                    },
                    "phone": {
                        "type": "string",
                        "description": "Номер телефона клиента",
                    },
                    "service": {
                        "type": "string",
                        "description": "Название услуги или процедуры",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "Желаемая дата и время записи",
                    },
                },
                "required": ["client_name", "phone", "service", "preferred_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Проверить свободные слоты времени для записи на выбранную дату",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {
                        "type": "string",
                        "description": "Дата для проверки свободных слотов в формате YYYY-MM-DD или DD.MM.YYYY",
                    },
                },
                "required": ["date_str"],
                "additionalProperties": False,
            },
        },
    },
]

# Слоты приёма по дням недели (0=пн … 6=вс)
_WEEKDAY_SLOTS: dict[int, list[str]] = {
    0: ["09:00", "11:00", "14:30", "16:00", "18:00"],  # Пн
    1: ["09:00", "11:00", "14:30", "16:00", "18:00"],  # Вт
    2: ["09:00", "11:00", "14:30", "16:00", "18:00"],  # Ср
    3: ["09:00", "11:00", "14:30", "16:00", "18:00"],  # Чт
    4: ["09:00", "11:00", "14:30", "16:00", "18:00"],  # Пт
    5: ["10:00", "12:00", "15:00", "17:00"],            # Сб
    6: ["10:00", "13:00", "17:00"],                     # Вс
}


def _parse_date(date_str: str, base_date: date | None = None) -> date | None:
    """Разбирает дату в календарных форматах YYYY-MM-DD / DD.MM.YYYY, а также относительные даты."""
    raw = date_str.strip().lower()
    today = base_date or date.today()

    # Относительные конструкции
    if raw in ("сегодня", "на сегодня"):
        return today
    if raw in ("завтра", "на завтра"):
        return today + timedelta(days=1)
    if raw in ("послезавтра", "на послезавтра"):
        return today + timedelta(days=2)

    # Дни недели
    ru_weekdays = {
        "понедельник": 0,
        "вторник": 1,
        "среду": 2,
        "среда": 2,
        "четверг": 3,
        "пятницу": 4,
        "пятница": 4,
        "субботу": 5,
        "суббота": 5,
        "воскресенье": 6,
    }

    # Проверка "в следующую пятницу", "на следующую пятницу", "в пятницу" и т.д.
    for day_name, target_weekday in ru_weekdays.items():
        if day_name in raw:
            days_ahead = (target_weekday - today.weekday()) % 7
            if "следующ" in raw:
                # В следующую неделю (если сегодня этот день или позже, +7 дней, иначе на след. неделе)
                days_ahead = days_ahead + 7 if days_ahead > 0 else 7
            elif days_ahead == 0:
                days_ahead = 7
            return today + timedelta(days=days_ahead)

    # Нормализация частых опечаток в разделителях даты (точки, запятые, слэши, буквы ю/б: '15ю11' -> '15.11')
    normalized = raw.replace("ю", ".").replace("б", ".").replace(",", ".").replace("/", "-")
    # Если замена слэша дала дефис, но формат вида ДД-ММ, нормализуем под стандартные форматы
    normalized = normalized.replace("..", ".").strip()

    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(normalized, fmt).date()
        except ValueError:
            continue

    # Если формат только день и месяц: ДД.ММ или ДД-ММ
    for sep in (".", "-"):
        parts = normalized.split(sep)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            try:
                day, month = int(parts[0]), int(parts[1])
                target_year = today.year
                candidate = date(target_year, month, day)
                if candidate < today:
                    candidate = date(target_year + 1, month, day)
                return candidate
            except ValueError:
                pass

    return None


async def _invoke_notify_lead(
    lead_data: dict[str, Any],
    bot: Bot | None = None,
    notify_func: Callable[..., Awaitable[None]] | None = None,
) -> None:
    if notify_func is not None:
        try:
            res = notify_func(lead_data, bot) if bot is not None else notify_func(lead_data)
            if inspect.isawaitable(res):
                await res
            return
        except TypeError:
            res = notify_func(lead_data)
            if inspect.isawaitable(res):
                await res
            return

    if bot is not None:
        from src.bot import notify_admins_about_lead
        await notify_admins_about_lead(lead_data, bot)


async def create_booking_lead(
    client_name: str,
    phone: str,
    service: str,
    preferred_time: str,
    bot: Bot | None = None,
    notify_func: Callable[..., Awaitable[None]] | None = None,
) -> str:
    crm_client = get_crm_client()
    crm_id = await crm_client.create_lead(
        name=client_name,
        phone=phone,
        service=service,
        slot_datetime=preferred_time,
    )
    lead_id = await add_lead(
        client_name=client_name,
        phone=phone,
        service=service,
        preferred_time=preferred_time,
        crm_lead_id=crm_id,
    )
    lead_data = {
        "lead_id": lead_id,
        "id": lead_id,
        "name": client_name,
        "client_name": client_name,
        "phone": phone,
        "service": service,
        "datetime_str": preferred_time,
        "preferred_time": preferred_time,
        "crm_lead_id": crm_id,
    }
    try:
        await _invoke_notify_lead(lead_data, bot=bot, notify_func=notify_func)
    except Exception as exc:
        logger.exception("Failed to send admin notification for lead #{}: {}", lead_id, exc)

    return f"Заявка успешно создана. Номер заявки: #{lead_id}."


async def check_available_slots(date_str: str) -> str:
    """Возвращает доступные слоты на указанную дату с базовой валидацией."""
    parsed = _parse_date(date_str)

    if parsed is None:
        return (
            f"Не удалось распознать дату «{date_str}». "
            "Пожалуйста, укажите дату в формате ДД.ММ.ГГГГ или ГГГГ-ММ-ДД."
        )

    today = date.today()
    if parsed < today:
        return f"Дата {date_str} уже прошла. Пожалуйста, укажите будущую дату для записи."

    slots = _WEEKDAY_SLOTS.get(parsed.weekday(), [])
    if not slots:
        return f"На {date_str} клиника не работает."

    slots_str = ", ".join(slots)
    return f"Свободные слоты на {date_str}: {slots_str}."


async def execute_tool(
    tool_name: str,
    arguments_json: str,
    bot: Bot | None = None,
    notify_func: Callable[..., Awaitable[None]] | None = None,
) -> str:
    logger.info("Executing tool '{}' with arguments: {}", tool_name, arguments_json)
    try:
        raw_args: dict[str, Any] = json.loads(arguments_json) if arguments_json else {}

        match tool_name:
            case "create_booking_lead":
                validated = CreateBookingLeadArgs.model_validate(raw_args)
                result = await create_booking_lead(
                    client_name=validated.client_name,
                    phone=validated.phone,
                    service=validated.service,
                    preferred_time=validated.preferred_time,
                    bot=bot,
                    notify_func=notify_func,
                )
                logger.info("Tool 'create_booking_lead' executed successfully: {}", result)
                return result

            case "check_available_slots":
                validated_slots = CheckAvailableSlotsArgs.model_validate(raw_args)
                result = await check_available_slots(date_str=validated_slots.date_str)
                logger.info("Tool 'check_available_slots' executed successfully: {}", result)
                return result

            case _:
                err_msg = f"Неизвестный инструмент: {tool_name}"
                logger.warning(err_msg)
                return err_msg

    except Exception as exc:
        err_msg = f"Ошибка выполнения инструмента {tool_name}: {exc}"
        logger.exception(err_msg)
        return err_msg

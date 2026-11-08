import io
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message
from loguru import logger

from src.agent import AIAgent
from src.config import settings
from src.crm import get_crm_client
from src.database import get_all_leads, get_booking_by_id, get_recent_leads, update_booking_status
from src.stt import SpeechToText


def _is_admin(user_id: int) -> bool:
    """Проверяет, есть ли user_id в списке администраторов из настроек."""
    return user_id in settings.ADMIN_USER_IDS


async def notify_admins_about_lead(lead_data: dict, bot: Bot) -> None:
    """Отправляет уведомление о новой записи всем администраторам из settings.ADMIN_USER_IDS."""
    name = lead_data.get("name") or lead_data.get("client_name", "")
    phone = lead_data.get("phone", "")
    service = lead_data.get("service", "")
    datetime_str = lead_data.get("datetime_str") or lead_data.get("preferred_time", "")
    lead_id = lead_data.get("lead_id") or lead_data.get("id", "")
    crm_lead_id = lead_data.get("crm_lead_id", 0) or 0

    message_text = (
        "🔔 Новая запись на приём!\n"
        f"👤 Пациент: {name} ({phone})\n"
        f"🩺 Услуга: {service}\n"
        f"📅 Время: {datetime_str}\n"
        f"🆔 Номер заявки: #{lead_id}"
    )

    reply_markup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Подтвердить запись",
                    callback_data=f"confirm_booking:{lead_id}:{crm_lead_id}",
                )
            ]
        ]
    )

    for admin_id in settings.ADMIN_USER_IDS:
        try:
            await bot.send_message(
                chat_id=admin_id,
                text=message_text,
                reply_markup=reply_markup,
            )
        except TelegramAPIError as exc:
            logger.warning(
                "Failed to send lead notification to admin user_id={}: {}",
                admin_id,
                exc,
            )


def build_leads_xlsx(leads: list[dict]) -> bytes:
    """Генерирует Excel-файл (.xlsx) с заявками в памяти."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Заявки"

    headers = ["ID", "Имя пациента", "Телефон", "Услуга", "Время записи", "Дата создания"]
    ws.append(headers)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="2A4B7C", end_color="2A4B7C", fill_type="solid")

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill

    # Закрепление верхней строки
    ws.freeze_panes = "A2"

    for row_idx, lead in enumerate(leads, start=2):
        lead_id = lead.get("id")
        lead_id_val = int(lead_id) if lead_id is not None and str(lead_id).isdigit() else lead_id

        name = lead.get("client_name") or "Не указано"
        phone = lead.get("phone") or lead.get("client_phone") or "Не указан"
        service = lead.get("service") or "Не указана"
        time_slot = lead.get("preferred_time") or lead.get("slot_datetime") or ""
        created_at = lead.get("created_at") or ""

        # Ячейка ID
        ws.cell(row=row_idx, column=1, value=lead_id_val)
        # Ячейка Имя
        ws.cell(row=row_idx, column=2, value=str(name))
        # Ячейка Телефон (явно строка + текстовый формат)
        c_phone = ws.cell(row=row_idx, column=3, value=str(phone))
        c_phone.number_format = "@"
        # Ячейка Услуга
        ws.cell(row=row_idx, column=4, value=str(service))
        # Ячейка Время записи
        ws.cell(row=row_idx, column=5, value=str(time_slot))
        # Ячейка Дата создания
        ws.cell(row=row_idx, column=6, value=str(created_at))

    # Автоподбор ширины столбцов по максимальной длине содержимого + отступ 3 символа
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            val_str = str(cell.value or "")
            if len(val_str) > max_len:
                max_len = len(val_str)
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def build_leads_csv(leads: list[dict]) -> bytes:
    """Для обратной совместимости."""
    return build_leads_xlsx(leads)


def create_router(agent: AIAgent, stt: SpeechToText | None = None) -> Router:
    """Создаёт и возвращает роутер с зарегистрированными хендлерами.

    Роутер создаётся внутри функции, а не на уровне модуля — это гарантирует
    что хендлеры не дублируются при повторных вызовах и агент передаётся
    явно, а не через closure глобальной переменной.
    """
    router = Router(name="bot_router")
    stt_service = stt or SpeechToText()

    @router.message(CommandStart())
    async def cmd_start(message: types.Message) -> None:
        user_name = message.from_user.first_name if message.from_user else "гость"
        welcome_text = (
            f"Здравствуйте, {user_name}! 👋\n"
            "Вас приветствует виртуальный администратор клиники **«Дента-Плюс»**.\n\n"
            "Я помогу вам сориентироваться по услугам, ценам и записаться на приём. Вот что вы можете у меня спросить:\n"
            "• 💰 **Цены и артикулы**: «Сколько стоит имплантация Nobel Biocare (арт. IMPL-01)?»\n"
            "• 💳 **Условия рассрочки**: «Есть ли рассрочка на лечение и на каких условиях?»\n"
            "• 🕒 **Свободные слоты**: «Какие слоты свободны на 15 ноября?»\n"
            "• 📅 **Запись на приём**: «Хочу записаться на чистку зубов на завтра к 14:00»\n\n"
            "Команды управления:\n"
            "/reset — начать диалог с чистого листа (сброс контекста)\n"
            "/leads — посмотреть последние 5 заявок в базе *(только для администраторов)*"
        )
        await message.answer(welcome_text, parse_mode="Markdown")

    @router.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id
        agent.clear_history(user_id)
        await message.answer("🔄 История диалога очищена. Чем я могу вам помочь?")

    @router.message(Command("leads"))
    async def cmd_leads(message: types.Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id

        if not _is_admin(user_id):
            logger.warning("Unauthorized /leads access attempt by user_id={}", user_id)
            await message.answer("⛔ Эта команда доступна только администраторам клиники.")
            return

        leads = await get_recent_leads(limit=5)
        if not leads:
            await message.answer("📋 Список заявок пока пуст.")
            return

        lines = ["📋 **Последние 5 заявок на приём:**\n"]
        for idx, lead in enumerate(leads, 1):
            created_at = lead.get("created_at", "—")
            lines.append(
                f"**{idx}. Заявка #{lead['id']}**\n"
                f"👤 Пациент: {lead['client_name']}\n"
                f"📞 Телефон: `{lead['phone']}`\n"
                f"🩺 Услуга: {lead['service']}\n"
                f"⏰ Желаемое время: {lead['preferred_time']}\n"
                f"📅 Создано: {created_at}\n"
            )

        await message.answer("\n".join(lines), parse_mode="Markdown")

    @router.message(Command("export"))
    async def cmd_export(message: types.Message) -> None:
        if not message.from_user or message.from_user.id not in settings.ADMIN_USER_IDS:
            await message.answer("Эта команда доступна только администраторам клиники.")
            return

        leads = await get_all_leads()
        if not leads:
            await message.answer("В базе пока нет заявок.")
            return

        xlsx_bytes = build_leads_xlsx(leads)

        document = BufferedInputFile(xlsx_bytes, filename="leads_export.xlsx")
        await message.answer_document(document=document)

    async def _process_and_reply(target_message: Message, user_text: str, bot: Bot) -> None:
        user_id = target_message.from_user.id if target_message.from_user else target_message.chat.id
        await bot.send_chat_action(chat_id=target_message.chat.id, action=ChatAction.TYPING)

        try:
            response = await agent.process_message(
                user_id=user_id,
                user_text=user_text,
                bot=bot,
            )
            response_text = getattr(response, "text", str(response))
            reply_markup = getattr(response, "reply_markup", None)

            try:
                await target_message.answer(
                    response_text,
                    reply_markup=reply_markup,
                    parse_mode="Markdown",
                )
            except TelegramBadRequest as parse_exc:
                logger.warning(
                    "Markdown parsing failed for user_id={}: {}. Falling back to plain text.",
                    user_id,
                    parse_exc,
                )
                await target_message.answer(response_text, reply_markup=reply_markup)
        except Exception as exc:
            logger.error("Error processing message for user_id={}: {}", user_id, exc)
            await target_message.answer(
                "Извините, произошла техническая ошибка при обработке запроса. "
                "Пожалуйста, попробуйте снова через несколько минут."
            )

    @router.message(F.voice)
    async def handle_voice_message(message: Message, bot: Bot) -> None:
        if not message.voice:
            return

        await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

        voice_buffer = io.BytesIO()
        await bot.download(message.voice, destination=voice_buffer)
        audio_bytes = voice_buffer.getvalue()
        logger.info("Received voice message buffer size: {} bytes", len(audio_bytes))

        text = await stt_service.transcribe_audio(audio_bytes)

        if not text:
            await message.answer(
                "Не удалось разобрать голосовое сообщение. Пожалуйста, повторите чётче или напишите текстом."
            )
            return

        user_id = message.from_user.id if message.from_user else message.chat.id
        logger.info(f"Transcribed voice from {user_id}: '{text}'")

        await _process_and_reply(target_message=message, user_text=text, bot=bot)

    @router.message(F.text)
    async def handle_message(message: Message, bot: Bot) -> None:
        if not message.text:
            return
        await _process_and_reply(target_message=message, user_text=message.text, bot=bot)

    @router.callback_query(lambda c: c.data and c.data.startswith("select_slot:"))
    async def handle_select_slot(callback: types.CallbackQuery) -> None:
        await callback.answer()
        if not callback.data:
            return

        time_slot = callback.data.removeprefix("select_slot:")
        text = (
            f"Вы выбрали время {time_slot}. "
            "Чтобы завершить запись, напишите ваше имя, телефон и услугу "
            "(или просто подтвердите имя и телефон)."
        )

        if callback.message:
            await callback.message.answer(text)

    @router.callback_query(lambda c: c.data and c.data.startswith("confirm_booking:"))
    async def handle_confirm_booking(callback: types.CallbackQuery, bot: Bot) -> None:
        if not callback.data:
            return

        parts = callback.data.split(":")
        if len(parts) != 3:
            return

        booking_id_str, crm_lead_id_str = parts[1], parts[2]
        booking_id = int(booking_id_str)
        crm_lead_id = int(crm_lead_id_str)

        crm_client = get_crm_client()
        if crm_lead_id > 0:
            await crm_client.update_lead_status(int(crm_lead_id), settings.AMO_CRM_STATUS_CONFIRMED_ID)

        await update_booking_status(booking_id, "confirmed")

        if callback.message and isinstance(callback.message, Message):
            current_text = callback.message.text or ""
            new_text = current_text + "\n\n✅ Статус: Подтверждено администратором (amoCRM обновлена)"
            try:
                await callback.message.edit_text(new_text, reply_markup=None)
            except TelegramBadRequest as exc:
                logger.warning("Failed to edit admin message on confirm_booking: {}", exc)

        booking_info = await get_booking_by_id(booking_id)
        patient_tg_id = None
        if booking_info:
            patient_tg_id = (
                booking_info.get("user_id")
                or booking_info.get("telegram_id")
                or booking_info.get("chat_id")
            )

        if patient_tg_id:
            try:
                await bot.send_message(
                    chat_id=int(patient_tg_id),
                    text="Ваша запись подтверждена администратором клиники! Ждём вас.",
                )
            except TelegramAPIError as exc:
                logger.warning("Failed to notify patient about confirmation: {}", exc)

        await callback.answer("Запись подтверждена")

    return router


def create_dispatcher(agent: AIAgent, stt: SpeechToText | None = None) -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(create_router(agent, stt=stt))
    return dp

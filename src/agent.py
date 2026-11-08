from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
import httpx
from loguru import logger
import openai
from openai import AsyncOpenAI, APIError, APITimeoutError, APIConnectionError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.rag import KnowledgeBaseRAG, RAGContextResult
from src.tools import TOOLS_SCHEMA, execute_tool

SYSTEM_PROMPT_TEMPLATE = """Ты — вежливый, внимательный администратор стоматологической клиники «Дента-Плюс». Твоя задача — консультировать пациентов, помогать с выбором услуг и записывать на приём.

{current_datetime_info}
При вызове инструментов `check_available_slots` и `create_booking_lead` всегда переводи любые относительные даты пользователя ('завтра', 'послезавтра', 'в эту пятницу', 'на следующей неделе') в точный абсолютный календарный формат 'YYYY-MM-DD' строго относительно текущей даты сервера.

ПРАВИЛА ПРОВЕРКИ И ОБРАБОТКИ ДАТ ПРИ БРОНИРОВАНИИ:
- При бронировании дата визита считается прошедшей ТОЛЬКО ЕСЛИ она строго меньше текущей даты сервера.
- Любая дата позже текущей (например, если сегодня 9 ноября, то 10, 15, 20 ноября — это БУДУЩИЕ даты) является корректной и должна приниматься без отказа.
- Обязательно учитывай возможные опечатки пользователя в разделителях даты (точки, запятые, слэши, буквы ю/б: например, '15ю11' или '15,11' или '15/11' -> '15.11') и корректно определяй дату (год, месяц, число) относительно текущей даты сервера.

ПРАВИЛА ОБЩЕНИЯ:
1. Обращайся к пациенту исключительно на «вы», вежливо и профессионально.
2. Отвечай кратко, емко и по существу, без лишней «воды».
3. Опирайся ТОЛЬКО на предоставленный ниже блок <context>. Не придумывай услуги, цены, скидки или врачей, которых нет в контексте.
4. Отвечай строго на русском языке, не используй иностранные слова и фразы вежливости на английском.

ЕСЛИ ИНФОРМАЦИИ НЕТ В КОНТЕКСТЕ:
- Ни в коем случае не говори клиенту «уточни у администратора».
- Если вопрос не относится к стоматологии (животные, посторонние услуги), вежливо скажи: «Мы специализированная стоматологическая клиника и оказываем помощь только людям. К сожалению, подобные услуги мы не предоставляем».
- Если вопрос стоматологический, но точной информации или цены нет в контексте, ответь от первого лица: «К сожалению, точная информация по этому вопросу в базе отсутствует. Я передам ваш запрос старшему администратору для уточнения, либо вы можете связаться с нами по телефону клиники».

ЗАПИСЬ НА ПРИЁМ (TOOLS):
- Для проверки окон используй `check_available_slots`.
- Если клиент хочет записаться, обязательно уточни: ФИО/имя, номер телефона, нужную услугу и дату/время. Только после получения всех 4 параметров вызывай инструмент `create_booking_lead`.
- После успешного вызова инструмента поблагодари за запись и назови номер заявки.

<context>
{context}
</context>"""


FALLBACK_OVERLOAD_MESSAGE = (
    "К сожалению, сервис клиники временно перегружен. "
    "Я передал ваш запрос дежурному администратору, с вами свяжутся в течение нескольких минут."
)

FALLBACK_PROMPT_DIRECTIVE = """
ВАЖНО (РЕЖИМ FALLBACK): Релевантная информация в базе знаний клиники не найдена или недоступна. 
Если точной информации нет в контексте, категорически запрещено выдумывать услуги и цены — предлагай консультацию терапевта."""


@dataclass(slots=True)
class AgentResponse:
    text: str
    reply_markup: InlineKeyboardMarkup | None = None

    def __str__(self) -> str:
        return self.text

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, AgentResponse):
            return self.text == other.text and self.reply_markup == other.reply_markup
        return False


def build_slots_keyboard(slots: list[str], columns: int = 2) -> InlineKeyboardMarkup:
    """Формирует InlineKeyboardMarkup со слотами времени в 2-3 колонки."""
    builder = InlineKeyboardBuilder()
    for time_slot in slots:
        builder.button(
            text=time_slot,
            callback_data=f"select_slot:{time_slot}",
        )
    builder.adjust(columns)
    return builder.as_markup()


def extract_slots_from_result(result_text: str) -> list[str]:
    """Извлекает доступные часы (например, HH:MM) из результата инструмента check_available_slots."""
    if "Свободные слоты" not in result_text:
        return []
    return re.findall(r"\b\d{2}:\d{2}\b", result_text)


class AIAgent:
    def __init__(
        self,
        rag: KnowledgeBaseRAG | None = None,
        bot: Bot | None = None,
        notify_func: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
        )
        self.rag = rag or KnowledgeBaseRAG()
        self.bot = bot
        self.notify_func = notify_func
        self.history: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._history_limit = settings.HISTORY_LIMIT
        self._max_tool_turns = settings.MAX_TOOL_TURNS

    def clear_history(self, user_id: int) -> None:
        if user_id in self.history:
            self.history.pop(user_id, None)
            logger.info("Chat history cleared for user_id={}", user_id)

    def _trim_history(self, user_id: int) -> None:
        """Усекает историю пользователя до скользящего окна `_history_limit`."""
        history = self.history[user_id]
        if len(history) > self._history_limit:
            self.history[user_id] = history[-self._history_limit:]

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, openai.APITimeoutError, openai.APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        reraise=True,
    )
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        with_tools: bool = True,
    ) -> Any:
        """Единая точка вызова LLM с опциональным набором инструментов."""
        kwargs: dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "messages": messages,
        }
        if with_tools:
            kwargs["tools"] = TOOLS_SCHEMA
            kwargs["tool_choice"] = "auto"
        return await self.client.chat.completions.create(**kwargs)

    async def process_message(
        self,
        user_id: int,
        user_text: str,
        bot: Bot | None = None,
    ) -> AgentResponse:
        active_bot = bot or self.bot
        # Шаг 1: получить релевантный контекст из базы знаний
        rag_result = await self.rag.retrieve_context(user_text)
        if isinstance(rag_result, RAGContextResult):
            context = rag_result.context
            fallback_mode = rag_result.fallback_mode
        else:
            context = str(rag_result)
            fallback_mode = False

        # Шаг 2: сформировать системный промпт с контекстом и актуальной датой
        now = datetime.now()
        current_datetime_info = f"Текущая дата и время сервера: {now.strftime('%Y-%m-%d %H:%M, %A')}."
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            current_datetime_info=current_datetime_info,
            current_date=now.strftime("%Y-%m-%d (%A)"),
            context=context,
        )
        if fallback_mode:
            system_prompt += f"\n\n{FALLBACK_PROMPT_DIRECTIVE}"

        # Добавить сообщение пользователя в скользящее окно истории
        self.history[user_id].append({"role": "user", "content": user_text})
        self._trim_history(user_id)

        # Формируем локальную цепочку для текущего запроса:
        # system + долговременная история (только user/assistant)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *self.history[user_id],
        ]

        final_text = ""
        available_slots: list[str] = []

        try:
            for turn in range(self._max_tool_turns):
                try:
                    response = await self._call_llm(messages, with_tools=True)
                except APIError as exc:
                    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)) or (
                        isinstance(getattr(exc, '__cause__', None), (httpx.TimeoutException, httpx.ConnectError))
                    ):
                        raise
                    # Провайдер не поддерживает tools — пробуем plain-text режим
                    logger.warning(
                        "LLM tool-calling failed for user_id={} (turn {}), falling back to plain text: {}",
                        user_id,
                        turn,
                        exc,
                    )
                    try:
                        fallback = await self._call_llm(messages, with_tools=False)
                        final_text = fallback.choices[0].message.content or ""
                    except APIError as fallback_exc:
                        logger.error(
                            "Plain-text fallback also failed for user_id={}: {}",
                            user_id,
                            fallback_exc,
                        )
                        final_text = (
                            "Извините, сервис временно недоступен. "
                            "Пожалуйста, попробуйте позже."
                        )
                    break

                message = response.choices[0].message

                # Нет tool_calls — модель вернула финальный текстовый ответ
                if not message.tool_calls:
                    final_text = message.content or ""
                    break

                # Добавляем ответ ассистента с tool_calls в локальный контекст
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                # Выполняем все инструменты параллельно и добавляем результаты
                for tool_call in message.tool_calls:
                    tool_result = await execute_tool(
                        tool_name=tool_call.function.name,
                        arguments_json=tool_call.function.arguments,
                        bot=active_bot,
                        notify_func=self.notify_func,
                    )
                    if tool_call.function.name == "check_available_slots":
                        extracted = extract_slots_from_result(str(tool_result))
                        if extracted:
                            available_slots = extracted

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result),
                        }
                    )

                # Достигли лимита итераций — принудительно получаем текстовый ответ
                if turn == self._max_tool_turns - 1:
                    logger.warning(
                        "Max tool turns ({}) reached for user_id={}, forcing text response",
                        self._max_tool_turns,
                        user_id,
                    )
                    try:
                        final_resp = await self._call_llm(messages, with_tools=False)
                        final_text = final_resp.choices[0].message.content or ""
                    except APIError as exc:
                        logger.error("Final forced LLM call failed for user_id={}: {}", user_id, exc)
                        final_text = "Извините, не удалось сформировать ответ. Попробуйте позже."
                    break
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            logger.critical("LLM service unavailable or overloaded after retries for user_id={}: {}", user_id, exc)
            final_text = FALLBACK_OVERLOAD_MESSAGE

        # Сохраняем только финальный текстовый ответ ассистента в долговременную историю
        self.history[user_id].append({"role": "assistant", "content": final_text})
        self._trim_history(user_id)

        reply_markup = build_slots_keyboard(available_slots) if available_slots else None
        return AgentResponse(text=final_text, reply_markup=reply_markup)

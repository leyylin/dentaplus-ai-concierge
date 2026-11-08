from pathlib import Path
import aiosqlite
from src.config import settings


async def init_db() -> None:
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                service TEXT NOT NULL,
                preferred_time TEXT NOT NULL,
                crm_lead_id INTEGER NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor = await db.execute("PRAGMA table_info(leads)")
        columns = {row[1] for row in await cursor.fetchall()}
        
        if "crm_lead_id" not in columns:
            try:
                await db.execute("ALTER TABLE leads ADD COLUMN crm_lead_id INTEGER NULL")
            except aiosqlite.OperationalError:
                pass

        if "status" not in columns:
            try:
                await db.execute("ALTER TABLE leads ADD COLUMN status TEXT DEFAULT 'new'")
            except aiosqlite.OperationalError:
                pass

        # Добавляем колонки для обратной совместимости, если они отсутствуют
        if "client_phone" not in columns:
            try:
                await db.execute("ALTER TABLE leads ADD COLUMN client_phone TEXT GENERATED ALWAYS AS (phone) VIRTUAL")
            except aiosqlite.OperationalError:
                pass

        if "slot_datetime" not in columns:
            try:
                await db.execute("ALTER TABLE leads ADD COLUMN slot_datetime TEXT GENERATED ALWAYS AS (preferred_time) VIRTUAL")
            except aiosqlite.OperationalError:
                pass

        await db.commit()


async def add_lead(
    client_name: str,
    phone: str,
    service: str,
    preferred_time: str,
    crm_lead_id: int | None = None,
) -> int:
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO leads (client_name, phone, service, preferred_time, crm_lead_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (client_name, phone, service, preferred_time, crm_lead_id),
        )
        await db.commit()
        return cursor.lastrowid  # type: ignore[return-value]


async def get_recent_leads(limit: int = 5) -> list[dict]:
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute(
                """
                SELECT id, client_name, phone, preferred_time, service, crm_lead_id, created_at,
                       client_phone, slot_datetime
                FROM leads
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.OperationalError:
            async with db.execute(
                """
                SELECT id, client_name, phone, preferred_time, service, NULL AS crm_lead_id, created_at,
                       phone AS client_phone, preferred_time AS slot_datetime
                FROM leads
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]


async def get_all_leads() -> list[dict]:
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            async with db.execute(
                """
                SELECT id, client_name, phone, preferred_time, service, crm_lead_id, created_at,
                       client_phone, slot_datetime
                FROM leads
                ORDER BY id DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.OperationalError:
            async with db.execute(
                """
                SELECT id, client_name, phone, preferred_time, service, NULL AS crm_lead_id, created_at,
                       phone AS client_phone, preferred_time AS slot_datetime
                FROM leads
                ORDER BY id DESC
                """
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]


async def update_booking_status(booking_id: int, status: str = "confirmed") -> None:
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        await db.execute(
            """
            UPDATE leads
            SET status = ?
            WHERE id = ?
            """,
            (status, booking_id),
        )
        await db.commit()


async def get_booking_by_id(booking_id: int) -> dict | None:
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM leads WHERE id = ?
            """,
            (booking_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None



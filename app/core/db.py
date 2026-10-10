"""Postgres storage (Neon or any Postgres) for chat messages and photos.

Used only when DATABASE_URL is set. Ticket state itself lives in LangGraph's own Postgres
checkpointer tables, created by `PostgresSaver.setup()`. Without DATABASE_URL the app keeps
using SQLite and in-memory stores, so local runs and the tests need no database.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    ts          TEXT NOT NULL,
    sender      TEXT NOT NULL,
    recipient   TEXT NOT NULL,
    conversation TEXT NOT NULL,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    ticket_id   TEXT,
    actions     JSONB NOT NULL DEFAULT '[]',
    attachments JSONB NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation, id);
CREATE TABLE IF NOT EXISTS photos (
    ticket_id  TEXT PRIMARY KEY,
    mime       TEXT NOT NULL,
    data       BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def open_pool(url: str):
    """A small connection pool that survives Neon suspending an idle database: connections are
    checked before use and recycled when idle, and a first request after a pause just waits for
    the database to wake."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        url,
        min_size=1,
        max_size=5,
        max_idle=60,
        check=ConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool


class Database:
    def __init__(self, pool):
        self._pool = pool

    def ensure_schema(self) -> None:
        with self._pool.connection() as conn:
            for statement in SCHEMA.split(";"):  # one statement per call: prepared statements allow one
                if statement.strip():
                    conn.execute(statement)

    # --- chat messages ----------------------------------------------------
    def load_messages(self) -> List[Dict]:
        with self._pool.connection() as conn:
            return list(conn.execute("SELECT * FROM messages ORDER BY id"))

    def add_message(self, m: Dict) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO messages (id, ts, sender, recipient, conversation, subject, body,
                                         ticket_id, actions, attachments)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO NOTHING""",
                (m["id"], m["ts"], m["sender"], m["recipient"], m["conversation"], m["subject"],
                 m["body"], m["ticket_id"], json.dumps(m["actions"]), json.dumps(m["attachments"])),
            )

    # --- photos -------------------------------------------------------------
    def put_photo(self, ticket_id: str, data: bytes, mime: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """INSERT INTO photos (ticket_id, mime, data) VALUES (%s,%s,%s)
                   ON CONFLICT (ticket_id) DO UPDATE SET mime = EXCLUDED.mime, data = EXCLUDED.data""",
                (ticket_id, mime, data),
            )

    def get_photo(self, ticket_id: str) -> Optional[Tuple[bytes, str]]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT data, mime FROM photos WHERE ticket_id = %s", (ticket_id,)).fetchone()
        return (bytes(row["data"]), row["mime"]) if row else None

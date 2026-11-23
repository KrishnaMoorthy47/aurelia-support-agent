"""Audit log of API requests and responses (Postgres table `api_logs`, or in memory for tests)."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

from app.rag.stores import SCHEMA_SQL

log = logging.getLogger("aurelia.audit")

MAX_BODY_CHARS = 50_000
COLUMNS = ("request_id", "method", "path", "status_code", "latency_ms", "session_id", "customer_id", "route", "request", "response")


def body_to_json(raw: bytes) -> Any:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    if len(text) > MAX_BODY_CHARS:
        return {"truncated": text[:MAX_BODY_CHARS]}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


class MemoryAuditLog:
    def __init__(self, size: int = 500) -> None:
        self.records: deque[dict] = deque(maxlen=size)

    def write(self, record: dict) -> None:
        self.records.append(record)

    def close(self) -> None:
        pass


class PostgresAuditLog:
    def __init__(self, dsn: str) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=2, open=True, check=ConnectionPool.check_connection,
            kwargs={"prepare_threshold": None, "autocommit": True},
        )
        with self._pool.connection() as c:
            c.execute(SCHEMA_SQL.read_text(encoding="utf-8"))

    def write(self, record: dict) -> None:
        from psycopg.types.json import Jsonb

        values = [Jsonb(record.get(k)) if k in ("request", "response") else record.get(k) for k in COLUMNS]
        with self._pool.connection() as c:
            c.execute(f"insert into api_logs ({', '.join(COLUMNS)}) values ({', '.join(['%s'] * len(COLUMNS))})", values)

    def close(self) -> None:
        self._pool.close()


class AuditWriter:
    """Writes in the background so logging never delays a reply or breaks a request."""

    def __init__(self, sink: MemoryAuditLog | PostgresAuditLog | None) -> None:
        self.sink = sink
        self._tasks: set[asyncio.Task] = set()

    def submit(self, record: dict) -> None:
        if self.sink is None:
            return
        task = asyncio.create_task(self._write(record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _write(self, record: dict) -> None:
        try:
            await asyncio.to_thread(self.sink.write, record)
        except Exception as exc:
            log.warning("audit log write failed: %s", type(exc).__name__)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


def make_sink(mode: str, dsn: str):
    if mode == "auto":
        mode = "postgres" if dsn else "off"
    if mode == "postgres":
        if not dsn:
            raise RuntimeError("AUDIT_LOG=postgres needs DATABASE_URL")
        return PostgresAuditLog(dsn)
    if mode == "memory":
        return MemoryAuditLog()
    return None

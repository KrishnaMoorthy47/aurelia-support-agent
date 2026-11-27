"""Audit log of API requests and responses (Postgres table `api_logs`, or in memory for tests)."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Any

from app.rag.stores import SCHEMA_SQL, open_pool

log = logging.getLogger("aurelia.audit")

MAX_BODY_CHARS = 50_000
MAX_GET_BODY_CHARS = 20_000  # successful GETs over this (the page itself, the eval report) are logged by size only
COLUMNS = ("request_id", "started_at", "method", "path", "query", "status_code", "latency_ms", "session_id", "customer_id",
           "route", "user_agent", "request", "response")


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


def response_to_json(raw: bytes, content_type: str, method: str, status: int) -> Any:
    """JSON responses are stored as JSON; other content (HTML, files) and large successful GETs only by type and size."""
    if not raw:
        return None
    if "json" not in (content_type or ""):
        return {"content_type": content_type, "bytes": len(raw)}
    if method == "GET" and status < 400 and len(raw) > MAX_GET_BODY_CHARS:
        return {"omitted": "large response", "bytes": len(raw)}
    return body_to_json(raw)


class MemoryAuditLog:
    def __init__(self, size: int = 500) -> None:
        self.records: deque[dict] = deque(maxlen=size)

    def write(self, record: dict) -> None:
        self.records.append(record)

    def close(self) -> None:
        pass


class PostgresAuditLog:
    def __init__(self, dsn: str) -> None:
        self._pool = open_pool(dsn, max_size=2)
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

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for pending writes. Polls rather than awaiting the tasks, so it also works when called from
        another event loop (the test client runs the app on its own loop in a separate thread)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while any(not t.done() for t in list(self._tasks)) and loop.time() < deadline:
            await asyncio.sleep(0.01)


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

"""Vector stores: in-memory and pgvector."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.chunking import Chunk

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "supabase" / "schema.sql"


class VectorStore(Protocol):
    def count(self, embedder: str) -> int: ...

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, embedder: str) -> None: ...

    def load_chunks(self, embedder: str) -> list[Chunk]: ...

    def search(self, vector: np.ndarray, k: int, embedder: str) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_similarity)] best first."""
        ...


class MemoryStore:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}
        self._embedder: dict[str, str] = {}

    def count(self, embedder: str) -> int:
        return sum(1 for e in self._embedder.values() if e == embedder)

    def upsert(self, chunks, vectors, embedder):
        for c, v in zip(chunks, vectors):
            self._chunks[c.id], self._vecs[c.id], self._embedder[c.id] = c, v, embedder

    def load_chunks(self, embedder):
        return [c for cid, c in self._chunks.items() if self._embedder[cid] == embedder]

    def search(self, vector, k, embedder):
        ids = [cid for cid, e in self._embedder.items() if e == embedder]
        if not ids:
            return []
        mat = np.stack([self._vecs[i] for i in ids])
        sims = mat @ vector
        order = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in order]


def open_pool(dsn: str, max_size: int):
    """Connection pool that fails fast on bad credentials.

    One direct connection attempt first: a pool would keep retrying a wrong password in the background,
    and Supabase's pooler blocks the client after repeated authentication failures.
    prepare_threshold=None: Supabase's transaction pooler rejects prepared statements.
    """
    import psycopg
    from psycopg_pool import ConnectionPool

    psycopg.connect(dsn, connect_timeout=10, prepare_threshold=None).close()
    return ConnectionPool(
        dsn, min_size=1, max_size=max_size, open=True, check=ConnectionPool.check_connection,
        kwargs={"prepare_threshold": None, "autocommit": True},
    )


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


class PgVectorStore:
    """pgvector on Supabase (or any Postgres with the `vector` extension).

    Creates its own table on first connect (supabase/schema.sql, all `if not exists`), so a fresh
    Supabase project needs no manual SQL. Uses plain psycopg and sends vectors as text literals,
    so no extra adapter package is needed.
    """

    def __init__(self, dsn: str, ensure_schema: bool = True) -> None:
        # A pool keeps connections open: a fresh TLS connection to a remote database costs ~1-2s per query.
        self._pool = open_pool(dsn, max_size=4)
        if ensure_schema:
            self.ensure_schema()

    def ensure_schema(self) -> None:
        """Idempotent: safe to run on every start."""
        with self._conn() as c:
            c.execute(SCHEMA_SQL.read_text(encoding="utf-8"))

    def _conn(self):
        return self._pool.connection()

    def close(self) -> None:
        self._pool.close()

    def count(self, embedder: str) -> int:
        with self._conn() as c:
            return c.execute("select count(*) from kb_chunks where embedder = %s", (embedder,)).fetchone()[0]

    def upsert(self, chunks, vectors, embedder):
        with self._conn() as c:
            for ch, v in zip(chunks, vectors):
                c.execute(
                    """
                    insert into kb_chunks (id, doc, title, section, content, embedder, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s::vector)
                    on conflict (id) do update set
                      doc = excluded.doc, title = excluded.title, section = excluded.section,
                      content = excluded.content, embedder = excluded.embedder, embedding = excluded.embedding
                    """,
                    (ch.id, ch.doc, ch.title, ch.section, ch.content, embedder, _vec_literal(v)),
                )

    def load_chunks(self, embedder):
        with self._conn() as c:
            rows = c.execute(
                "select id, doc, title, section, content from kb_chunks where embedder = %s order by id", (embedder,)
            ).fetchall()
        return [Chunk(*r) for r in rows]

    def search(self, vector, k, embedder):
        lit = _vec_literal(vector)
        with self._conn() as c:
            rows = c.execute(
                """
                select id, 1 - (embedding <=> %s::vector) as sim
                from kb_chunks where embedder = %s
                order by embedding <=> %s::vector limit %s
                """,
                (lit, embedder, lit, k),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

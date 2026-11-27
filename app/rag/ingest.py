"""Embed the knowledge base and write it to the configured vector store.

    python -m app.rag.ingest            # uses VECTOR_STORE / DATABASE_URL from .env
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.rag.chunking import load_chunks
from app.rag.embedders import Embedder, get_embedder
from app.rag.retriever import HybridRetriever
from app.rag.stores import MemoryStore, PgVectorStore, VectorStore

log = logging.getLogger("aurelia.ingest")


def make_store() -> VectorStore:
    s = get_settings()
    if s.vector_store == "pgvector":
        if not s.database_url:
            raise RuntimeError("VECTOR_STORE=pgvector needs DATABASE_URL (your Supabase connection string)")
        return PgVectorStore(s.database_url)
    return MemoryStore()


def ingest(store: VectorStore, embedder: Embedder) -> int:
    chunks = load_chunks()
    store.upsert(chunks, embedder.embed([c.text_for_embedding for c in chunks]), embedder.name)
    log.info("ingested %d chunks with %s", len(chunks), embedder.name)
    return len(chunks)


def sync(store: VectorStore, embedder: Embedder) -> tuple[int, int]:
    """Bring the store in line with the markdown files: embed new or edited sections, drop removed ones.

    Returns (embedded, deleted). Unchanged sections are not re-embedded.
    """
    stored = {c.id: c for c in store.load_chunks(embedder.name)}
    current = load_chunks()
    changed = [c for c in current if stored.get(c.id) != c]
    stale = sorted(set(stored) - {c.id for c in current})
    if changed:
        store.upsert(changed, embedder.embed([c.text_for_embedding for c in changed]), embedder.name)
    if stale:
        store.delete(stale, embedder.name)
    if changed or stale:
        log.info("knowledge base sync: %d embedded, %d removed", len(changed), len(stale))
    return len(changed), len(stale)


def build_retriever(store: VectorStore | None = None, embedder: Embedder | None = None) -> HybridRetriever:
    """Used by the API at startup. Embeds any section that is new or edited since the last run."""
    store = store or make_store()
    embedder = embedder or get_embedder()
    sync(store, embedder)
    return HybridRetriever(store, embedder)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = ingest(make_store(), get_embedder())
    print(f"Ingested {n} chunks.")

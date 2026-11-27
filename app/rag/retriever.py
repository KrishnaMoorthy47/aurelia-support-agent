"""Multi-query hybrid retrieval.

RRF over: vector search on the original question, vector search on the router's rewrite, and
BM25 on the rewrite. Without a rewrite it uses vector + BM25 on the original question.
See eval/retrieval_bench.py for the comparison between variants.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.rag.chunking import Chunk
from app.rag.embedders import Embedder, terms
from app.rag.stores import VectorStore

RRF_K = 60
BM25_GATE = 1.5  # only used with the offline hash embedder


@dataclass
class Hit:
    chunk: Chunk
    rrf: float
    vector_score: float | None
    bm25_rank: int | None
    vector_rank: int | None


@dataclass
class Retrieval:
    hits: list[Hit]
    best_vector_score: float
    confident: bool


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store, self.embedder = store, embedder
        self._chunks = store.load_chunks(embedder.name)
        self._by_id = {c.id: c for c in self._chunks}
        self._bm25 = BM25Okapi([terms(c.text_for_embedding) for c in self._chunks]) if self._chunks else None

    def _vector(self, query: str, pool: int) -> list[tuple[str, float]]:
        return self.store.search(self.embedder.embed([query])[0], pool, self.embedder.name)

    def _bm25_ranked(self, query: str, pool: int) -> list[str]:
        scores = self._bm25.get_scores(terms(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
        return [self._chunks[i].id for i in order if scores[i] > 0]

    def search(self, query: str, rewrite: str | None = None, k: int = 3, pool: int = 8) -> Retrieval:
        if not self._chunks:
            return Retrieval([], 0.0, False)
        rewrite = (rewrite or "").strip()
        use_rewrite = bool(rewrite) and rewrite.lower() != query.strip().lower()

        vec_orig = self._vector(query, pool)
        vec_lists = [vec_orig] + ([self._vector(rewrite, pool)] if use_rewrite else [])
        bm25 = self._bm25_ranked(rewrite if use_rewrite else query, pool)

        fused: dict[str, float] = {}
        for ranked in [[cid for cid, _ in v] for v in vec_lists] + [bm25]:
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        best_cos: dict[str, float] = {}
        for v in vec_lists:
            for cid, s in v:
                best_cos[cid] = max(best_cos.get(cid, -1.0), s)
        vec_rank = {cid: r for r, (cid, _) in enumerate(vec_orig)}
        bm_rank = {cid: r for r, cid in enumerate(bm25)}

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        hits = [Hit(self._by_id[c], fused[c], best_cos.get(c), bm_rank.get(c), vec_rank.get(c)) for c in top]
        best = float(max(best_cos.values())) if best_cos else 0.0
        # With a real embedding model, gate on semantic similarity only: BM25 scores on short chunks are high
        # for any shared word ("match", "account"), which let unrelated questions through.
        confident = best >= self.embedder.min_score
        if not self.embedder.semantic:
            q_terms = set(terms(rewrite if use_rewrite else query))
            scores = self._bm25.get_scores(list(q_terms))
            # The keyword stand-in also needs two shared words when the question has two, so one
            # coincidental word ("capital" in "Capital Market Authority") is not enough.
            overlap = max((len(q_terms & set(terms(h.chunk.text_for_embedding))) for h in hits), default=0)
            confident = (confident or float(max(scores)) >= BM25_GATE) and overlap >= min(2, len(q_terms))
        return Retrieval(hits, best, confident)

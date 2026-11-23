"""Compare retrieval strategies on labelled queries.

    python eval/retrieval_bench.py              # no LLM calls: BM25, vector, hybrid on the raw question
    python eval/retrieval_bench.py --rewrite    # also the production path: router rewrite + multi-query RRF
                                                # (one small-model call per query)

Metrics: hit@1, hit@3 (the app sends the top 3 chunks to the answer model) and MRR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from app.rag.embedders import FastEmbedEmbedder, LexicalEmbedder  # noqa: E402
from app.rag.ingest import build_retriever  # noqa: E402
from app.rag.stores import MemoryStore  # noqa: E402

GROUPS = ("keyword", "paraphrase")


def first_hit(order: list[str], expect: list[str]) -> int | None:
    return next((i for i, cid in enumerate(order) if cid in expect), None)


async def rewrite_all(cases: dict) -> dict[str, str]:
    from app.agents import prompts
    from app.llm import get_llm

    llm, out = get_llm(), {}
    for group in (*GROUPS, "off_topic"):
        for c in cases[group]:
            raw = await llm.complete("route", prompts.ROUTER_SYSTEM, f"MESSAGE: {c['q']}")
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                out[c["q"]] = (json.loads(m.group(0)).get("search_query") or "") if m else ""
            except json.JSONDecodeError:
                out[c["q"]] = ""
            await asyncio.sleep(1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical", action="store_true", help="offline hash embedder instead of bge-small")
    ap.add_argument("--rewrite", action="store_true", help="include the router's query rewrite (uses the LLM)")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    embedder = LexicalEmbedder() if args.lexical else FastEmbedEmbedder()
    r = build_retriever(MemoryStore(), embedder)
    cases = yaml.safe_load((ROOT / "eval" / "retrieval_cases.yaml").read_text(encoding="utf-8"))
    rewrites = asyncio.run(rewrite_all(cases)) if args.rewrite else {}
    n_chunks = len(r._chunks)
    print(f"embedder {embedder.name}, {n_chunks} chunks\n")

    def strategies(q: str) -> dict[str, list[str]]:
        out = {
            "bm25": r._bm25_ranked(q, n_chunks),
            "vector": [cid for cid, _ in r._vector(q, n_chunks)],
            "hybrid rrf, no rewrite": [h.chunk.id for h in r.search(q, k=n_chunks).hits],
        }
        if args.rewrite:
            rw = rewrites.get(q, "")
            out["vector on rewrite"] = [cid for cid, _ in r._vector(rw or q, n_chunks)]
            out["multi-query rrf (app)"] = [h.chunk.id for h in r.search(q, rw, k=n_chunks).hits]
        return out

    print(f"{'':11} {'strategy':24} {'hit@1':>6} {'hit@3':>6} {'MRR':>5}")
    misses = []
    for group in GROUPS:
        totals: dict[str, np.ndarray] = {}
        for c in cases[group]:
            for name, order in strategies(c["q"]).items():
                pos = first_hit(order, c["expect"])
                totals.setdefault(name, np.zeros(3))
                totals[name] += (pos == 0, pos is not None and pos < 3, 0 if pos is None else 1 / (pos + 1))
                if name.endswith("(app)" if args.rewrite else "no rewrite") and (pos is None or pos >= 3):
                    misses.append(f"[{group}] {c['q']}  (rewrite: {rewrites.get(c['q'], '-')!r}) -> {order[:3]}")
        for name, t in totals.items():
            h1, h3, mrr = t / len(cases[group])
            print(f"{group:11} {name:24} {h1:6.0%} {h3:6.0%} {mrr:5.2f}")
        print()

    print(f"Confidence gate: answer only if the best cosine >= {embedder.min_score}")
    for group in (*GROUPS, "off_topic"):
        conf = [r.search(c["q"], rewrites.get(c["q"])) for c in cases[group]]
        scores = sorted(round(x.best_vector_score, 2) for x in conf)
        print(f"  {group:11} passes {sum(x.confident for x in conf)}/{len(conf)}   best-cosine range {scores[0]}-{scores[-1]}")

    if args.show_misses and misses:
        print("\nMisses (expected chunk not in top 3):")
        print("\n".join("  " + m for m in misses))


if __name__ == "__main__":
    main()

"""Run the scripted conversations in eval/cases.yaml through the agent graph and write a report.

    python eval/run_eval.py               # uses .env
    python eval/run_eval.py --offline     # rule-based LLM, no network
    python eval/run_eval.py --only faq    # one category or case id
    python eval/run_eval.py --tag before  # also save eval/runs/<date>-before.json

Scores route, facts, citations and compliance per turn, plus latency, model calls and tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # .env and the knowledge base are found relative to the project root

import yaml  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agents.graph import Deps, build_graph  # noqa: E402
from app.rules import reply_violations  # noqa: E402

EVAL_DIR = ROOT / "eval"
HOLDING = "(waiting for a human reviewer)"




def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("‑", "-").replace(" ", " ").lower()
    t = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", t)  # 5,000 / 5 000 -> 5000
    return re.sub(r"(?<=\d)\s+%", "%", t)  # 0.5 % -> 0.5%


def _matches(reply: str, item: str) -> bool:
    if item.startswith("re:"):
        return re.search(item[3:], reply) is not None
    return _norm(item) in reply


def check_turn(expect: dict, reply: str, route: str | None, status: str, sources: list[dict]) -> list[str]:
    fails: list[str] = []
    r = _norm(reply)
    want_route = expect.get("route")
    if want_route is not None:
        ok_routes = want_route if isinstance(want_route, list) else [want_route]
        if route not in ok_routes:
            fails.append(f"route: expected {'/'.join(ok_routes)}, got {route}")
    if expect.get("status") and status != expect["status"]:
        fails.append(f"status: expected {expect['status']}, got {status}")
    for item in expect.get("has") or []:
        options = item if isinstance(item, list) else [item]
        if not any(_matches(r, o) for o in options):
            fails.append("missing: " + " | ".join(options))
    for item in expect.get("has_not") or []:
        if _matches(r, item):
            fails.append(f"forbidden: {item}")
    if expect.get("cited") and not sources:
        fails.append("citations: none returned")
    if expect.get("source") and not any(expect["source"].lower() in s.get("source", "").lower() for s in sources):
        fails.append(f"citations: no source matching '{expect['source']}'")
    if status == "done":
        bad = reply_violations(reply)
        if bad:
            fails.append("compliance: " + ", ".join(bad))
    return fails




def _is_rate_limit(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return "429" in s or "rate limit" in s or "ratelimit" in s or "rate_limit" in s


async def _invoke(graph, payload, cfg, retries: int = 3):
    for attempt in range(retries + 1):
        try:
            return await graph.ainvoke(payload, cfg)
        except Exception as exc:  # noqa: BLE001
            if attempt < retries and _is_rate_limit(exc):
                wait = 20 * (attempt + 1)
                print(f"    rate limited - waiting {wait}s", flush=True)
                await asyncio.sleep(wait)
                continue
            raise


async def run_case(graph, llm, case: dict, run_id: str, delay: float) -> dict:
    cfg = {"configurable": {"thread_id": f"{case['id']}-{run_id}"}}
    turns_out = []
    for turn in case["turns"]:
        calls0, tokens0, t0 = getattr(llm, "calls", 0), getattr(llm, "tokens", 0), time.perf_counter()
        error = None
        try:
            if "decide" in turn:
                out = await _invoke(graph, Command(resume=turn["decide"]), cfg)
            else:
                out = await _invoke(graph, {
                    "message": turn["say"], "customer_id": case.get("customer", ""), "route": "", "reply": "",
                    "sources": [], "trace": [], "revisions": 0, "violations": [], "evidence": "", "critic_reason": "",
                }, cfg)
        except Exception as exc:  # noqa: BLE001
            out, error = {}, f"{type(exc).__name__}: {exc}"[:300]
        latency = time.perf_counter() - t0
        status = "pending_approval" if out.get("__interrupt__") else "done"
        reply = HOLDING if status == "pending_approval" else (out.get("reply") or "")
        sources = out.get("sources") or []
        fails = [f"error: {error}"] if error else check_turn(turn.get("expect") or {}, reply, out.get("route"), status, sources)
        turns_out.append({
            "input": turn.get("say") or {"decision": turn["decide"]},
            "kind": "decide" if "decide" in turn else "say",
            "reply": reply, "route": out.get("route"), "status": status,
            "sources": [s.get("source") for s in sources],
            "steps": [s.get("step") for s in out.get("trace") or []],
            "failures": fails, "latency_s": round(latency, 3),
            "llm_calls": getattr(llm, "calls", 0) - calls0, "tokens": getattr(llm, "tokens", 0) - tokens0,
        })
        if delay:
            await asyncio.sleep(delay)
    return {
        "id": case["id"], "category": case["category"], "customer": case.get("customer"), "note": case.get("note"),
        "passed": all(not t["failures"] for t in turns_out), "turns": turns_out,
    }




def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def summarize(results: list[dict]) -> dict:
    turns = [t for c in results for t in c["turns"]]
    said = [t for t in turns if t["kind"] == "say"]
    lat = sorted(t["latency_s"] for t in said) or [0.0]
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for c in results:
        by_cat[c["category"]].append(c["passed"])
    route_turns = [t for t in turns if not any(f.startswith("error") for f in t["failures"])]
    return {
        "cases": len(results),
        "cases_passed": sum(c["passed"] for c in results),
        "turns": len(turns),
        "by_category": {k: {"passed": sum(v), "total": len(v)} for k, v in sorted(by_cat.items())},
        "route_errors": sum(any(f.startswith("route") for f in t["failures"]) for t in route_turns),
        "fact_misses": sum(any(f.startswith("missing") for f in t["failures"]) for t in turns),
        "compliance_violations": sum(any(f.startswith(("compliance", "forbidden")) for f in t["failures"]) for t in turns),
        "errors": sum(any(f.startswith("error") for f in t["failures"]) for t in turns),
        "latency_p50_s": round(statistics.median(lat), 2),
        "latency_p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
        "llm_calls": sum(t["llm_calls"] for t in turns),
        "tokens": sum(t["tokens"] for t in turns),
        "avg_llm_calls_per_message": round(sum(t["llm_calls"] for t in said) / max(1, len(said)), 2),
    }


def write_report(meta: dict, summary: dict, results: list[dict], tag: str | None, base: str = "report") -> Path:
    data = {"meta": meta, "summary": summary, "results": results}
    (EVAL_DIR / f"{base}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if tag:
        runs = EVAL_DIR / "runs"
        runs.mkdir(exist_ok=True)
        (runs / f"{meta['started'][:10]}-{tag}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    s = summary
    lines = [
        "# Evaluation report", "",
        f"- **Run:** {meta['started']} · model `{meta['llm']}` · embedder `{meta['embedder']}`",
        f"- **Conversations passed:** {s['cases_passed']} / {s['cases']} ({_pct(s['cases_passed'], s['cases'])})",
        f"- **Turns checked:** {s['turns']} · route errors {s['route_errors']} · missing facts {s['fact_misses']} · "
        f"compliance violations **{s['compliance_violations']}** · errors {s['errors']}",
        f"- **Latency per message:** p50 {s['latency_p50_s']}s · p95 {s['latency_p95_s']}s",
        f"- **Cost:** {s['llm_calls']} model calls ({s['avg_llm_calls_per_message']} per message), {s['tokens']:,} tokens",
        "", "| Category | Passed |", "|---|---|",
    ]
    lines += [f"| {k} | {v['passed']} / {v['total']} ({_pct(v['passed'], v['total'])}) |" for k, v in s["by_category"].items()]
    failed = [c for c in results if not c["passed"]]
    lines += ["", f"## Failures ({len(failed)})", ""]
    if not failed:
        lines.append("None.")
    for c in failed:
        lines.append(f"### {c['id']} ({c['category']})")
        for t in c["turns"]:
            if t["failures"]:
                said = t["input"] if isinstance(t["input"], str) else json.dumps(t["input"])
                lines.append(f"- **Input:** {said}")
                lines.append(f"  - route `{t['route']}`, status `{t['status']}`")
                lines.append(f"  - reply: {t['reply'][:300].replace(chr(10), ' ')}")
                for f in t["failures"]:
                    lines.append(f"  - ✗ {f}")
        lines.append("")
    path = EVAL_DIR / f"{base}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="rule-based LLM + lexical embedder, no network")
    ap.add_argument("--only", help="run only this category or case id")
    ap.add_argument("--delay", type=float, default=None, help="seconds between turns (default 2 with a real model, 0 offline)")
    ap.add_argument("--tag", help="also save eval/runs/<date>-<tag>.json, for before/after comparisons")
    args = ap.parse_args()

    from app.mcp_server import db
    from app.tools import McpToolGateway

    workdir = tempfile.mkdtemp(prefix="aurelia-eval-")
    os.environ["MOCK_DB_PATH"] = str(Path(workdir) / "mock.db")  # never touch the demo's real mock database

    if args.offline:
        from app.llm import OfflineLLM
        from app.rag.embedders import LexicalEmbedder
        from app.rag.ingest import build_retriever
        from app.rag.stores import MemoryStore

        llm, retriever = OfflineLLM(), build_retriever(MemoryStore(), LexicalEmbedder())
    else:
        from app.llm import get_llm
        from app.rag.ingest import build_retriever

        llm, retriever = get_llm(), build_retriever()
    delay = args.delay if args.delay is not None else (0.0 if llm.name == "offline-rules" else 2.0)

    cases = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in (c["id"], c["category"])]
    meta = {"started": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "llm": llm.name, "embedder": retriever.embedder.name,
            "cases_file": "eval/cases.yaml"}
    print(f"Running {len(cases)} conversations with {llm.name} / {retriever.embedder.name}\n", flush=True)

    run_id = str(int(time.time()))
    results = []
    async with McpToolGateway(os.environ["MOCK_DB_PATH"]) as gw:
        graph = build_graph(Deps(llm, retriever, gw), MemorySaver())
        for i, case in enumerate(cases, 1):
            db.reset()  # every conversation starts from the same seeded back office
            res = await run_case(graph, llm, case, run_id, delay)
            results.append(res)
            mark = "PASS" if res["passed"] else "FAIL"
            print(f"[{i:2}/{len(cases)}] {mark}  {case['id']}", flush=True)
            for t in res["turns"]:
                for f in t["failures"]:
                    print(f"         ✗ {f}", flush=True)

    summary = summarize(results)
    # offline and partial runs never overwrite the full report the demo page shows
    base = "report" if llm.name != "offline-rules" and not args.only else "report.scratch"
    path = write_report(meta, summary, results, args.tag, base)
    print(f"\n{summary['cases_passed']}/{summary['cases']} conversations passed · "
          f"{summary['compliance_violations']} compliance violations · report: {path.relative_to(ROOT)}")
    return 0 if summary["cases_passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""FastAPI service: chat, human-approval queue, health.

    uvicorn app.api:app --reload
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.graph import Deps, build_graph
from app.audit import AuditWriter, body_to_json, make_sink, response_to_json
from app.config import get_settings
from app.i18n import detect_language, msg
from app.llm import get_llm
from app.mcp_server import db as mock_db
from app.mcp_server.db import SEED_CUSTOMERS
from app.rag.ingest import build_retriever
from app.rag.stores import MemoryStore
from app.tools import McpToolGateway

log = logging.getLogger("aurelia.api")


def _reason(exc: Exception) -> str:
    """Last line of a database error (e.g. 'FATAL: password authentication failed'); never includes the DSN."""
    lines = [l.strip() for l in str(exc).splitlines() if l.strip()]
    return f"{type(exc).__name__}: {lines[-1][:200] if lines else ''}"

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

STATIC_DIR = Path(__file__).parent / "static"
EVAL_REPORT = Path(__file__).resolve().parents[1] / "eval" / "report.json"



class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    customer_id: str | None = Field(default=None, pattern=r"^C-\d{4}$")
    message: str = Field(max_length=5000)


class Decision(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=300)


class RateLimiter:
    """Sliding-window limiter per client. In-memory: fine for a single demo instance."""

    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def check(self, key: str) -> None:
        now, q = time.monotonic(), self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            raise HTTPException(429, "Too many requests - please wait a moment.")
        q.append(now)


class DailyCap:
    """Caps total chat messages per UTC day so a public demo can't drain the free LLM quota. 0 = no cap."""

    def __init__(self, limit: int):
        self.limit, self.day, self.count = limit, "", 0

    def check(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.count = today, 0
        if self.limit and self.count >= self.limit:
            raise HTTPException(429, "The demo has reached today's message limit (it runs on a free model quota). Please try again tomorrow.")
        self.count += 1


async def _make_checkpointer(stack: AsyncExitStack):
    s = get_settings()
    if s.checkpointer == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        if not s.database_url:
            raise RuntimeError("CHECKPOINTER=postgres needs DATABASE_URL")
        if sys.platform == "win32":  # async psycopg needs a selector loop; the MCP subprocess needs the proactor loop
            raise RuntimeError("CHECKPOINTER=postgres runs on Linux/Docker only. On Windows use CHECKPOINTER=memory.")
        saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(s.database_url))
        await saver.setup()
        return saver
    return MemorySaver()


def create_app(*, llm=None, retriever=None, mock_db_path: str | None = None) -> FastAPI:
    """Factory so tests can inject an offline LLM / in-memory retriever / temp database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = get_settings()
        async with AsyncExitStack() as stack:
            the_llm = llm or get_llm()
            app.state.warnings = []
            try:
                the_retriever = retriever or await asyncio.to_thread(build_retriever)
            except Exception as exc:
                log.error("vector store unavailable (%s); using in-memory vectors", _reason(exc))
                app.state.warnings.append(f"vector store unavailable ({_reason(exc)}); using in-memory vectors")
                the_retriever = await asyncio.to_thread(build_retriever, MemoryStore())
            gateway = await stack.enter_async_context(McpToolGateway(mock_db_path))

            http = httpx.AsyncClient(timeout=3.0)
            stack.push_async_callback(http.aclose)

            async def notify(payload: dict) -> None:
                if s.escalation_webhook_url:
                    await http.post(s.escalation_webhook_url, json=payload)

            checkpointer = await _make_checkpointer(stack)
            app.state.graph = build_graph(Deps(the_llm, the_retriever, gateway, notify), checkpointer)
            app.state.llm, app.state.retriever, app.state.gateway = the_llm, the_retriever, gateway
            app.state.pending: dict[str, dict] = {}
            app.state.decisions: list[dict] = []  # every reviewer decision, newest last
            app.state.limiter = RateLimiter(s.rate_limit_per_minute)
            app.state.daily_cap = DailyCap(s.demo_daily_cap)
            app.state.mock_db_path = mock_db_path or s.mock_db_path
            try:
                sink = await asyncio.to_thread(make_sink, s.audit_log, s.database_url)
            except Exception as exc:
                log.error("request log unavailable (%s)", _reason(exc))
                app.state.warnings.append(f"request log unavailable ({_reason(exc)})")
                sink = None
            app.state.audit = AuditWriter(sink)
            yield
            await app.state.audit.drain()
            if sink is not None:
                sink.close()

    app = FastAPI(title="Aurelia Support Agent (demo)", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def audit_log(request: Request, call_next):
        """Log every request with its timing, request body and response body (never headers or IP addresses)."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        body = await request.body()

        def record(status: int, req_json: Any, resp_json: Any) -> None:
            audit = getattr(request.app.state, "audit", None)
            if audit is None:
                return
            req_d = req_json if isinstance(req_json, dict) else {}
            resp_d = resp_json if isinstance(resp_json, dict) else {}
            path_params = request.scope.get("path_params") or {}
            audit.submit({
                "request_id": request_id, "started_at": started_at, "method": request.method, "path": request.url.path,
                "query": request.url.query[:500] or None, "status_code": status,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "session_id": req_d.get("session_id") or path_params.get("approval_id") or path_params.get("session_id") or resp_d.get("approval_id"),
                "customer_id": req_d.get("customer_id"), "route": resp_d.get("route"),
                "user_agent": (request.headers.get("user-agent") or "")[:300] or None,
                "request": req_json, "response": resp_json,
            })

        try:
            response = await call_next(request)
        except Exception as exc:
            record(500, body_to_json(body), {"error": type(exc).__name__, "detail": str(exc)[:500]})
            raise

        chunks = [chunk async for chunk in response.body_iterator]
        resp_body = b"".join(chunks)
        record(response.status_code, body_to_json(body),
               response_to_json(resp_body, response.headers.get("content-type", ""), request.method, response.status_code))
        headers = dict(response.headers)
        headers["X-Request-ID"] = request_id
        return Response(content=resp_body, status_code=response.status_code, headers=headers, media_type=response.media_type)

    def need_api_key(x_api_key: str | None = Header(default=None)) -> None:
        key = get_settings().api_key
        if key and x_api_key != key:
            raise HTTPException(401, "Missing or wrong X-API-Key")

    def need_reviewer_key(x_reviewer_key: str | None = Header(default=None)) -> None:
        key = get_settings().reviewer_key
        if key and x_reviewer_key != key:
            raise HTTPException(401, "Missing or wrong X-Reviewer-Key")

    def public(state: dict) -> dict[str, Any]:
        return {"route": state.get("route"), "sources": state.get("sources") or [], "trace": state.get("trace") or []}

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health(request: Request):
        s = get_settings()
        return {
            "status": "ok",
            "llm": request.app.state.llm.name,
            "embedder": request.app.state.retriever.embedder.name,
            "vector_store": "pgvector" if type(request.app.state.retriever.store).__name__ == "PgVectorStore" else "memory",
            "request_log": type(request.app.state.audit.sink).__name__ if request.app.state.audit.sink else "off",
            "warnings": request.app.state.warnings,
            "checkpointer": s.checkpointer,
            "model_visible_tools": [t["function"]["name"] for t in request.app.state.gateway.model_tool_specs],
            "reviewer_key_required": bool(s.reviewer_key),
            "messages_today": request.app.state.daily_cap.count,
            "daily_cap": s.demo_daily_cap,
        }

    @app.get("/eval/report")
    async def eval_report():
        """Latest evaluation run (written by eval/run_eval.py)."""
        if not EVAL_REPORT.exists():
            raise HTTPException(404, "No evaluation report yet - run: python eval/run_eval.py")
        data = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
        return {"meta": data["meta"], "summary": data["summary"],
                "failures": [{"id": c["id"], "category": c["category"]} for c in data["results"] if not c["passed"]]}

    @app.post("/demo/reset", dependencies=[Depends(need_api_key)])
    async def demo_reset(request: Request):
        """Put the mock back office (limits, tickets, closures) back to its starting state."""
        request.app.state.limiter.check(request.client.host if request.client else "unknown")
        await asyncio.to_thread(mock_db.reset, request.app.state.mock_db_path)
        request.app.state.pending.clear()
        return {"status": "reset"}

    @app.get("/customers")
    async def customers():
        """Demo customers for the UI's sign-in picker (names only - no account data)."""
        return [{"id": c[0], "name": c[1], "account_type": c[2]} for c in SEED_CUSTOMERS]

    @app.post("/chat", dependencies=[Depends(need_api_key)])
    async def chat(req: ChatRequest, request: Request):
        request.app.state.limiter.check(request.client.host if request.client else "unknown")
        request.app.state.daily_cap.check()
        graph, cfg = request.app.state.graph, {"configurable": {"thread_id": req.session_id}}

        snapshot = await graph.aget_state(cfg)
        if snapshot.next:
            return {"session_id": req.session_id, "status": "pending_approval", "reply": msg("still_pending", detect_language(req.message)),
                    "approval_id": req.session_id, "route": None, "sources": [], "trace": []}

        result = await graph.ainvoke(
            {"message": req.message, "customer_id": req.customer_id or "", "route": "", "reply": "", "sources": [], "trace": [], "revisions": 0, "violations": [], "evidence": "", "critic_reason": ""},
            cfg,
        )
        interrupts = result.get("__interrupt__")
        if interrupts:
            action = interrupts[0].value["action"]
            request.app.state.pending[req.session_id] = {
                "approval_id": req.session_id, "customer_id": action["customer_id"], "action": action,
                "trace_id": getattr(request.state, "request_id", None), "requested_at": _now(),
            }
            return {"session_id": req.session_id, "status": "pending_approval", "reply": msg("holding", result.get("lang", "en")), "approval_id": req.session_id, **public(result)}
        return {"session_id": req.session_id, "status": "done", "reply": result["reply"], **public(result)}

    @app.get("/sessions/{session_id}", dependencies=[Depends(need_api_key)])
    async def session(session_id: str, request: Request):
        """Lets the chat page poll for the outcome of a request that was waiting for a human."""
        snap = await request.app.state.graph.aget_state({"configurable": {"thread_id": session_id}})
        if not snap.values:
            raise HTTPException(404, "Unknown session")
        history = snap.values.get("history") or []
        last = next((h["content"] for h in reversed(history) if h["role"] == "assistant"), None)
        return {"session_id": session_id, "status": "pending_approval" if snap.next else "done", "last_reply": last, "history": history,
                "route": snap.values.get("route"), "sources": snap.values.get("sources") or [], "trace": snap.values.get("trace") or []}

    @app.get("/approvals", dependencies=[Depends(need_reviewer_key)])
    async def approvals(request: Request):
        return list(request.app.state.pending.values())

    @app.post("/approvals/{approval_id}/decision", dependencies=[Depends(need_reviewer_key)])
    async def decide(approval_id: str, decision: Decision, request: Request):
        if approval_id not in request.app.state.pending:
            raise HTTPException(404, "No pending approval with that id")
        cfg = {"configurable": {"thread_id": approval_id}}
        result = await request.app.state.graph.ainvoke(Command(resume=decision.model_dump()), cfg)
        pending = request.app.state.pending.pop(approval_id, None) or {}
        record = {
            "approval_id": approval_id,
            "action": (pending.get("action") or {}).get("type"),
            "customer_id": pending.get("customer_id"),
            "decision": "approved" if decision.approved else "declined",
            "reviewer": decision.reviewer,
            "note": decision.note,
            "requested_at": pending.get("requested_at"),
            "decided_at": _now(),
            "request_trace_id": pending.get("trace_id"),
            "decision_trace_id": getattr(request.state, "request_id", None),
        }
        request.app.state.decisions.append(record)
        del request.app.state.decisions[:-200]
        return {"approval_id": approval_id, "status": "done", "reply": result["reply"], "audit": record, **public(result)}

    @app.get("/approvals/audit", dependencies=[Depends(need_reviewer_key)])
    async def approval_audit(request: Request):
        """Reviewer decisions with the trace ids of the request and the decision; each one is also in the request log."""
        return list(reversed(request.app.state.decisions))

    return app


app = create_app()

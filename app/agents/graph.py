"""Support agent workflow (LangGraph).

Safety decisions are plain code; the model only writes text. Requests that change an account
pause at `approval_gate` until a reviewer decides, and every node appends to `trace`.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents import prompts
from app.i18n import MESSAGES, detect_language, msg, reply_language_line
from app.llm import LLM
from app.mcp_server.db import TIER_LIMITS
from app.rag.retriever import HybridRetriever
from app.rules import ROUTES, account_intents, normalize, check_input, extract_amount, is_advice_request, is_info_question, reply_violations, rule_route
from app.tools import McpToolGateway, ToolError

log = logging.getLogger("aurelia.graph")

MAX_HISTORY = 8
SENSITIVE_ROUTES = {"limit_exception", "account_closure"}  # decided by rules, never by the LLM alone
RULE_DECIDES = SENSITIVE_ROUTES | {"advice", "account", "complaint"}  # the LLM may only refine the rest (faq vs out_of_scope)


class AgentState(TypedDict, total=False):
    customer_id: str
    message: str
    history: list[dict]
    route: str
    reply: str
    sources: list[dict]
    trace: list[dict]
    slot: dict | None  # a half-finished request waiting for the customer's next message
    pending_action: dict | None  # a finished request waiting for a human reviewer
    revisions: int
    violations: list[str]
    critic_reason: str
    search_query: str
    evidence: str  # what the reply was written from (sources / tool results), so the critic can check grounding
    status: str  # done | pending_approval
    lang: str  # "en" or "ar", detected from the customer's message


@dataclass
class Deps:
    llm: LLM
    retriever: HybridRetriever
    gateway: McpToolGateway
    notify: Callable[[dict], Awaitable[None]] | None = None  # e.g. webhook for new complaints



SAFE_FALLBACK = msg("safe_fallback")
ADVICE_REFUSAL = msg("advice_refusal")
OUT_OF_SCOPE = msg("out_of_scope")
NEED_SIGN_IN = msg("need_sign_in")
SAFE_FALLBACKS = {msg("safe_fallback", "en"), msg("safe_fallback", "ar")}
_MESSAGE_KEYS = set(MESSAGES)


def _t(state: AgentState, step: str, **info: Any) -> list[dict]:
    return [*(state.get("trace") or []), {"step": step, **info}]


def _timed(fn: Callable[[AgentState], Awaitable[dict]], llm: Any) -> Callable[[AgentState], Awaitable[dict]]:
    """Stamp the step a node adds with its duration and the model calls and tokens it used.

    The model counters are shared by all requests, so under concurrent traffic the per-step numbers are approximate."""

    async def node(state: AgentState) -> dict:
        start, calls, tokens = time.perf_counter(), getattr(llm, "calls", 0), getattr(llm, "tokens", 0)
        out = await fn(state)
        before = len(state.get("trace") or [])
        trace = out.get("trace") if isinstance(out, dict) else None
        if trace and len(trace) > before:
            last = {"ms": round((time.perf_counter() - start) * 1000), **trace[-1],
                    "llm_calls": getattr(llm, "calls", 0) - calls, "tokens": getattr(llm, "tokens", 0) - tokens}
            out["trace"] = [*trace[:-1], last]
        return out

    return node


def _json_from(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _history_text(history: list[dict], n: int = 4) -> str:
    return "\n".join(f"{h['role']}: {h['content']}" for h in (history or [])[-n:])


_NATIVE_CITE = re.compile(r"【\s*(\d+)[^】]*】|\[(\d+)†[^\]]*\]")


def _clean_citations(text: str) -> str:
    """Some models (gpt-oss) cite as 【1】 or 【1†L1-L2】; normalise to [1] so sources can be shown."""
    return _NATIVE_CITE.sub(lambda m: f"[{m.group(1) or m.group(2)}]", text)


_REASON_AR = re.compile(r"(لان|بسبب|لدفع|لسداد|رسوم|ايجار|علاج|الجامعه|دراسه|سفر|شراء|عقار|انتقال|مغادره|اغادر|سبب)")


def _has_reason(message: str) -> bool:
    return bool(
        re.search(r"\b(because|for|to pay|reason|since|tuition|invoice|purchase|property|medical|rent|moving|leaving|switching)\b", message.lower())
        or _REASON_AR.search(normalize(message))
        or len(message.split()) >= 9
    )




def build_graph(deps: Deps, checkpointer: Any):
    llm, retriever, gw = deps.llm, deps.retriever, deps.gateway

    async def guard_in(state: AgentState) -> dict:
        lang = detect_language(state["message"])
        res = check_input(state["message"])
        if res.ok:
            return {"route": "", "lang": lang, "trace": _t(state, "input_guard", ok=True, lang=lang)}
        key = f"blocked_{res.reason}"
        return {
            "route": "blocked",
            "lang": lang,
            "reply": msg(key, lang) if key in _MESSAGE_KEYS else msg("safe_fallback", lang),
            "sources": [],
            "trace": _t(state, "input_guard", ok=False, reason=res.reason, lang=lang),
        }

    async def route(state: AgentState) -> dict:
        message = state["message"]
        slot = state.get("slot")
        floor = rule_route(message)
        changed_topic = floor in {"advice", "complaint"} or (floor in {"faq", "account"} and is_info_question(message))
        if slot and not changed_topic:
            return {"route": slot["intent"], "trace": _t(state, "route", route=slot["intent"], by="open_request")}

        label, by, search_query = floor, "rules", ""
        if floor not in RULE_DECIDES:
            prev_user = next((h["content"] for h in reversed(state.get("history") or []) if h["role"] == "user"), "")
            router_input = f"PREVIOUS MESSAGE: {prev_user}\nMESSAGE: {message}" if prev_user else f"MESSAGE: {message}"
            try:
                parsed = _json_from(await llm.complete("route", prompts.ROUTER_SYSTEM, router_input))
                guess = parsed.get("route")
                # Sending a message to the advice refusal is always safe. The rule floor covers English well, so
                # there the LLM may not add refusals; for Arabic the regexes are thinner, so the LLM may.
                may_refuse = state.get("lang") == "ar"
                if guess in ROUTES and guess not in SENSITIVE_ROUTES and (guess != "advice" or may_refuse):
                    label, by = guess, "llm"
                search_query = str(parsed.get("search_query") or "")[:200]
            except Exception as exc:
                log.warning("router LLM failed, using rules: %s", exc)
                by = "rules (llm error)"
        return {"route": label, "search_query": search_query, "slot": None if label != (slot or {}).get("intent") else slot,
                "trace": _t(state, "route", route=label, by=by, rule_floor=floor, search_query=search_query)}

    def pick_route(state: AgentState) -> str:
        return {
            "faq": "faq", "account": "account", "complaint": "complaint", "limit_exception": "limit_exception",
            "account_closure": "closure", "advice": "advice", "out_of_scope": "out_of_scope",
        }.get(state["route"], "out_of_scope")

    async def faq(state: AgentState) -> dict:
        message = state["message"]
        query = message
        rewrite = state.get("search_query") or ""
        lang = state.get("lang", "en")
        prev_user = next((h["content"] for h in reversed(state.get("history") or []) if h["role"] == "user"), None)
        if lang == "ar" and rewrite:
            query = rewrite  # the documents and the embedding model are English; the router's rewrite is English
        elif prev_user and len(message.split()) < 5 and not rewrite:
            query = f"{prev_user} {message}"  # without a rewrite, short follow-ups need the previous question
        t0 = time.perf_counter()
        found = await _to_thread(retriever.search, query, rewrite)
        trace = _t(state, "retrieve", ms=round((time.perf_counter() - t0) * 1000), query=query, rewrite=rewrite, confident=found.confident, best_score=round(found.best_vector_score, 3),
                   hits=[{"source": h.chunk.citation, "rrf": round(h.rrf, 4), "vector": None if h.vector_score is None else round(h.vector_score, 3),
                          "bm25_rank": h.bm25_rank, "vector_rank": h.vector_rank, "snippet": h.chunk.content[:220]} for h in found.hits])
        if not found.confident:
            return {"reply": msg("not_in_documents", lang), "sources": [], "trace": trace}
        sources_txt = "\n".join(f"[{i + 1}] {h.chunk.citation}: {h.chunk.content}" for i, h in enumerate(found.hits))
        user = (f"CONVERSATION:\n{_history_text(state.get('history') or [])}\n\nQUESTION: {message}\n\nSOURCES:\n{sources_txt}"
                + reply_language_line(lang))
        t0 = time.perf_counter()
        reply = _clean_citations(await llm.complete("answer", prompts.ANSWER_SYSTEM, user))
        answer_ms = round((time.perf_counter() - t0) * 1000)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", reply)}  # only show sources the answer actually used
        return {
            "reply": reply,
            "sources": [{"n": i + 1, "source": h.chunk.citation} for i, h in enumerate(found.hits) if (i + 1) in cited],
            "evidence": sources_txt,
            "trace": [*trace, {"step": "answer", "ms": answer_ms, "model": llm.name, "cited": sorted(cited)}],
        }

    async def account(state: AgentState) -> dict:
        cid, lang = state.get("customer_id"), state.get("lang", "en")
        if not cid:
            return {"reply": msg("need_sign_in", lang), "sources": [], "trace": _t(state, "account", error="not_signed_in")}
        chosen = await llm.select_tools(state["message"], gw.model_tool_specs)
        how = "llm_tool_calling"
        if not chosen:
            how = "rules"
            m = {"kyc": "get_kyc_status", "limits": "get_withdrawal_limits", "tickets": "list_tickets", "summary": "get_account_summary"}
            chosen = [{"name": m[i], "args": {}} for i in account_intents(state["message"])]
        results: dict[str, Any] = {}
        calls = []
        for call in chosen[:3]:
            try:
                results[call["name"]] = await gw.call(call["name"], cid, call.get("args"))
                calls.append({"tool": call["name"], "ok": True})
            except ToolError as exc:
                calls.append({"tool": call["name"], "ok": False, "error": str(exc)})
        trace = _t(state, "tools", selected_by=how, calls=calls, customer=cid)
        if not results:
            return {"reply": msg("safe_fallback", lang), "sources": [], "trace": trace}
        user = f"QUESTION: {state['message']}\n\nTOOL_RESULTS: {json.dumps(results)}" + reply_language_line(lang)
        reply = await llm.complete("account_reply", prompts.ACCOUNT_SYSTEM, user)
        return {"reply": reply, "sources": [], "evidence": json.dumps(results), "trace": [*trace, {"step": "answer", "model": llm.name}]}

    async def complaint(state: AgentState) -> dict:
        cid, lang = state.get("customer_id"), state.get("lang", "en")
        if not cid:
            return {"reply": msg("complaint_sign_in", lang), "sources": [], "trace": _t(state, "complaint", error="not_signed_in")}
        summary = state["message"][:400]
        res = await gw.call("create_ticket", cid, {"category": "complaint", "summary": summary, "priority": "high"})
        if "error" in res:
            return {"reply": msg("safe_fallback", lang), "sources": [], "trace": _t(state, "complaint", error=res["error"])}
        if deps.notify:
            try:
                await deps.notify({"event": "complaint_logged", "ticket_id": res["ticket_id"], "customer_id": cid, "summary": summary})
            except Exception as exc:
                log.warning("complaint webhook failed: %s", exc)
        return {
            "reply": msg("complaint_logged", lang, ticket_id=res["ticket_id"]),
            "sources": [{"n": 1, "source": "Complaints Policy > What happens next"}],
            "trace": _t(state, "complaint", ticket_id=res["ticket_id"], webhook=bool(deps.notify)),
        }

    async def limit_exception(state: AgentState) -> dict:
        cid, lang = state.get("customer_id"), state.get("lang", "en")
        if not cid:
            return {"reply": msg("need_sign_in", lang), "sources": [], "trace": _t(state, "limit_exception", error="not_signed_in")}
        slot = dict(state.get("slot") or {"intent": "limit_exception", "amount": None, "reason": None})
        text = state["message"]
        slot["amount"] = slot["amount"] or extract_amount(text)
        if not slot["reason"] and _has_reason(text):
            slot["reason"] = text[:300]
        if not slot["amount"]:
            return {"reply": msg("limit_ask_amount", lang), "slot": slot, "sources": [],
                    "trace": _t(state, "limit_exception", waiting_for="amount")}
        if not slot["reason"]:
            return {"reply": msg("limit_ask_reason", lang), "slot": slot, "sources": [],
                    "trace": _t(state, "limit_exception", waiting_for="reason")}

        limits = await gw.call("get_withdrawal_limits", cid)
        kyc = await gw.call("get_kyc_status", cid)
        current, ceiling = limits.get("daily_withdrawal_limit_usd"), TIER_LIMITS.get((kyc.get("kyc_tier") or 0) + 1)
        amount = int(slot["amount"])
        base = _t(state, "policy_precheck", current_limit=current, requested=amount, ceiling=ceiling)
        if ceiling is None or current is None or amount <= current or amount > ceiling:
            ceiling_part = msg("limit_ceiling_part", lang, ceiling=ceiling) if ceiling else ""
            return {
                "reply": msg("limit_not_eligible", lang, current=current, ceiling_part=ceiling_part),
                "sources": [{"n": 1, "source": "Deposits and Withdrawals > Withdrawal limits and exception requests"}],
                "slot": None, "trace": base,
            }
        action = {
            "type": "limit_exception", "customer_id": cid, "requested_daily_limit": amount, "reason": slot["reason"], "lang": lang,
            "summary": f"Raise daily withdrawal limit from {current} to {amount} USD for {cid}. Reason: {slot['reason']}",
        }
        return {"slot": None, "pending_action": action, "trace": [*base, {"step": "needs_approval", "action": action["type"]}]}

    async def closure(state: AgentState) -> dict:
        cid, lang = state.get("customer_id"), state.get("lang", "en")
        if not cid:
            return {"reply": msg("need_sign_in", lang), "sources": [], "trace": _t(state, "account_closure", error="not_signed_in")}
        slot = dict(state.get("slot") or {"intent": "account_closure", "reason": None})
        text = state["message"]
        if not slot.get("reason") and _has_reason(text) and not re.fullmatch(r"(please )?(close|delete|terminate) (my )?account\.?", text.lower().strip()):
            slot["reason"] = text[:300]
        if not slot.get("reason"):
            return {"reply": msg("closure_ask_reason", lang),
                    "slot": slot, "sources": [], "trace": _t(state, "account_closure", waiting_for="reason")}
        summary = await gw.call("get_account_summary", cid)
        if summary.get("has_open_positions"):
            return {"reply": msg("closure_open_positions", lang),
                    "sources": [{"n": 1, "source": "Deposits and Withdrawals > Account closure"}], "slot": None,
                    "trace": _t(state, "account_closure", blocked="open_positions")}
        action = {"type": "account_closure", "customer_id": cid, "reason": slot["reason"], "lang": lang,
                  "summary": f"Close account for {cid}. Reason: {slot['reason']}"}
        return {"slot": None, "pending_action": action, "trace": _t(state, "needs_approval", action="account_closure")}

    def needs_human(state: AgentState) -> str:
        return "approval_gate" if state.get("pending_action") else "critic"

    async def approval_gate(state: AgentState) -> dict:
        action = state["pending_action"]
        # NOTE: on resume LangGraph re-runs this node from the top, so nothing above interrupt() may have side effects.
        decision = interrupt({"action": action, "question": "Approve this customer request?"})
        approved = bool(decision.get("approved"))
        reviewer = (decision.get("reviewer") or "").strip() or "unknown-reviewer"
        trace = _t(state, "human_decision", approved=approved, reviewer=reviewer, note=decision.get("note", ""))
        lang = action.get("lang", "en")
        if not approved:
            note = decision.get("note") or msg("review_declined_default_note", lang)
            return {"reply": msg("review_declined", lang, note=note), "sources": [], "pending_action": None, "trace": trace}
        cid = action["customer_id"]
        if action["type"] == "limit_exception":
            res = await gw.call("submit_limit_exception_request", cid,
                                {"requested_daily_limit": action["requested_daily_limit"], "reason": action["reason"]},
                                approved_by=reviewer, privileged=True)
            ok = res.get("status") == "granted"
            reply = (msg("limit_granted", lang, amount=res["new_daily_limit_usd"])
                     if ok else msg("limit_apply_failed", lang, error=res.get("error", "unknown error")))
        else:
            res = await gw.call("submit_account_closure_request", cid, {"reason": action["reason"]}, approved_by=reviewer, privileged=True)
            ok = res.get("status") == "closure_pending"
            reply = (msg("closure_approved", lang) if ok
                     else msg("closure_apply_failed", lang, error=res.get("error", "unknown error")))
        return {"reply": reply, "sources": [], "pending_action": None, "trace": [*trace, {"step": "tool_executed", "tool": action["type"], "ok": ok}]}

    async def advice(state: AgentState) -> dict:
        return {"reply": msg("advice_refusal", state.get("lang", "en")), "sources": [{"n": 1, "source": "Risk Disclosure > No investment advice"}], "trace": _t(state, "advice_refusal")}

    async def out_of_scope(state: AgentState) -> dict:
        return {"reply": msg("out_of_scope", state.get("lang", "en")), "sources": [], "trace": _t(state, "out_of_scope")}

    async def critic(state: AgentState) -> dict:
        reply = state["reply"]
        found = reply_violations(reply)
        llm_reason = ""
        if state.get("route") in {"faq", "account"} and not found:
            try:
                evidence = state.get("evidence") or "(none)"
                verdict = _json_from(await llm.complete("critic", prompts.CRITIC_SYSTEM, f"EVIDENCE:\n{evidence}\n\nDRAFT:\n{reply}"))
                if verdict.get("verdict") == "fail":
                    found = ["llm_critic"]
                    llm_reason = str(verdict.get("reason", ""))[:200]
            except Exception as exc:
                log.warning("critic LLM failed, deterministic check only: %s", exc)
        return {"violations": found, "critic_reason": llm_reason, "trace": _t(state, "critic", violations=found, llm_reason=llm_reason, revision=state.get("revisions", 0))}

    def after_critic(state: AgentState) -> str:
        return "revise" if state.get("violations") else "finalize"

    async def revise(state: AgentState) -> dict:
        n = state.get("revisions", 0)
        lang = state.get("lang", "en")
        if n >= 1:  # one rewrite attempt, then a safe fallback - never loop forever
            return {"reply": msg("safe_fallback", lang), "sources": [], "violations": [], "revisions": n + 1, "trace": _t(state, "fallback", reason="still_violating_after_revision")}
        problems = ", ".join(state["violations"]) + (f" ({state['critic_reason']})" if state.get("critic_reason") else "")
        evidence = state.get("evidence") or "(none)"
        rewritten = _clean_citations(await llm.complete(
            "rewrite", prompts.REWRITE_SYSTEM,
            f"PROBLEMS: {problems}\n\nEVIDENCE:\n{evidence}\n\nDRAFT:\n{state['reply']}" + reply_language_line(lang)))
        return {"reply": rewritten, "revisions": n + 1, "trace": _t(state, "revise", problems=state["violations"])}

    def after_revise(state: AgentState) -> str:
        return "finalize" if state.get("reply") in SAFE_FALLBACKS else "critic"

    async def finalize(state: AgentState) -> dict:
        history = [*(state.get("history") or []), {"role": "user", "content": state["message"]}, {"role": "assistant", "content": state["reply"]}]
        return {"history": history[-MAX_HISTORY:], "status": "done", "trace": _t(state, "done")}

    g = StateGraph(AgentState)
    for name, fn in [
        ("guard_in", guard_in), ("route", route), ("faq", faq), ("account", account), ("complaint", complaint),
        ("limit_exception", limit_exception), ("closure", closure), ("approval_gate", approval_gate), ("advice", advice),
        ("out_of_scope", out_of_scope), ("critic", critic), ("revise", revise), ("finalize", finalize),
    ]:
        g.add_node(name, _timed(fn, llm))

    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", lambda s: "finalize" if s.get("route") == "blocked" else "route", {"finalize": "finalize", "route": "route"})
    g.add_conditional_edges("route", pick_route, {k: k for k in ("faq", "account", "complaint", "limit_exception", "closure", "advice", "out_of_scope")})
    for n in ("faq", "account", "complaint", "advice", "out_of_scope", "approval_gate"):
        g.add_edge(n, "critic")
    for n in ("limit_exception", "closure"):
        g.add_conditional_edges(n, needs_human, {"approval_gate": "approval_gate", "critic": "critic"})
    g.add_conditional_edges("critic", after_critic, {"revise": "revise", "finalize": "finalize"})
    g.add_conditional_edges("revise", after_revise, {"critic": "critic", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


async def _to_thread(fn, *args):
    import asyncio

    return await asyncio.to_thread(fn, *args)

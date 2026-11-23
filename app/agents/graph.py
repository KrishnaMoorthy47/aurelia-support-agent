"""Support agent workflow (LangGraph).

Safety decisions are plain code; the model only writes text. Requests that change an account
pause at `approval_gate` until a reviewer decides, and every node appends to `trace`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents import prompts
from app.llm import LLM
from app.mcp_server.db import TIER_LIMITS
from app.rag.retriever import HybridRetriever
from app.rules import ROUTES, account_intents, check_input, extract_amount, is_advice_request, is_info_question, reply_violations, rule_route
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


@dataclass
class Deps:
    llm: LLM
    retriever: HybridRetriever
    gateway: McpToolGateway
    notify: Callable[[dict], Awaitable[None]] | None = None  # e.g. webhook for new complaints



SAFE_FALLBACK = (
    "I'm not able to help with that in a way that fits our support policy. "
    "A colleague can follow up if you'd like - just say 'log a ticket'."
)
ADVICE_REFUSAL = (
    "I can't give investment advice, trading signals, or price predictions, and nobody at Aurelia support can promise "
    "outcomes - trading CFDs carries a high risk of losing money quickly [1]. I can explain our fees, margin and "
    "leverage rules, verification, or how a demo account works if that helps."
)
OUT_OF_SCOPE = (
    "I can only help with Aurelia Markets support: fees, margin and leverage, verification, deposits and withdrawals, "
    "account types, complaints, or the status of your own account. What would you like to know?"
)
NEED_SIGN_IN = "Please sign in first so I can look at your own account. (In this demo, pick a demo customer at the top.)"
BLOCKED_REPLIES = {
    "prompt_injection": "I can't act on instructions that try to change how I work. I'm happy to help with questions about your account or our policies.",
    "empty_message": "Could you type your question? I can help with fees, margin, verification, withdrawals, or your own account.",
    "message_too_long": "That message is too long for me to process. Could you shorten it to the main question?",
}




def _t(state: AgentState, step: str, **info: Any) -> list[dict]:
    return [*(state.get("trace") or []), {"step": step, **info}]


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


def _has_reason(message: str) -> bool:
    return bool(
        re.search(r"\b(because|for|to pay|reason|since|tuition|invoice|purchase|property|medical|rent|moving|leaving|switching)\b", message.lower())
        or len(message.split()) >= 9
    )




def build_graph(deps: Deps, checkpointer: Any):
    llm, retriever, gw = deps.llm, deps.retriever, deps.gateway

    async def guard_in(state: AgentState) -> dict:
        res = check_input(state["message"])
        if res.ok:
            return {"route": "", "trace": _t(state, "input_guard", ok=True)}
        return {
            "route": "blocked",
            "reply": BLOCKED_REPLIES.get(res.reason, SAFE_FALLBACK),
            "sources": [],
            "trace": _t(state, "input_guard", ok=False, reason=res.reason),
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
                if guess in ROUTES and guess not in SENSITIVE_ROUTES and guess != "advice":
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
        prev_user = next((h["content"] for h in reversed(state.get("history") or []) if h["role"] == "user"), None)
        if prev_user and len(message.split()) < 5 and not rewrite:
            query = f"{prev_user} {message}"  # without a rewrite, short follow-ups need the previous question
        found = await _to_thread(retriever.search, query, rewrite)
        trace = _t(state, "retrieve", query=query, rewrite=rewrite, confident=found.confident, best_score=round(found.best_vector_score, 3),
                   hits=[{"source": h.chunk.citation, "rrf": round(h.rrf, 4), "vector": None if h.vector_score is None else round(h.vector_score, 3),
                          "bm25_rank": h.bm25_rank, "vector_rank": h.vector_rank} for h in found.hits])
        if not found.confident:
            return {
                "reply": "I don't have that in the Aurelia Markets documents I can see. I can log a ticket so a colleague can answer - just say 'log a ticket'.",
                "sources": [], "trace": trace,
            }
        sources_txt = "\n".join(f"[{i + 1}] {h.chunk.citation}: {h.chunk.content}" for i, h in enumerate(found.hits))
        user = f"CONVERSATION:\n{_history_text(state.get('history') or [])}\n\nQUESTION: {message}\n\nSOURCES:\n{sources_txt}"
        reply = _clean_citations(await llm.complete("answer", prompts.ANSWER_SYSTEM, user))
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", reply)}  # only show sources the answer actually used
        return {
            "reply": reply,
            "sources": [{"n": i + 1, "source": h.chunk.citation} for i, h in enumerate(found.hits) if (i + 1) in cited],
            "evidence": sources_txt,
            "trace": [*trace, {"step": "answer", "model": llm.name, "cited": sorted(cited)}],
        }

    async def account(state: AgentState) -> dict:
        cid = state.get("customer_id")
        if not cid:
            return {"reply": NEED_SIGN_IN, "sources": [], "trace": _t(state, "account", error="not_signed_in")}
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
            return {"reply": SAFE_FALLBACK, "sources": [], "trace": trace}
        user = f"QUESTION: {state['message']}\n\nTOOL_RESULTS: {json.dumps(results)}"
        reply = await llm.complete("account_reply", prompts.ACCOUNT_SYSTEM, user)
        return {"reply": reply, "sources": [], "evidence": json.dumps(results), "trace": [*trace, {"step": "answer", "model": llm.name}]}

    async def complaint(state: AgentState) -> dict:
        cid = state.get("customer_id")
        if not cid:
            return {
                "reply": "I'm sorry you're having a bad experience. Please sign in so I can log your complaint, or email complaints@aurelia-markets.example and we'll respond within 15 business days.",
                "sources": [], "trace": _t(state, "complaint", error="not_signed_in"),
            }
        summary = state["message"][:400]
        res = await gw.call("create_ticket", cid, {"category": "complaint", "summary": summary, "priority": "high"})
        if "error" in res:
            return {"reply": SAFE_FALLBACK, "sources": [], "trace": _t(state, "complaint", error=res["error"])}
        if deps.notify:
            try:
                await deps.notify({"event": "complaint_logged", "ticket_id": res["ticket_id"], "customer_id": cid, "summary": summary})
            except Exception as exc:
                log.warning("complaint webhook failed: %s", exc)
        return {
            "reply": (
                f"I'm sorry about that. I've logged your complaint as ticket #{res['ticket_id']}. You'll get an acknowledgement within "
                "2 business days and a final written response within 15 business days [1]. If you're unhappy with the response, you can ask "
                "for it to be escalated to the Head of Compliance."
            ),
            "sources": [{"n": 1, "source": "Complaints Policy > What happens next"}],
            "trace": _t(state, "complaint", ticket_id=res["ticket_id"], webhook=bool(deps.notify)),
        }

    async def limit_exception(state: AgentState) -> dict:
        cid = state.get("customer_id")
        if not cid:
            return {"reply": NEED_SIGN_IN, "sources": [], "trace": _t(state, "limit_exception", error="not_signed_in")}
        slot = dict(state.get("slot") or {"intent": "limit_exception", "amount": None, "reason": None})
        msg = state["message"]
        slot["amount"] = slot["amount"] or extract_amount(msg)
        if not slot["reason"] and _has_reason(msg):
            slot["reason"] = msg[:300]
        if not slot["amount"]:
            return {"reply": "Happy to submit that for review. What daily withdrawal limit (in USD) would you like?", "slot": slot, "sources": [],
                    "trace": _t(state, "limit_exception", waiting_for="amount")}
        if not slot["reason"]:
            return {"reply": "Thanks. What is the reason for the higher limit? A compliance reviewer will read it.", "slot": slot, "sources": [],
                    "trace": _t(state, "limit_exception", waiting_for="reason")}

        limits = await gw.call("get_withdrawal_limits", cid)
        kyc = await gw.call("get_kyc_status", cid)
        current, ceiling = limits.get("daily_withdrawal_limit_usd"), TIER_LIMITS.get((kyc.get("kyc_tier") or 0) + 1)
        amount = int(slot["amount"])
        base = _t(state, "policy_precheck", current_limit=current, requested=amount, ceiling=ceiling)
        if ceiling is None or current is None or amount <= current or amount > ceiling:
            tip = f"it must be above your current {current} USD" + (f" and no higher than {ceiling} USD (the next verification tier's limit)" if ceiling else "")
            return {
                "reply": f"I can't send that request for review: {tip} [1]. Completing the next level of verification is usually the faster route - I can tell you what's needed.",
                "sources": [{"n": 1, "source": "Deposits and Withdrawals > Withdrawal limits and exception requests"}],
                "slot": None, "trace": base,
            }
        action = {
            "type": "limit_exception", "customer_id": cid, "requested_daily_limit": amount, "reason": slot["reason"],
            "summary": f"Raise daily withdrawal limit from {current} to {amount} USD for {cid}. Reason: {slot['reason']}",
        }
        return {"slot": None, "pending_action": action, "trace": [*base, {"step": "needs_approval", "action": action["type"]}]}

    async def closure(state: AgentState) -> dict:
        cid = state.get("customer_id")
        if not cid:
            return {"reply": NEED_SIGN_IN, "sources": [], "trace": _t(state, "account_closure", error="not_signed_in")}
        slot = dict(state.get("slot") or {"intent": "account_closure", "reason": None})
        msg = state["message"]
        if not slot.get("reason") and _has_reason(msg) and not re.fullmatch(r"(please )?(close|delete|terminate) (my )?account\.?", msg.lower().strip()):
            slot["reason"] = msg[:300]
        if not slot.get("reason"):
            return {"reply": "I can pass a closure request to our compliance team. Could you tell me briefly why you'd like to close the account?",
                    "slot": slot, "sources": [], "trace": _t(state, "account_closure", waiting_for="reason")}
        summary = await gw.call("get_account_summary", cid)
        if summary.get("has_open_positions"):
            return {"reply": "You still have open positions. Please close them (and withdraw any balance) first, then ask me again [1].",
                    "sources": [{"n": 1, "source": "Deposits and Withdrawals > Account closure"}], "slot": None,
                    "trace": _t(state, "account_closure", blocked="open_positions")}
        action = {"type": "account_closure", "customer_id": cid, "reason": slot["reason"], "summary": f"Close account for {cid}. Reason: {slot['reason']}"}
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
        if not approved:
            note = decision.get("note") or "it didn't meet our policy requirements at this time"
            return {"reply": f"A compliance reviewer looked at your request and could not approve it: {note}. You can ask me about the verification steps that would raise your limits.",
                    "sources": [], "pending_action": None, "trace": trace}
        cid = action["customer_id"]
        if action["type"] == "limit_exception":
            res = await gw.call("submit_limit_exception_request", cid,
                                {"requested_daily_limit": action["requested_daily_limit"], "reason": action["reason"]},
                                approved_by=reviewer, privileged=True)
            ok = res.get("status") == "granted"
            reply = (f"Good news - a compliance reviewer approved your request. Your daily withdrawal limit is now {res['new_daily_limit_usd']} USD."
                     if ok else f"The reviewer approved it, but the system could not apply it ({res.get('error', 'unknown error')}). I've kept your request on file.")
        else:
            res = await gw.call("submit_account_closure_request", cid, {"reason": action["reason"]}, approved_by=reviewer, privileged=True)
            ok = res.get("status") == "closure_pending"
            reply = ("A compliance reviewer approved your request. Your account is now pending closure; you'll receive a confirmation email."
                     if ok else f"The reviewer approved it, but it could not be applied ({res.get('error', 'unknown error')}).")
        return {"reply": reply, "sources": [], "pending_action": None, "trace": [*trace, {"step": "tool_executed", "tool": action["type"], "ok": ok}]}

    async def advice(state: AgentState) -> dict:
        return {"reply": ADVICE_REFUSAL, "sources": [{"n": 1, "source": "Risk Disclosure > No investment advice"}], "trace": _t(state, "advice_refusal")}

    async def out_of_scope(state: AgentState) -> dict:
        return {"reply": OUT_OF_SCOPE, "sources": [], "trace": _t(state, "out_of_scope")}

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
        if n >= 1:  # one rewrite attempt, then a safe fallback - never loop forever
            return {"reply": SAFE_FALLBACK, "sources": [], "violations": [], "revisions": n + 1, "trace": _t(state, "fallback", reason="still_violating_after_revision")}
        problems = ", ".join(state["violations"]) + (f" ({state['critic_reason']})" if state.get("critic_reason") else "")
        evidence = state.get("evidence") or "(none)"
        rewritten = _clean_citations(await llm.complete(
            "rewrite", prompts.REWRITE_SYSTEM, f"PROBLEMS: {problems}\n\nEVIDENCE:\n{evidence}\n\nDRAFT:\n{state['reply']}"))
        return {"reply": rewritten, "revisions": n + 1, "trace": _t(state, "revise", problems=state["violations"])}

    def after_revise(state: AgentState) -> str:
        return "finalize" if state.get("reply") == SAFE_FALLBACK else "critic"

    async def finalize(state: AgentState) -> dict:
        history = [*(state.get("history") or []), {"role": "user", "content": state["message"]}, {"role": "assistant", "content": state["reply"]}]
        return {"history": history[-MAX_HISTORY:], "status": "done", "trace": _t(state, "done")}

    g = StateGraph(AgentState)
    for name, fn in [
        ("guard_in", guard_in), ("route", route), ("faq", faq), ("account", account), ("complaint", complaint),
        ("limit_exception", limit_exception), ("closure", closure), ("approval_gate", approval_gate), ("advice", advice),
        ("out_of_scope", out_of_scope), ("critic", critic), ("revise", revise), ("finalize", finalize),
    ]:
        g.add_node(name, fn)

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

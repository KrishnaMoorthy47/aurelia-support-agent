"""LLM backends: Groq, any LangChain chat model, and a rule-based offline stand-in used by tests."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from app.config import get_settings
from app.rules import account_intents, rule_route

log = logging.getLogger("aurelia.llm")

# Which model handles which job. Routing and the critic are cheap classification
# jobs -> small model. Writing the customer-facing answer -> larger model.
FAST_TASKS = {"route", "critic"}
_THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM(Protocol):
    name: str

    async def complete(self, task: str, system: str, user: str) -> str: ...

    async def select_tools(self, message: str, tool_specs: list[dict]) -> list[dict] | None:
        """Return [{"name": ..., "args": {...}}]. None means 'not supported, use rules'."""
        ...




class ChatModelLLM:
    """Wraps LangChain chat models. Subclassed by GroqLLM."""

    def __init__(self, fast_model: Any, strong_model: Any, name: str):
        self._fast = fast_model
        self._strong = strong_model
        self.name = name
        self.calls = 0
        self.tokens = 0

    def _count(self, msg: Any) -> None:
        self.calls += 1
        usage = getattr(msg, "usage_metadata", None) or {}
        self.tokens += int(usage.get("total_tokens", 0) or 0)

    def _model(self, task: str):
        return self._fast if task in FAST_TASKS else self._strong

    async def complete(self, task: str, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg = await self._model(task).ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        self._count(msg)
        content = msg.content
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return _THINK.sub("", str(content)).strip()

    async def select_tools(self, message: str, tool_specs: list[dict]) -> list[dict] | None:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            bound = self._strong.bind_tools(tool_specs)
            msg = await bound.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You help a broker's support team look up a customer's own account data. "
                            "Pick the tool(s) that answer the customer's question. Never invent arguments."
                        )
                    ),
                    HumanMessage(content=message),
                ]
            )
            self._count(msg)
            return [{"name": c["name"], "args": c.get("args", {})} for c in (msg.tool_calls or [])]
        except Exception as exc:
            log.warning("tool selection failed, falling back to rules: %s", exc)
            return None


class GroqLLM(ChatModelLLM):
    def __init__(self) -> None:
        from langchain_groq import ChatGroq

        s = get_settings()

        def extra(model: str) -> dict:  # only reasoning models accept reasoning_effort
            return {"reasoning_effort": s.reasoning_effort} if model.startswith("openai/gpt-oss") else {}

        super().__init__(
            fast_model=ChatGroq(model=s.fast_model, api_key=s.groq_api_key, temperature=0, max_retries=2,
                                max_tokens=s.fast_max_tokens, **extra(s.fast_model)),
            strong_model=ChatGroq(model=s.strong_model, api_key=s.groq_api_key, temperature=0.1, max_retries=2,
                                  max_tokens=s.strong_max_tokens, **extra(s.strong_model)),
            name=f"groq:{s.fast_model}+{s.strong_model}",
        )


def _langchain_provider(provider: str) -> ChatModelLLM:
    """Anthropic / OpenAI via LangChain. Needs the extra package and a paid key."""
    s = get_settings()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatModelLLM(
            ChatAnthropic(model=s.fast_model, temperature=0), ChatAnthropic(model=s.strong_model, temperature=0.1), "anthropic"
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatModelLLM(
            ChatOpenAI(model=s.fast_model, temperature=0), ChatOpenAI(model=s.strong_model, temperature=0.1), "openai"
        )
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}")




class OfflineLLM:
    """Deterministic, no-network stand-in. Good enough to exercise every code path."""

    name = "offline-rules"
    calls = 0
    tokens = 0

    async def complete(self, task: str, system: str, user: str) -> str:
        self.calls += 1
        if task == "route":
            return json.dumps({"route": rule_route(user)})
        if task == "critic":
            return json.dumps({"verdict": "pass", "reason": "offline critic defers to deterministic rules"})
        if task == "answer":
            return self._extractive_answer(user)
        if task == "account_reply":
            return self._render_tool_results(user)
        return "I can only help with Aurelia Markets support questions."

    async def select_tools(self, message: str, tool_specs: list[dict]) -> list[dict] | None:
        names = {s["function"]["name"] for s in tool_specs}
        mapping = {
            "kyc": "get_kyc_status",
            "limits": "get_withdrawal_limits",
            "tickets": "list_tickets",
            "summary": "get_account_summary",
        }
        return [{"name": mapping[i], "args": {}} for i in account_intents(message) if mapping[i] in names]


    @staticmethod
    def _extractive_answer(user: str) -> str:
        """Pick the source sentence that best overlaps the question (earlier sources win ties)."""
        from app.rag.embedders import terms

        q_match = re.search(r"QUESTION:\s*(.*?)\n\s*SOURCES:", user, re.S)
        q_terms = set(terms(q_match.group(1) if q_match else user)) - {"happen", "happens"}
        best: tuple[float, str, str] | None = None
        for num, body in re.findall(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", user.split("SOURCES:")[-1], re.S):
            title, _, text = body.partition(": ")
            for sent in re.split(r"(?<=[.!?])\s+|\n+", text or body):
                score = len(q_terms & set(terms(sent))) - 0.05 * int(num)
                if sent.strip() and (best is None or score > best[0]):
                    best = (score, sent.strip(), num)
        if not best or best[0] <= 0:
            return "I don't have that in the Aurelia Markets documents I can see."
        return f"{best[1]} [{best[2]}]"

    @staticmethod
    def _render_tool_results(user: str) -> str:
        m = re.search(r"TOOL_RESULTS:\s*(\{.*\}|\[.*\])", user, re.S)
        try:
            results = json.loads(m.group(1)) if m else {}
        except json.JSONDecodeError:
            return "I couldn't read your account details just now."
        lines: list[str] = []
        for tool, data in results.items():
            if isinstance(data, dict) and data.get("error"):
                lines.append(f"- {tool}: {data['error']}")
            elif isinstance(data, dict):
                lines.append("- " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in data.items()))
            elif isinstance(data, list):
                lines.append(f"- {tool}: " + ("; ".join(str(i) for i in data) if data else "nothing on file"))
        return "Here is what I can see on your account:\n" + "\n".join(lines)



_instance: LLM | None = None


def get_llm() -> LLM:
    global _instance
    if _instance is None:
        provider = get_settings().resolved_provider
        if provider == "groq":
            _instance = GroqLLM()
        elif provider == "offline":
            _instance = OfflineLLM()
        else:
            _instance = _langchain_provider(provider)
        log.info("LLM backend: %s", _instance.name)
    return _instance


def set_llm(llm: LLM | None) -> None:
    """Tests inject a stub here."""
    global _instance
    _instance = llm

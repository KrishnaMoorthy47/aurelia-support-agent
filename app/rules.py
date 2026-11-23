"""Deterministic guardrails: injection and advice detection, reply checks and the routing floor.

These run around the model, so they can't be talked out of their decisions.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_INPUT_CHARS = 1500

ROUTES = ("faq", "account", "complaint", "limit_exception", "account_closure", "advice", "out_of_scope")

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)


def normalize(text: str) -> str:
    """Unicode-normalise and lowercase so obfuscated text (full-width letters,
    zero-width joiners) still matches the patterns below."""
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", t).strip().lower()



_INJECTION = [
    r"ignore (all |any |the |your )?(previous|prior|above|earlier|preceding) (instructions|rules|prompts?|messages?)",
    r"disregard (all |any |the |your )?(previous|prior|above|earlier)?\s*(instructions|rules|guidelines|prompts?)",
    r"(reveal|show|print|repeat|output|display|leak|tell me) (me )?(your|the) (system|hidden|initial|original|secret) (prompt|instructions?|message)",
    r"you are now (dan|an? (unrestricted|unfiltered|evil|jailbroken))",
    r"\bdeveloper mode\b|\bjailbreak(ed)?\b|\bdan mode\b|do anything now",
    r"(forget|override|bypass) (your|all|the) (rules|instructions|guidelines|safety|guardrails|restrictions)",
    r"pretend (that )?(you|there) (are|is|have|has) no (rules|restrictions|limits|guardrails)",
    r"act as (if you (have|had) no|an? (unrestricted|unfiltered))",
    r"<\s*/?\s*(system|assistant|developer)\s*>|\[/?inst\]|###\s*(system|instruction)",
]
_INJECTION_RE = [re.compile(p) for p in _INJECTION]


@dataclass
class InputCheck:
    ok: bool
    reason: str = ""


def check_input(text: str) -> InputCheck:
    if not text or not text.strip():
        return InputCheck(False, "empty_message")
    if len(text) > MAX_INPUT_CHARS:
        return InputCheck(False, "message_too_long")
    norm = normalize(text)
    for rx in _INJECTION_RE:
        if rx.search(norm):
            return InputCheck(False, "prompt_injection")
    return InputCheck(True)



_ADVICE_REQUEST = [
    r"\bshould i (buy|sell|short|long|invest|trade|go long|go short|enter|exit|hold|put)\b",
    r"\bwill\b.{0,40}\b(go up|go down|rise|fall|drop|crash|rally|increase|decrease|hit \d|moon)\b",
    r"\b(best|good|hot|top) (trade|stock|pair|crypto|coin|asset|instrument|investment)s?\b.{0,20}\b(to|for|right now|today|this week)\b",
    r"\b(give|share|send|got|have) (me )?(a |any )?(trading |forex |buy |sell )?(signal|tip|prediction|forecast)s?\b",
    r"\bwhat (should|do you recommend|would you recommend|do you suggest) i\b.{0,15}\b(trade|buy|sell|invest)",
    r"\bwhere (should|can) i (invest|put my money)\b",
    r"\bhow much (money )?(will|can|could) i (make|earn|profit)\b",
    r"\bis it (a )?(good|bad|right) (time|moment) to (buy|sell|trade|invest)\b",
    r"\bwhich (stock|pair|coin|crypto|asset|currency)s? (should|to|will)\b",
    r"\bpredict(ion)?s?\b.{0,30}\b(price|market|gold|oil|bitcoin|eur|usd|gbp)\b",
    r"\b(guarantee[sd]?|sure.?shot|risk.?free)\b.{0,30}\b(profit|returns?|win|gains?)\b",
]
_ADVICE_REQUEST_RE = [re.compile(p) for p in _ADVICE_REQUEST]


def is_advice_request(text: str) -> bool:
    norm = normalize(text)
    return any(rx.search(norm) for rx in _ADVICE_REQUEST_RE)


# Output side: things the *assistant* must never say. Negated phrasing
# ("we do not guarantee profits") is allowed.
_REPLY_VIOLATIONS: dict[str, re.Pattern] = {
    "guarantee_of_returns": re.compile(r"\bguarantee[sd]?\b.{0,30}\b(profit|returns?|gains?|win|income)\b"),
    "risk_free_claim": re.compile(r"\brisk.?free\b"),
    "promise_of_earnings": re.compile(r"\byou(?:'ll| will)\s+(?:definitely |certainly |surely )?(make|earn|profit|win|get rich)\b"),
    "buy_sell_recommendation": re.compile(
        r"\b(i|we)\s+(recommend|suggest|advise)\s+(that\s+)?you\s+(buy|sell|trade|invest|go long|go short)\b"
    ),
    "should_buy_sell": re.compile(r"\byou should (buy|sell|short|go long|go short|invest in|open a (long|short))\b"),
    "price_prediction": re.compile(
        r"\b(price|market|gold|oil|bitcoin|btc|eur/?usd|gbp/?usd|pair|stock|index|nasdaq)\b[^.!?]{0,40}\b"
        r"(will|is going to|is likely to|should)\s+(\w+\s+)?(rise|go up|increase|climb|rally|fall|drop|go down|crash)"
    ),
    # "can't lose more than your deposit" is negative-balance protection (a fact), not a promise
    "cant_lose": re.compile(r"\b(can'?t|cannot) lose\b(?!\s+more than)|\bsure (thing|profit|win)\b"),
}
_NEGATION = re.compile(r"\b(not|no|never|cannot|can't|cant|don't|doesn't|do not|does not|without|nor)\b|n't\b")


def reply_violations(text: str) -> list[str]:
    norm = normalize(text)
    found: list[str] = []
    for name, rx in _REPLY_VIOLATIONS.items():
        for m in rx.finditer(norm):
            window = norm[max(0, m.start() - 35) : m.start()]
            if _NEGATION.search(window):
                continue  # "we do not guarantee profits" is fine
            found.append(name)
            break
    return found



_INFO_Q = re.compile(r"^(how|what|where|when|who|which|why|do you|does|is there|are there|can i|could i|is it)\b")
_STRONG_GRIEVANCE = re.compile(
    r"\b(unhappy|unacceptable|terrible|furious|angry|scam|fraud(ulent)?|ridiculous|disgusted|appalled|useless|worst)\b"
)
_COMPLAINT = re.compile(
    r"\b(i (want|would like|'d like|wish) to (complain|file a complaint|lodge a complaint)|"
    r"my complaint|i am complaining|i'm complaining|file a complaint|lodge a complaint|"
    r"speak to (a |the )?(manager|supervisor)|escalate (this|my))\b"
)
_LIMIT_EXC = re.compile(
    r"(increase|raise|lift|extend|exceed|higher|bigger|more).{0,45}\b(limit|cap)\b|"
    r"\b(limit|cap)\b.{0,45}(increase|raise|exception|higher|lifted)|limit exception"
)
_CLOSURE = re.compile(r"\b(close|delete|terminate|shut down|cancel) (my )?(trading )?account\b|account closure")
_ACCOUNT = re.compile(
    r"\bmy (account|kyc|verification|documents?|withdrawals?|deposits?|limits?|tickets?|status|profile|application)\b|"
    r"\bam i (verified|approved)\b|\bstatus of my\b|\bwhy (is|was) my\b"
)
_FAQ = re.compile(
    r"\b(fee|fees|spread|spreads|commission|swap|rollover|leverage|margin|stop.?out|margin call|deposit|withdraw|withdrawal|"
    r"kyc|verify|verification|account type|islamic|swap.?free|complain|complaint|ombudsman|platform|trading hours|"
    r"support hours|inactiv\w+|risk|cfd|currency conversion|minimum|negative balance|raw account|standard account|"
    r"proof of address|source of funds|processing time|aurelia)\b"
)


def is_info_question(text: str) -> bool:
    """Looks like a general question ("what is...", "how do I...") rather than an answer to one."""
    return bool(_INFO_Q.match(normalize(text)))


def rule_route(text: str) -> str:
    """Deterministic router. Used as the offline router and as a safety floor
    under the LLM router (advice always wins)."""
    t = normalize(text)
    if is_advice_request(t):
        return "advice"
    if _CLOSURE.search(t):
        return "account_closure"
    if _LIMIT_EXC.search(t):
        return "limit_exception"
    info_q = bool(_INFO_Q.match(t))
    if _STRONG_GRIEVANCE.search(t) or (_COMPLAINT.search(t) and not info_q):
        return "complaint"
    if _ACCOUNT.search(t):
        return "account"
    if _FAQ.search(t):
        return "faq"
    return "out_of_scope"


def account_intents(text: str) -> list[str]:
    """Which account facts the customer is asking about (deterministic fallback
    when the LLM's own tool selection is unavailable)."""
    t = normalize(text)
    intents: list[str] = []
    if re.search(r"\b(kyc|verif\w+|documents?|proof|identity|id)\b", t):
        intents.append("kyc")
    if re.search(r"\b(limit|limits|withdraw\w*|cap)\b", t):
        intents.append("limits")
    if re.search(r"\b(ticket|tickets|case|complaint|request)s?\b", t):
        intents.append("tickets")
    if not intents or re.search(r"\b(account|status|profile)\b", t):
        intents.append("summary")
    return list(dict.fromkeys(intents))


_AMOUNT = re.compile(r"(?:\$|usd\s*|aed\s*)?\b(\d{1,3}(?:,\d{3})+|\d{3,7}|\d{1,3}(?=\s*k\b))(?:\.\d+)?\s*(k\b)?", re.IGNORECASE)


def extract_amount(text: str) -> float | None:
    m = _AMOUNT.search(text.replace("$ ", "$"))
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    if m.group(2):
        value *= 1000
    return value

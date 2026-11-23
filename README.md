# Aurelia Support Agent

A customer-support agent for a fictional CFD broker, "Aurelia Markets". It answers policy questions from the broker's documents with citations, looks up the signed-in customer's account through an MCP server, refuses investment advice, and sends anything that changes an account to a human reviewer.

All company data is invented. Nothing here is financial advice.

**Stack:** LangGraph · Groq (`qwen3.8-27b` for routing and review, `gpt-oss-120b` for answers) · Supabase pgvector · fastembed (`bge-small-en-v1.5`) · MCP Python SDK · FastAPI · Docker

## What it does

| Customer says | What happens |
|---|---|
| "What's the withdrawal fee?" | Multi-query hybrid retrieval over the policy documents; the answer cites its sources, or says the documents don't cover it |
| "Why is my verification still pending?" | The model picks read-only tools that run on an MCP server; the customer id comes from the session, not the model |
| "Should I buy gold?" | Refused by a deterministic rule, with the risk disclosure cited |
| "Ignore previous instructions…" | Blocked before any model sees it |
| "I want to complain…" | Logged as a ticket immediately, and a webhook fires |
| "Raise my withdrawal limit to 5000" | Collects amount and reason → policy check → waits for a reviewer → applied over MCP (the server also refuses without a reviewer id) |

Every reply goes through a critic: deterministic rules plus a small-model check that sees the same evidence the reply was written from. A failing draft gets one rewrite, then a safe fallback.

## Architecture

```
message → input guard → router (rules + LLM; also rewrites the question for search)
            ├─ faq ─────────► multi-query hybrid RAG → grounded answer with citations
            ├─ account ─────► MCP tools (read-only)
            ├─ complaint ───► ticket + webhook
            ├─ limit / closure ► policy check → human approval (LangGraph interrupt) → MCP write tool
            ├─ advice ──────► fixed refusal
            └─ off topic ───► redirect
        → critic ⇄ rewrite → reply + step-by-step trace
```

Safety-critical decisions (injection, advice, whose account is visible, who approves what) are plain code. The model writes text.

## Retrieval

- **Chunking:** one chunk per `##` section (37 chunks, 20–75 words), each prefixed with its document title so it can be cited.
- **Query rewriting:** the router call that classifies the message also returns `search_query`, the question restated in the documents' vocabulary ("I want to leave Aurelia for good" → "account closure"). This adds no extra model calls.
- **Fusion:** Reciprocal Rank Fusion over three lists: vector search on the original question, vector search on the rewrite, and BM25 on the rewrite.
- **Gate:** the answer step only runs if the best cosine similarity is at least 0.55. In-domain questions score 0.60 or more; unrelated ones score 0.42–0.52.

`eval/retrieval_bench.py` compares the options on 35 labelled questions (15 using the documents' wording, 20 paraphrased in a customer's words):

| Strategy | Keyword hit@3 | Paraphrase hit@1 | Paraphrase hit@3 |
|---|---|---|---|
| BM25 | 100% | 30% | 50% |
| Vector | 100% | 55% | 75% |
| Vector + BM25 with RRF (no rewrite) | 100% | 50% | 65% |
| Vector on the rewrite | 100% | 95% | 95% |
| **Multi-query RRF (used)** | **100%** | **85%** | **100%** |

Plain hybrid search was *worse* than vectors alone on paraphrases, because BM25 on a customer's wording matches incidental words. BM25 becomes useful once it runs on the rewritten query, and fusing it with the original question protects against a bad rewrite.

## Evaluation

`eval/run_eval.py` runs 49 scripted conversations (56 turns) through the real graph and scores routing, required facts, citations, compliance (on every turn), the approval flow, latency and token use.

| Run | Passed | Notes |
|---|---|---|
| First run on Groq | 34 / 49 | gpt-oss's citation format dropped sources; the critic couldn't see its evidence and rejected correct account answers |
| After grounding the critic and normalising citations | 49 / 49 | |
| On Supabase pgvector, multi-query retrieval | 49 / 49 | 0 compliance violations, median 1.2 s per message, about 1.7 model calls per message |

Run histories are in `eval/runs/`. These cases were written alongside the system, so they work as a regression suite rather than an independent benchmark.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # add GROQ_API_KEY (free at console.groq.com)
uvicorn app.api:app --reload                            # http://localhost:8000
```

Without `GROQ_API_KEY` it runs in offline mode with a rule-based stand-in for the model. Set `EMBEDDER=lexical` to skip the 130 MB embedding model download.

Demo customers: `C-1001` (verified, has open positions), `C-1002` (proof of address missing), `C-1003` (restricted), `C-1004` (Tier 1, use for limit requests).

### Supabase

1. Create a free project and copy the **Session pooler** connection string (Connect → Direct).
2. Set `DATABASE_URL=…` and `VECTOR_STORE=pgvector` in `.env`.

The tables, indexes and `vector` extension are created on first start (`supabase/schema.sql`, all `if not exists`), and the documents are embedded automatically if the table is empty. Free projects pause after a week without activity.

### Request log

With `DATABASE_URL` set, every POST (chat messages, reviewer decisions, resets) and every error response is written to the `api_logs` table: timestamp, request id (also returned as the `X-Request-ID` header), path, status, latency, session, customer, route, and the full request and response bodies including the agent's step trace. Headers are never stored. Writes happen in the background, so a logging failure never affects the reply. The UI's polling GETs aren't logged.

```sql
select created_at, route, request->>'message' as message, response->>'reply' as reply, latency_ms
from api_logs order by created_at desc limit 20;
```

## Tests

```bash
pytest                                  # 97 tests, offline, no keys needed
python eval/run_eval.py                 # full conversation eval against Groq
python eval/retrieval_bench.py --rewrite
```

The tests cover the guardrails, retrieval, the MCP server and its guards, every graph path (approval, decline, rewrite, fallback, a misbehaving model), and the API. CI runs them on every push and builds the Docker image.

## Notes

- Groq's free tier allows 1,000 requests a day and 8,000 tokens a minute per model, and rejects requests that could produce more than about 1,000 output tokens a minute, so output is capped (`FAST_MAX_TOKENS`, `STRONG_MAX_TOKENS`). `DEMO_DAILY_CAP` limits total messages a day.
- Switching to Claude or OpenAI is `LLM_PROVIDER=anthropic|openai` plus the matching LangChain package. This hasn't been run, since neither has a free tier.
- Approval state is kept in memory. `CHECKPOINTER=postgres` persists paused conversations on Linux (async psycopg and the MCP subprocess need different event loops on Windows).

## Layout

```
app/
  api.py           FastAPI: /chat, /approvals, /sessions, /eval/report, /demo/reset, /health
  agents/          LangGraph graph and prompts
  rag/             chunking, embedders, vector stores, retriever, ingest
  mcp_server/      MCP server and mock back-office data
  tools.py         MCP client gateway
  audit.py         request/response log (Supabase api_logs)
  rules.py         guardrails and routing floor
  llm.py           model backends
  static/          demo page
  kb/docs/         the fictional broker's documents
eval/              conversation eval, retrieval benchmark, run history
supabase/          pgvector schema
tests/
```

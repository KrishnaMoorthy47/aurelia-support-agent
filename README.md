# Aurelia Support Agent

Customer-support agent for a fictional CFD broker serving UAE clients. It answers policy questions from the broker's documents with citations, in English or Arabic, reads the signed-in customer's account through an MCP server, refuses investment advice, and holds account changes for human approval.

**Live demo:** https://aurelia-support-agent.onrender.com (free instance; the first request after idling takes about a minute)

All company data is invented. Nothing here is financial advice.

**Stack:** LangGraph · Groq · Supabase pgvector · fastembed · MCP · FastAPI · Docker

## Features

| Request | Behaviour |
|---|---|
| Policy question ("What's the withdrawal fee?") | Retrieval over the policy documents; cited answer, or "not covered" |
| Own account ("Why is my verification pending?") | Read-only MCP tools; customer id comes from the session, not the model |
| Investment advice ("Should I buy gold?") | Refused by rule |
| Prompt injection | Blocked before any model call |
| Complaint | Ticket logged immediately; webhook fired |
| Limit increase / account closure | Details collected → policy check → reviewer approval → applied over MCP |
| UAE question ("Can I verify with my Emirates ID?") | Answered from the UAE documents: regulator, client money, onboarding, data privacy |
| Arabic message ("ما هي رسوم السحب؟") | Same pipeline; searched in English, answered in Arabic with citations |

## Architecture

```
                         ┌──────────────┐
   message ──► guard ──► │    router    │  rules + LLM, also rewrites the query for search
                         └──────┬───────┘
      ┌──────────┬──────────┬───┴──────┬────────────────┬──────────┐
      ▼          ▼          ▼          ▼                ▼          ▼
     faq      account   complaint   limit / closure   advice   off-topic
   (RAG)    (MCP read)  (ticket)    policy check     (refuse)  (redirect)
      │          │          │          ▼                │          │
      │          │          │    human approval         │          │
      │          │          │    (MCP write)            │          │
      └──────────┴──────────┴──────────┼────────────────┴──────────┘
                                       ▼
                         critic ⇄ rewrite ──► reply + trace
```

- Safety decisions (injection, advice, account access, approvals) are plain code; the model only writes text.
- The critic checks every reply with rules and a small model that sees the same evidence the reply was written from. A failing draft gets one rewrite, then a safe fallback.
- Fixed replies (refusals, approval messages, holding replies) are written by hand in English and Arabic (`app/i18n.py`), so they read the same every time.
- Models: `qwen/qwen3.8-27b` for routing and review, `openai/gpt-oss-120b` for answers. Switching to Claude or OpenAI is a config change (`LLM_PROVIDER`).

## Retrieval

- One chunk per document section (57 chunks across 12 documents), embedded with `BAAI/bge-small-en-v1.5` (384-dim, local ONNX via fastembed) and stored in pgvector.
- The router also returns `search_query`, the question in the documents' vocabulary. This costs no extra call.
- Reciprocal Rank Fusion over vector search on the original question, vector search on the rewrite, and BM25 on the rewrite.
- The answer step runs only if the best cosine similarity is at least 0.55.
- On startup only new or edited sections are embedded, and removed sections are deleted, so editing a document never needs a manual re-ingest.

Benchmark (`eval/retrieval_bench.py`, 35 labelled questions):

| Strategy | Keyword hit@3 | Paraphrase hit@1 | Paraphrase hit@3 |
|---|---|---|---|
| BM25 | 100% | 30% | 50% |
| Vector | 100% | 55% | 75% |
| Vector + BM25 (RRF) | 100% | 50% | 65% |
| **Multi-query RRF (used)** | **100%** | **85%** | **100%** |

BM25 on a customer's own wording adds noise; on the rewritten query it adds precision.

## UAE knowledge base

Three documents describe how a UAE-facing broker works (`app/kb/docs/09`–`11`):

| Document | Covers |
|---|---|
| UAE Regulation and Client Money | Capital Market Authority (CMA, formerly SCA, renamed 1 January 2026), DIFC/DFSA and ADGM/FSRA, segregated client money, retail vs professional clients, AED deposits |
| UAE Account Opening and Verification | Emirates ID and UAE Pass, accepted proofs of address, Arabic name spellings, source of funds under Federal Decree-Law No. 10 of 2025 (AML) |
| UAE Risk Warning, Data Privacy and Language | Retail loss percentage, UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021), record keeping, Arabic support, complaints to the CMA |

The regulators and laws are real; Aurelia's licence, figures and policies are invented.

## Arabic

- A message is treated as Arabic when most of its letters are Arabic.
- The router writes the English search query, so Arabic questions are matched against the English documents with the same retrieval. The answer, account and rewrite steps are told to reply in Arabic and keep numbers, currency codes and citations unchanged.
- Rule-based checks cover Arabic too: prompt injection, advice requests, limit and closure requests, complaints, and the reply check for guarantees, recommendations and price predictions. Text is normalised first (vowel marks, alef forms, taa marbuta, Arabic-Indic digits).
- The English rules cover advice requests well, so the model may not add refusals there. For Arabic the rules are thinner, so the model may send a message to the advice refusal, which is always safe. It still can't start a limit change or closure.

## Evaluation

`eval/run_eval.py` runs 65 scripted conversations through the graph and scores routing, facts, citations, compliance, the approval flow, latency and tokens.

| Run | Passed |
|---|---|
| Initial | 34 / 49 |
| Critic grounded on evidence, citation format fixed | 49 / 49 |
| Supabase + multi-query retrieval | 49 / 49, 0 compliance violations, ~1.2 s median |

The runs above cover the original 49 conversations. Sixteen UAE and Arabic conversations (categories `uae` and `arabic`) were added afterwards: `python eval/run_eval.py --only uae` and `--only arabic`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set GROQ_API_KEY and DATABASE_URL
uvicorn app.api:app --reload                           # http://localhost:8000
```

Without `GROQ_API_KEY` it runs offline with a rule-based model stand-in. Without `DATABASE_URL` it uses in-memory vectors and no request log.

Demo customers: `C-1001` verified with open positions · `C-1002` proof of address missing · `C-1003` restricted · `C-1004` Tier 1.

## Supabase

Set `DATABASE_URL` to the project's **Session pooler** connection string and `VECTOR_STORE=pgvector`. On startup the app applies `supabase/schema.sql` (idempotent) and embeds any document section that is new or has changed.

| Table | Contents |
|---|---|
| `kb_chunks` | document chunks and embeddings |
| `api_logs` | every POST and error response: timestamp, request id, latency, session, route, request and response bodies |

Both tables have RLS enabled with no policies, so only the app's database role can access them.

## Tests

```bash
pytest                                     # 147 tests, offline
python eval/run_eval.py                    # conversation eval (uses Groq)
python eval/retrieval_bench.py --rewrite   # retrieval benchmark
```

CI runs the tests and builds the Docker image on every push.

## Layout

```
app/
  api.py          FastAPI endpoints and request logging middleware
  agents/         LangGraph graph and prompts
  rag/            chunking, embeddings, vector stores, retriever
  mcp_server/     MCP server and mock back-office data
  tools.py        MCP client gateway
  rules.py        guardrails and routing rules (English and Arabic)
  i18n.py         language detection and fixed replies in English and Arabic
  audit.py        request log
  llm.py          model backends
  static/         demo UI
  kb/docs/        knowledge base
eval/             conversation eval, retrieval benchmark, results
supabase/         database schema
tests/
```

-- Applied automatically on startup (every statement is idempotent).
-- Supabase ships the pgvector extension; this switches it on.
create extension if not exists vector;

create table if not exists kb_chunks (
  id        text primary key,
  doc       text not null,
  title     text not null,
  section   text not null,
  content   text not null,
  embedder  text not null,           -- which model produced the vector; never compare across models
  embedding vector(384) not null
);

-- Approximate nearest-neighbour index for cosine distance.
create index if not exists kb_chunks_embedding_idx
  on kb_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists kb_chunks_embedder_idx on kb_chunks (embedder);

-- Request/response audit log written by the API (app/audit.py).
create table if not exists api_logs (
  id          bigserial primary key,
  created_at  timestamptz not null default now(),
  request_id  uuid not null,
  method      text not null,
  path        text not null,
  status_code int not null,
  latency_ms  int not null,
  session_id  text,
  customer_id text,
  route       text,
  request     jsonb,
  response    jsonb
);

alter table api_logs add column if not exists started_at timestamptz;
alter table api_logs add column if not exists query text;
alter table api_logs add column if not exists user_agent text;

create index if not exists api_logs_created_at_idx on api_logs (created_at desc);
create index if not exists api_logs_session_idx on api_logs (session_id);
create index if not exists api_logs_request_idx on api_logs (request_id);
create index if not exists api_logs_path_idx on api_logs (path, created_at desc);

-- Keep 30 days of request logs so the free database doesn't fill up.
delete from api_logs where created_at < now() - interval '30 days';

-- No policies: only the database owner (the app) can read or write; the public Data API can't.
alter table kb_chunks enable row level security;
alter table api_logs enable row level security;

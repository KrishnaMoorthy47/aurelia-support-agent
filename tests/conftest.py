import asyncio
import os
import sys

# Tests never depend on the developer's .env: no network, no API keys, no database.
# (Environment variables take precedence over .env in pydantic-settings.)
for _k, _v in {"LLM_PROVIDER": "offline", "EMBEDDER": "lexical", "VECTOR_STORE": "memory", "CHECKPOINTER": "memory",
               "AUDIT_LOG": "memory", "API_KEY": "", "REVIEWER_KEY": "", "ESCALATION_WEBHOOK_URL": "", "DEMO_DAILY_CAP": "300"}.items():
    os.environ[_k] = _v

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.llm import OfflineLLM  # noqa: E402
from app.rag.embedders import LexicalEmbedder  # noqa: E402
from app.rag.ingest import build_retriever  # noqa: E402
from app.rag.stores import MemoryStore  # noqa: E402


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="session")
def retriever():
    return build_retriever(MemoryStore(), LexicalEmbedder())


@pytest.fixture
def offline_llm():
    return OfflineLLM()


@pytest.fixture
def mock_db(tmp_path):
    return str(tmp_path / "mock.db")

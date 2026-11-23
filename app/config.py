"""Settings, read from environment variables / .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    llm_provider: str = "auto"  # auto | groq | offline
    fast_model: str = "qwen/qwen3.8-27b"
    strong_model: str = "openai/gpt-oss-120b"
    # Groq's free tier rejects requests whose possible output exceeds ~1000 tokens/min, so always cap output.
    fast_max_tokens: int = 150
    strong_max_tokens: int = 800
    reasoning_effort: str = "low"

    embedder: str = "auto"  # auto | fastembed | lexical
    vector_store: str = "memory"  # memory | pgvector
    database_url: str = ""
    checkpointer: str = "memory"
    audit_log: str = "auto"  # auto (postgres if DATABASE_URL is set) | postgres | memory | off  # memory | postgres

    api_key: str = ""
    reviewer_key: str = ""
    cors_origins: str = "*"
    rate_limit_per_minute: int = 20
    demo_daily_cap: int = 300
    escalation_webhook_url: str = ""
    mock_db_path: str = "data/aurelia_mock.db"

    @property
    def resolved_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "groq" if self.groq_api_key else "offline"


@lru_cache
def get_settings() -> Settings:
    return Settings()

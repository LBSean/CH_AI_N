from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str

    # ── LiteLLM ───────────────────────────────────────────────────────────────
    # mode: "proxy" = external LiteLLM container (dev default)
    #       "library" = litellm imported directly (production, no network hop)
    litellm_mode: str = "proxy"
    litellm_base_url: str = "http://litellm:4000/v1"
    litellm_master_key: str

    # ── Models ────────────────────────────────────────────────────────────────
    primary_model: str = "gpt-4o"
    fallback_model: str = "claude-sonnet-4-6"
    fast_model: str = "gpt-4o-mini"          # used for summarisation, validation retries
    embedding_model: str = "text-embedding-3-small"

    # ── Auth (JWT) ────────────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "ai-workspace"

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = "development"
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = {"env_file": ".env"}

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()

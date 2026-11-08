from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BOT_TOKEN: str = Field(default="test_bot_token", description="Telegram Bot API Token")
    LLM_BASE_URL: str = Field(default="http://localhost:11434/v1", description="OpenAI-compatible LLM base URL")
    LLM_API_KEY: str = Field(default="ollama", description="API Key for LLM provider")
    LLM_MODEL: str = Field(default="qwen2.5:7b", description="Model name/identifier")
    CHROMA_PERSIST_DIR: str = Field(default="./storage/chroma", description="Directory to persist ChromaDB vectors")
    SQLITE_DB_PATH: str = Field(default="./storage/leads.db", description="Path to SQLite database")
    TOP_K_RESULTS: int = Field(default=3, description="Number of context chunks to retrieve from RAG")
    HISTORY_LIMIT: int = Field(default=10, description="Max messages per user stored in memory (sliding window)")
    MAX_TOOL_TURNS: int = Field(default=3, description="Max agentic loop iterations before forcing a text response")
    ADMIN_USER_IDS: list[int] = Field(default=[], description="Telegram user IDs with access to admin commands")

    # RAG chunking tuning
    RAG_MAX_CHUNK_SIZE: int = Field(default=1200, description="Max character length of a single RAG chunk")
    RAG_OVERLAP_SIZE: int = Field(default=150, description="Overlap in characters between consecutive RAG chunks")
    RAG_DISTANCE_THRESHOLD: float = Field(default=1.2, description="Distance threshold for RAG retrieval")

    # Healthcheck HTTP server & Ollama
    HEALTH_HOST: str = Field(default="0.0.0.0", description="Healthcheck server bind host")
    HEALTH_PORT: int = Field(default=8080, description="Healthcheck server bind port")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Base URL for Ollama API")

    # amoCRM Integration
    AMO_CRM_ENABLED: bool = Field(default=False, description="Enable amoCRM integration")
    AMO_CRM_SUBDOMAIN: str = Field(default="", description="amoCRM subdomain")
    AMO_CRM_ACCESS_TOKEN: str = Field(default="", description="amoCRM access token (OAuth/Long-lived)")
    AMO_CRM_FIELD_VISIT_TIME_ID: int = Field(default=101, description="amoCRM custom field ID for visit time")
    AMO_CRM_FIELD_SERVICE_ID: int = Field(default=102, description="amoCRM custom field ID for service")
    AMO_CRM_STATUS_CONFIRMED_ID: int = Field(default=103, description="amoCRM status ID for confirmed lead")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

from typing import Literal
from pydantic import BaseModel, SecretStr, field_validator ,Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class LLMSettings(BaseModel):
    groq_api_key: SecretStr
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: SecretStr
    gemini_model: str = "gemini-1.5-flash"
    judge_provider: Literal["groq", "gemini"] = "gemini"


class VectorDBSettings(BaseModel):
    host: str = "qdrant"
    port: int = 6333
    collection_name: str = "financial_reports"


class DatabaseSettings(BaseModel):
    url: SecretStr


class RedisSettings(BaseModel):
    host: str = "redis"
    port: int = 6379
    cache_db: int = 0
    celery_db: int = 1


class MCPSettings(BaseModel):
    server_url: str
    auth_token: SecretStr


class SecuritySettings(BaseModel):
    bearer_token: SecretStr
    rate_limit_per_minute: int = 60


class MonitoringSettings(BaseModel):
    langsmith_api_key: SecretStr
    langsmith_project: str = "investor-intelligence-v2"
    prometheus_enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    database: DatabaseSettings
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mcp: MCPSettings
    security: SecuritySettings
    monitoring: MonitoringSettings

    @field_validator("app")
    @classmethod
    def validate_environment(cls, v: AppSettings) -> AppSettings:
        return v


settings = Settings()
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Quran Studio AI Alignment Engine"
    environment: str = "development"

    api_key: str = Field(
        default="development-placeholder-key",
        alias="ALIGNMENT_API_KEY",
    )

    maximum_upload_mb: int = Field(
        default=50,
        alias="MAXIMUM_UPLOAD_MB",
    )

    temporary_directory: str = Field(
        default="/tmp/quran-studio",
        alias="TEMPORARY_DIRECTORY",
    )

    model_name: str = Field(
        default="small",
        alias="WHISPER_MODEL_NAME",
    )

    model_loaded: bool = False
    corpus_loaded: bool = False
    engine_version: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
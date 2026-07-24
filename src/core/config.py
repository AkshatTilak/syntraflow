"""Configuration settings module for SyntraFlow.

This module uses pydantic-settings to load, validate, and type environment
variables from a local .env file.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """System settings for SyntraFlow.

    Validates presence of required variables and sets defaults for optional ones.
    """

    # App Settings
    app_env: str = Field(default="development", alias="APP_ENV")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")

    # Database Settings
    database_url: str = Field(..., alias="DATABASE_URL")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # API Keys
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Spark Configs
    spark_master: str = Field(default="local[*]", alias="SPARK_MASTER")
    spark_driver_memory: str = Field(default="2g", alias="SPARK_DRIVER_MEMORY")

    # MLflow Configs
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")

    class Config:
        """Pydantic model configuration configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Discard unmapped variables


settings = Settings()

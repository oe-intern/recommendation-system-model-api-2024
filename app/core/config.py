import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pydantic import Field
import logging


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="UTF-8",
        env_nested_delimiter="__",
        env_prefix="",
    )

    # Backend information
    DEBUG: bool = Field(default=False)
    ENABLE_OPENAPI: bool = Field(default=False)
    HOST: str = Field(default="localhost")
    PORT: int = Field(default=8080)
    WORKERS_COUNT: int = Field(default=1)
    # CORS information
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    CORS_ALLOW_ORIGIN: List[str] = Field(default=["*"])
    # Project information
    PROJECT_NAME: str = Field(default="Recommendation System Model API")
    API_PREFIX: str = Field(default="")
    # Logging information
    LOG_LEVEL: int = Field(logging.WARNING)
    LOG_FORMAT_EXTENDED: bool = Field(default=False)
    # Server mail infomation
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_PORT: int
    MAIL_SERVER: str
    MAIL_FROM_NAME: str
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    USE_CREDENTIALS: bool = True
    # AWS S3 information
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str
    AWS_BUCKET_NAME: str
    # Redis information
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str = Field(default="")
    REDIS_DB: int = Field(default=0)
    REDIS_EXPIRE: int = Field(default=3600)
    REDIS_MAX_CONNECTIONS: int = Field(default=10)
    # Logging information
    LOG_LEVEL: int = Field(default=10)
    # Celery information
    CELERY_BROKER_URL: str = Field(default=f"redis:${REDIS_HOST}:${REDIS_PORT}/0")
    CELERY_RESULT_BACKEND: str = Field(default=f"redis:${REDIS_HOST}:${REDIS_PORT}/1")


settings = Settings()

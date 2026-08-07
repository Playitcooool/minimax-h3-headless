"""Environment-driven settings with safe network defaults."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="H3_", env_file=".env", extra="ignore")

    gateway_host: str = "127.0.0.1"
    gateway_port: int = Field(default=8080, ge=1, le=65535)
    gateway_api_key: str = Field(default="", min_length=0)
    backend: Literal["sglang", "vllm_omni"] = "sglang"
    fl2va_url: str = "http://127.0.0.1:30010"
    ref2va_url: str = "http://127.0.0.1:30011"
    request_timeout_seconds: float = Field(default=30, gt=0)
    poll_interval_seconds: float = Field(default=2, gt=0)
    job_timeout_seconds: float = Field(default=3600, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

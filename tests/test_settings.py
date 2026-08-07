"""Tests for environment-driven gateway configuration."""

import pytest
from pydantic import ValidationError

from minimax_h3_headless.settings import Settings, get_settings


H3_ENV_VARS = {
    "H3_GATEWAY_HOST",
    "H3_GATEWAY_PORT",
    "H3_GATEWAY_API_KEY",
    "H3_BACKEND",
    "H3_FL2VA_URL",
    "H3_REF2VA_URL",
    "H3_REQUEST_TIMEOUT_SECONDS",
    "H3_POLL_INTERVAL_SECONDS",
    "H3_JOB_TIMEOUT_SECONDS",
}


@pytest.fixture(autouse=True)
def isolate_h3_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in H3_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.gateway_host == "127.0.0.1"
    assert settings.gateway_port == 8080
    assert settings.backend == "sglang"
    assert settings.fl2va_url == "http://127.0.0.1:30010"
    assert settings.ref2va_url == "http://127.0.0.1:30011"
    assert settings.request_timeout_seconds == 30
    assert settings.poll_interval_seconds == 2
    assert settings.job_timeout_seconds == 3600


def test_all_documented_environment_variables_map_and_coerce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "H3_GATEWAY_HOST": "0.0.0.0",
        "H3_GATEWAY_PORT": "9000",
        "H3_GATEWAY_API_KEY": "secret",
        "H3_BACKEND": "vllm_omni",
        "H3_FL2VA_URL": "http://fl2va.internal:30010",
        "H3_REF2VA_URL": "http://ref2va.internal:30011",
        "H3_REQUEST_TIMEOUT_SECONDS": "1.5",
        "H3_POLL_INTERVAL_SECONDS": "0.25",
        "H3_JOB_TIMEOUT_SECONDS": "77",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.gateway_host == "0.0.0.0"
    assert settings.gateway_port == 9000
    assert settings.gateway_api_key == "secret"
    assert settings.backend == "vllm_omni"
    assert settings.fl2va_url == "http://fl2va.internal:30010"
    assert settings.ref2va_url == "http://ref2va.internal:30011"
    assert settings.request_timeout_seconds == 1.5
    assert settings.poll_interval_seconds == 0.25
    assert settings.job_timeout_seconds == 77


@pytest.mark.parametrize("port", ["0", "65536"])
def test_gateway_port_must_be_valid(
    monkeypatch: pytest.MonkeyPatch, port: str
) -> None:
    monkeypatch.setenv("H3_GATEWAY_PORT", port)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "name",
    [
        "H3_REQUEST_TIMEOUT_SECONDS",
        "H3_POLL_INTERVAL_SECONDS",
        "H3_JOB_TIMEOUT_SECONDS",
    ],
)
def test_timeouts_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_backend_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("H3_BACKEND", "unknown")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("H3_GATEWAY_PORT", "9000")
    first = get_settings()
    monkeypatch.setenv("H3_GATEWAY_PORT", "9001")

    assert get_settings() is first
    assert get_settings().gateway_port == 9000


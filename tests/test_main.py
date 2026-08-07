"""Tests for the installed ``h3-gateway`` entry point."""

from unittest.mock import patch

from minimax_h3_headless.main import run
from minimax_h3_headless.settings import Settings


def test_run_starts_uvicorn_with_configured_listener_and_safe_proxy_setting() -> None:
    configured = Settings(
        _env_file=None,
        gateway_host="0.0.0.0",
        gateway_port=9876,
        gateway_api_key="secret",
    )

    with (
        patch("minimax_h3_headless.main.get_settings", return_value=configured),
        patch("minimax_h3_headless.main.uvicorn.run") as uvicorn_run,
    ):
        run()

    uvicorn_run.assert_called_once_with(
        "minimax_h3_headless.app:app",
        host="0.0.0.0",
        port=9876,
        proxy_headers=False,
    )

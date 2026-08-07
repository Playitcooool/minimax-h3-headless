"""Console entry point."""

import uvicorn

from .settings import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "minimax_h3_headless.app:app",
        host=settings.gateway_host,
        port=settings.gateway_port,
        proxy_headers=False,
    )

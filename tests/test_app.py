"""HTTP contract tests for the authenticated FastAPI gateway."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from minimax_h3_headless.app import create_app
from minimax_h3_headless.backends import BackendError, RoutedJob
from minimax_h3_headless.models import GenerateRequest
from minimax_h3_headless.settings import Settings


class FakeBackend:
    def __init__(self) -> None:
        self.created: list[GenerateRequest] = []
        self.status_jobs: list[RoutedJob] = []
        self.content_jobs: list[RoutedJob] = []
        self.health_result = {"fl2va": True, "ref2va": False}
        self.status_result: dict[str, Any] = {"status": "running"}
        self.create_error: BackendError | None = None
        self.status_error: BackendError | None = None
        self.content_error: BackendError | None = None
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def health(self) -> dict[str, bool]:
        return self.health_result

    async def create(self, request: GenerateRequest) -> RoutedJob:
        self.created.append(request)
        if self.create_error:
            raise self.create_error
        family = "ref2va" if request.task.value == "ref2va" else "fl2va"
        return RoutedJob(family, "backend-id")

    async def status(self, job: RoutedJob) -> dict[str, Any]:
        self.status_jobs.append(job)
        if self.status_error:
            raise self.status_error
        return self.status_result

    async def content(self, job: RoutedJob) -> tuple[AsyncIterator[bytes], str]:
        self.content_jobs.append(job)
        if self.content_error:
            raise self.content_error

        async def body() -> AsyncIterator[bytes]:
            yield b"video-"
            yield b"bytes"

        return body(), "video/x-test"


@asynccontextmanager
async def gateway_client(
    *, key: str = "secret", backend: FakeBackend | None = None
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeBackend]]:
    fake = backend or FakeBackend()
    configured = Settings(_env_file=None, gateway_api_key=key, backend="sglang")
    app = create_app(configured, fake)  # type: ignore[arg-type]
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            yield client, fake


def auth(value: str = "secret") -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


@pytest.mark.asyncio
async def test_health_is_public_and_reports_partial_availability() -> None:
    async with gateway_client() as (client, _):
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "backends": {"fl2va": True, "ref2va": False},
    }


@pytest.mark.asyncio
async def test_health_is_degraded_when_no_backend_is_available() -> None:
    backend = FakeBackend()
    backend.health_result = {"fl2va": False, "ref2va": False}
    async with gateway_client(backend=backend) as (client, _):
        response = await client.get("/healthz")

    assert response.json()["status"] == "degraded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "secret"},
        {"Authorization": "Basic secret"},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "bearer secret"},
        {"Authorization": "Bearer secret "},
    ],
)
async def test_protected_routes_reject_invalid_authorization(headers: dict[str, str]) -> None:
    async with gateway_client() as (client, backend):
        response = await client.post(
            "/v1/generations", json={"prompt": "A lake"}, headers=headers
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert backend.created == []


@pytest.mark.asyncio
async def test_missing_server_key_fails_closed() -> None:
    async with gateway_client(key="") as (client, _):
        response = await client.post(
            "/v1/generations", json={"prompt": "A lake"}, headers=auth()
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "H3_GATEWAY_API_KEY is not configured"


@pytest.mark.asyncio
async def test_create_validates_and_returns_public_routed_id() -> None:
    async with gateway_client() as (client, backend):
        response = await client.post(
            "/v1/generations", json={"prompt": " A lake "}, headers=auth()
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "fl2va:backend-id",
        "status": "queued",
        "backend": "sglang",
    }
    assert backend.created[0].prompt == "A lake"


@pytest.mark.asyncio
async def test_create_rejects_invalid_payload_before_backend() -> None:
    async with gateway_client() as (client, backend):
        response = await client.post(
            "/v1/generations",
            json={"prompt": "A lake", "task": "t2va", "conditions": [{"x": 1}]},
            headers=auth(),
        )

    assert response.status_code == 422
    assert backend.created == []


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["create", "status", "content"])
async def test_backend_errors_keep_status_and_hide_no_translation_details(endpoint: str) -> None:
    backend = FakeBackend()
    setattr(backend, f"{endpoint}_error", BackendError("backend says no", 429))
    async with gateway_client(backend=backend) as (client, _):
        if endpoint == "create":
            response = await client.post(
                "/v1/generations", json={"prompt": "A lake"}, headers=auth()
            )
        else:
            suffix = "/content" if endpoint == "content" else ""
            response = await client.get(
                f"/v1/generations/fl2va:job{suffix}", headers=auth()
            )

    assert response.status_code == 429
    assert response.json() == {"detail": "backend says no"}


@pytest.mark.asyncio
async def test_running_status_has_no_content_url() -> None:
    async with gateway_client() as (client, backend):
        response = await client.get("/v1/generations/ref2va:job-1", headers=auth())

    assert response.status_code == 200
    assert response.json() == {
        "id": "ref2va:job-1",
        "status": "running",
        "error": None,
        "content_url": None,
    }
    assert backend.status_jobs == [RoutedJob("ref2va", "job-1")]


@pytest.mark.asyncio
async def test_completed_status_builds_absolute_content_url() -> None:
    backend = FakeBackend()
    backend.status_result = {"status": "completed"}
    async with gateway_client(backend=backend) as (client, _):
        response = await client.get("/v1/generations/fl2va:job-1", headers=auth())

    assert response.json()["content_url"] == (
        "http://gateway.test/v1/generations/fl2va:job-1/content"
    )


@pytest.mark.asyncio
async def test_status_normalizes_backend_error_to_string() -> None:
    backend = FakeBackend()
    backend.status_result = {"status": "failed", "error": {"code": 7}}
    async with gateway_client(backend=backend) as (client, _):
        response = await client.get("/v1/generations/fl2va:job-1", headers=auth())

    assert response.json()["error"] == "{'code': 7}"


@pytest.mark.asyncio
async def test_content_streams_status_type_and_body() -> None:
    async with gateway_client() as (client, backend):
        response = await client.get(
            "/v1/generations/fl2va:job-1/content", headers=auth()
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/x-test"
    assert response.content == b"video-bytes"
    assert backend.content_jobs == [RoutedJob("fl2va", "job-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("job_id", ["bad", "t2va:id", "fl2va:a%3Fb"])
async def test_invalid_job_ids_are_rejected_without_backend_call(job_id: str) -> None:
    async with gateway_client() as (client, backend):
        response = await client.get(f"/v1/generations/{job_id}", headers=auth())

    assert response.status_code == 404
    assert backend.status_jobs == []


@pytest.mark.asyncio
async def test_lifespan_closes_backend() -> None:
    backend = FakeBackend()
    async with gateway_client(backend=backend):
        assert not backend.closed

    assert backend.closed

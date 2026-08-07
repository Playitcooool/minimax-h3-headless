"""Authenticated FastAPI gateway for agent-facing MiniMax H3 access."""

import hmac
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .backends import BackendError, H3Backend, RoutedJob
from .models import GenerateRequest, JobAccepted, JobStatus
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None, backend: H3Backend | None = None) -> FastAPI:
    configured = settings or get_settings()
    service = backend or H3Backend(configured)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await service.close()

    app = FastAPI(title="MiniMax H3 Headless Gateway", version="0.1.0", lifespan=lifespan)
    app.state.backend = service
    app.state.settings = configured

    def authenticate(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not configured.gateway_api_key:
            raise HTTPException(503, "H3_GATEWAY_API_KEY is not configured")
        expected = f"Bearer {configured.gateway_api_key}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    def translate(exc: BackendError) -> HTTPException:
        return HTTPException(exc.status_code, str(exc))

    @app.get("/healthz")
    async def health() -> dict[str, object]:
        backends = await service.health()
        return {"status": "ok" if any(backends.values()) else "degraded", "backends": backends}

    @app.post("/v1/generations", response_model=JobAccepted, dependencies=[Depends(authenticate)])
    async def create_generation(payload: GenerateRequest) -> JobAccepted:
        try:
            job = await service.create(payload)
        except BackendError as exc:
            raise translate(exc) from exc
        return JobAccepted(id=job.public_id, backend=configured.backend)

    @app.get("/v1/generations/{job_id}", response_model=JobStatus, dependencies=[Depends(authenticate)])
    async def generation_status(job_id: str, request: Request) -> JobStatus:
        try:
            job = RoutedJob.parse(job_id)
            payload = await service.status(job)
        except BackendError as exc:
            raise translate(exc) from exc
        status = str(payload.get("status", "unknown"))
        error = payload.get("error")
        content_url = None
        if status == "completed":
            content_url = str(request.url_for("generation_content", job_id=job_id))
        return JobStatus(id=job_id, status=status, error=str(error) if error else None, content_url=content_url)

    @app.get("/v1/generations/{job_id}/content", name="generation_content", dependencies=[Depends(authenticate)])
    async def generation_content(job_id: str) -> StreamingResponse:
        try:
            job = RoutedJob.parse(job_id)
            chunks, content_type = await service.content(job)
        except BackendError as exc:
            raise translate(exc) from exc
        return StreamingResponse(chunks, media_type=content_type)

    return app


app = create_app()

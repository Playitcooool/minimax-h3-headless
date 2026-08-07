"""Adapters for the two official MiniMax H3 serving options."""

import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .models import GenerateRequest, Task
from .settings import Settings


class BackendError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class RoutedJob:
    family: str
    backend_id: str

    @property
    def public_id(self) -> str:
        return f"{self.family}:{self.backend_id}"

    @classmethod
    def parse(cls, value: str) -> "RoutedJob":
        family, separator, backend_id = value.partition(":")
        if not separator or family not in {"fl2va", "ref2va"} or not backend_id:
            raise BackendError("invalid job id", 404)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", backend_id):
            raise BackendError("invalid job id", 404)
        if backend_id in {".", ".."}:
            raise BackendError("invalid job id", 404)
        return cls(family=family, backend_id=backend_id)


class H3Backend:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(timeout=settings.request_timeout_seconds)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _route(self, task: Task) -> tuple[str, str]:
        if task is Task.REF2VA:
            return "ref2va", self.settings.ref2va_url.rstrip("/")
        return "fl2va", self.settings.fl2va_url.rstrip("/")

    async def create(self, request: GenerateRequest) -> RoutedJob:
        family, base_url = self._route(request.task)
        if self.settings.backend == "sglang":
            response = await self._request(
                "POST", f"{base_url}/v1/videos", json=request.model_dump(mode="json")
            )
        else:
            form = self._vllm_form(request)
            response = await self._request(
                "POST",
                f"{base_url}/v1/videos",
                files={name: (None, value) for name, value in form.items()},
            )
        payload = self._json(response)
        backend_id = payload.get("id")
        if not isinstance(backend_id, str) or not backend_id:
            raise BackendError("backend response did not include a job id")
        return RoutedJob.parse(f"{family}:{backend_id}")

    async def status(self, job: RoutedJob) -> dict[str, Any]:
        base_url = self._base_url(job.family)
        response = await self._request("GET", f"{base_url}/v1/videos/{job.backend_id}")
        return self._json(response)

    async def content(self, job: RoutedJob) -> tuple[AsyncIterator[bytes], str]:
        base_url = self._base_url(job.family)
        request = self.client.build_request(
            "GET", f"{base_url}/v1/videos/{job.backend_id}/content"
        )
        try:
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise BackendError(f"backend connection failed: {exc}") from exc
        if response.is_error:
            body = (await response.aread()).decode(errors="replace")[:1000]
            await response.aclose()
            raise BackendError(f"backend returned {response.status_code}: {body}", response.status_code)

        async def chunks() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return chunks(), response.headers.get("content-type", "video/mp4")

    async def health(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for family, base_url in (
            ("fl2va", self.settings.fl2va_url),
            ("ref2va", self.settings.ref2va_url),
        ):
            try:
                response = await self.client.get(f"{base_url.rstrip('/')}/health")
                results[family] = response.is_success
            except httpx.HTTPError:
                results[family] = False
        return results

    def _base_url(self, family: str) -> str:
        value = self.settings.ref2va_url if family == "ref2va" else self.settings.fl2va_url
        return value.rstrip("/")

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"backend connection failed: {exc}") from exc
        if response.is_error:
            raise BackendError(
                f"backend returned {response.status_code}: {response.text[:1000]}",
                response.status_code,
            )
        return response

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendError("backend returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise BackendError("backend returned an unexpected JSON value")
        return payload

    @staticmethod
    def _vllm_form(request: GenerateRequest) -> dict[str, str]:
        sizes = {
            "21:9": (1536, 672),
            "16:9": (1344, 768),
            "4:3": (1024, 768),
            "1:1": (768, 768),
            "3:4": (768, 1024),
            "9:16": (768, 1344),
        }
        form = {
            "prompt": request.prompt,
            "fps": "24",
            "num_inference_steps": str(request.num_inference_steps),
            "flow_shift": str(request.flow_shift),
            "seed": str(request.seed),
            "extra_params": json.dumps(
                {
                    "task": request.task.value,
                    "duration": request.target.duration_seconds,
                    "audio_flow_shift": request.audio_flow_shift,
                },
                separators=(",", ":"),
            ),
        }
        if request.target.aspect_ratio != "auto":
            width, height = sizes[request.target.aspect_ratio]
            form.update(width=str(width), height=str(height))
        if request.task is Task.FL2VA:
            if len(request.conditions) != 1:
                raise BackendError("vLLM-Omni currently supports one FL2VA image per request", 422)
            form["image_reference"] = json.dumps({"image_url": request.conditions[0].uri})
        elif request.task is Task.REF2VA:
            images = [c for c in request.conditions if c.type == "image"]
            audios = [c for c in request.conditions if c.type == "audio"]
            videos = [c for c in request.conditions if c.type in {"video", "video_audio"}]
            if videos and (images or audios):
                raise BackendError("vLLM-Omni cannot mix video with separate image/audio references", 422)
            if images:
                if len(images) != 1 or len(audios) > 1:
                    raise BackendError("vLLM-Omni supports exactly one image and at most one audio", 422)
                form["image_reference"] = json.dumps({"image_url": images[0].uri})
                if audios:
                    form["audio_reference"] = json.dumps({"audio_url": audios[0].uri})
            elif videos:
                values = [{"video_url": condition.uri} for condition in videos]
                form["video_reference"] = json.dumps(values[0] if len(values) == 1 else values)
            else:
                raise BackendError("unsupported vLLM-Omni Ref2VA reference combination", 422)
        return form

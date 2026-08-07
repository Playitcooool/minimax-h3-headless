"""Unit tests for backend routing and official-server wire adapters."""

import json
from collections.abc import AsyncIterator
from email import policy
from email.parser import BytesParser

import httpx
import pytest

from minimax_h3_headless.backends import BackendError, H3Backend, RoutedJob
from minimax_h3_headless.models import GenerateRequest
from minimax_h3_headless.settings import Settings


def settings(backend: str = "sglang") -> Settings:
    return Settings(
        _env_file=None,
        backend=backend,
        fl2va_url="http://fl2va.test/root/",
        ref2va_url="http://ref2va.test/ref/",
    )


def fl2va_request(count: int = 1) -> GenerateRequest:
    return GenerateRequest(
        prompt="A fox runs",
        task="fl2va",
        conditions=[
            {
                "type": "image",
                "uri": f"file:///media/frame-{index}.png",
                "role": "keyframe",
                "frame_index": 0 if index == 0 else -1,
            }
            for index in range(count)
        ],
        target={"aspect_ratio": "9:16", "duration_seconds": 7},
        seed=42,
        num_inference_steps=31,
        flow_shift=11.5,
        audio_flow_shift=2.5,
    )


def ref2va_request(*media: str) -> GenerateRequest:
    return GenerateRequest(
        prompt="Use references",
        task="ref2va",
        conditions=[
            {
                "type": kind,
                "uri": f"file:///media/{index}.{kind}",
                "role": "reference",
            }
            for index, kind in enumerate(media)
        ],
    )


@pytest.mark.parametrize(
    ("value", "family", "backend_id"),
    [("fl2va:abc-123", "fl2va", "abc-123"), ("ref2va:a:b", "ref2va", "a:b")],
)
def test_routed_job_round_trip(value: str, family: str, backend_id: str) -> None:
    job = RoutedJob.parse(value)

    assert (job.family, job.backend_id, job.public_id) == (family, backend_id, value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        ":id",
        "t2va:id",
        "fl2va:",
        "fl2va:a/b",
        "fl2va:a?b",
        "fl2va:a#b",
        "fl2va:.",
        "fl2va:..",
        "fl2va:%2fhealth",
        "fl2va:a\\b",
        "fl2va:a\nb",
    ],
)
def test_routed_job_rejects_values_that_can_change_backend_path(value: str) -> None:
    with pytest.raises(BackendError) as caught:
        RoutedJob.parse(value)

    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_sglang_create_posts_full_json_contract_and_routes_t2va_to_fl2va() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "server-job"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)
    request = GenerateRequest(prompt="A lake", target={"duration_seconds": 5})

    job = await backend.create(request)

    assert job == RoutedJob("fl2va", "server-job")
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "http://fl2va.test/root/v1/videos"
    assert seen[0].headers["content-type"] == "application/json"
    assert json.loads(seen[0].content) == request.model_dump(mode="json")
    await client.aclose()


@pytest.mark.asyncio
async def test_sglang_ref2va_uses_ref2va_server() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"id": "ref-job"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    job = await backend.create(ref2va_request("image"))

    assert job.public_id == "ref2va:ref-job"
    assert seen == ["http://ref2va.test/ref/v1/videos"]
    await client.aclose()


def test_vllm_t2va_form_maps_generation_controls_and_dimensions() -> None:
    request = GenerateRequest(
        prompt="A lake",
        target={"aspect_ratio": "4:3", "duration_seconds": 8},
        seed=12,
        num_inference_steps=22,
        flow_shift=10.25,
        audio_flow_shift=2.75,
    )

    form = H3Backend._vllm_form(request)

    assert form == {
        "prompt": "A lake",
        "fps": "24",
        "num_inference_steps": "22",
        "flow_shift": "10.25",
        "seed": "12",
        "extra_params": '{"task":"t2va","duration":8.0,"audio_flow_shift":2.75}',
        "width": "1024",
        "height": "768",
    }


def test_vllm_auto_aspect_omits_dimensions() -> None:
    request = GenerateRequest(prompt="A lake", target={"aspect_ratio": "auto"})

    form = H3Backend._vllm_form(request)

    assert "width" not in form
    assert "height" not in form


def test_vllm_fl2va_maps_single_image_reference() -> None:
    form = H3Backend._vllm_form(fl2va_request())

    assert json.loads(form["image_reference"]) == {
        "image_url": "file:///media/frame-0.png"
    }


def test_vllm_fl2va_rejects_two_keyframes() -> None:
    with pytest.raises(BackendError, match="one FL2VA image") as caught:
        H3Backend._vllm_form(fl2va_request(2))

    assert caught.value.status_code == 422


@pytest.mark.parametrize(
    ("media", "field", "expected"),
    [
        (("image",), "image_reference", {"image_url": "file:///media/0.image"}),
        (("video",), "video_reference", {"video_url": "file:///media/0.video"}),
        (
            ("video", "video_audio"),
            "video_reference",
            [
                {"video_url": "file:///media/0.video"},
                {"video_url": "file:///media/1.video_audio"},
            ],
        ),
    ],
)
def test_vllm_ref2va_visual_mapping(
    media: tuple[str, ...], field: str, expected: object
) -> None:
    form = H3Backend._vllm_form(ref2va_request(*media))

    assert json.loads(form[field]) == expected


def test_vllm_ref2va_image_and_audio_mapping() -> None:
    form = H3Backend._vllm_form(ref2va_request("image", "audio"))

    assert json.loads(form["image_reference"]) == {
        "image_url": "file:///media/0.image"
    }
    assert json.loads(form["audio_reference"]) == {
        "audio_url": "file:///media/1.audio"
    }


@pytest.mark.parametrize("media", [("video", "audio"), ("image", "image")])
def test_vllm_rejects_unsupported_reference_combinations(media: tuple[str, ...]) -> None:
    with pytest.raises(BackendError) as caught:
        H3Backend._vllm_form(ref2va_request(*media))

    assert caught.value.status_code == 422


@pytest.mark.asyncio
async def test_vllm_create_submits_form_fields() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "vllm-job"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings("vllm_omni"), client)

    await backend.create(GenerateRequest(prompt="A lake"))

    content_type = seen[0].headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + seen[0].content
    )
    parts = list(message.iter_parts())
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        ).decode()
        for part in parts
    }
    assert fields == {
        "prompt": "A lake",
        "fps": "24",
        "num_inference_steps": "50",
        "flow_shift": "12.0",
        "seed": "0",
        "extra_params": '{"task":"t2va","duration":5,"audio_flow_shift":3.0}',
        "width": "1344",
        "height": "768",
    }
    assert all(part.get_content_disposition() == "form-data" for part in parts)
    assert all(part.get_filename() is None for part in parts)
    await client.aclose()


@pytest.mark.asyncio
async def test_status_routes_by_embedded_family() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"status": "running"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    assert await backend.status(RoutedJob("ref2va", "job-7")) == {"status": "running"}
    assert seen == ["http://ref2va.test/ref/v1/videos/job-7"]
    await client.aclose()


class ClosingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"one"
        yield b"two"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_content_streams_bytes_preserves_type_and_closes_response() -> None:
    stream = ClosingStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "video/custom"}, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    chunks, content_type = await backend.content(RoutedJob("fl2va", "j1"))
    assert content_type == "video/custom"
    assert [chunk async for chunk in chunks] == [b"one", b"two"]
    assert stream.closed
    await client.aclose()


@pytest.mark.asyncio
async def test_content_error_reads_bounded_body_and_closes() -> None:
    stream = ClosingStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    with pytest.raises(BackendError) as caught:
        await backend.content(RoutedJob("fl2va", "missing"))

    assert caught.value.status_code == 404
    assert stream.closed
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], "text"])
async def test_unexpected_json_values_are_rejected(payload: object) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    with pytest.raises(BackendError, match="unexpected JSON"):
        await backend.status(RoutedJob("fl2va", "job"))
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_json_body_is_rejected_as_invalid_json() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    with pytest.raises(BackendError, match="invalid JSON"):
        await backend.status(RoutedJob("fl2va", "job"))
    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_json_and_missing_create_id_are_rejected() -> None:
    responses = iter(
        [httpx.Response(200, text="not-json"), httpx.Response(200, json={"status": "queued"})]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    with pytest.raises(BackendError, match="invalid JSON"):
        await backend.status(RoutedJob("fl2va", "job"))
    with pytest.raises(BackendError, match="job id"):
        await backend.create(GenerateRequest(prompt="A lake"))
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_id", ["../health", "%2fhealth", "line\nbreak"])
async def test_create_rejects_unsafe_job_id_returned_by_backend(backend_id: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": backend_id})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    with pytest.raises(BackendError, match="invalid job id"):
        await backend.create(GenerateRequest(prompt="A lake"))
    await client.aclose()


@pytest.mark.asyncio
async def test_http_status_and_connection_errors_are_translated() -> None:
    async def status_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="busy")

    client = httpx.AsyncClient(transport=httpx.MockTransport(status_handler))
    backend = H3Backend(settings(), client)
    with pytest.raises(BackendError, match="429: busy") as caught:
        await backend.status(RoutedJob("fl2va", "job"))
    assert caught.value.status_code == 429
    await client.aclose()

    async def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    backend = H3Backend(settings(), client)
    with pytest.raises(BackendError, match="connection failed") as caught:
        await backend.status(RoutedJob("fl2va", "job"))
    assert caught.value.status_code == 502
    await client.aclose()


@pytest.mark.asyncio
async def test_health_checks_both_servers_and_tolerates_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "fl2va.test":
            return httpx.Response(204)
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = H3Backend(settings(), client)

    assert await backend.health() == {"fl2va": True, "ref2va": False}
    await client.aclose()


@pytest.mark.asyncio
async def test_close_only_closes_an_internally_owned_client() -> None:
    external = httpx.AsyncClient()
    external_backend = H3Backend(settings(), external)
    await external_backend.close()
    assert not external.is_closed
    await external.aclose()

    owned_backend = H3Backend(settings())
    await owned_backend.close()
    assert owned_backend.client.is_closed

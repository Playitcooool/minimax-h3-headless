"""Unit tests for the synchronous agent client."""

import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from minimax_h3_headless.client import H3Client


def make_client(handler: httpx.MockTransport) -> H3Client:
    wrapper = H3Client("http://gateway.test/root/", "secret", timeout=12)
    wrapper.client.close()
    wrapper.client = httpx.Client(
        base_url="http://gateway.test/root",
        headers={"Authorization": "Bearer secret"},
        transport=handler,
    )
    return wrapper


def test_constructor_normalizes_url_and_sets_bearer_and_timeout() -> None:
    client = H3Client("http://gateway.test///", "key", timeout=17)

    assert str(client.client.base_url) == "http://gateway.test"
    assert client.client.headers["Authorization"] == "Bearer key"
    assert client.client.timeout.connect == 17
    client.close()


def test_submit_posts_json_and_returns_payload() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "fl2va:j1"})

    with make_client(httpx.MockTransport(handler)) as client:
        result = client.submit({"prompt": "A lake"})

    assert result == {"id": "fl2va:j1"}
    assert seen[0].url.path == "/root/v1/generations"
    assert seen[0].headers["authorization"] == "Bearer secret"
    assert json.loads(seen[0].content) == {"prompt": "A lake"}


@pytest.mark.parametrize("operation", ["status", "download"])
def test_client_rejects_job_ids_that_are_not_safe_path_segments(
    operation: str, tmp_path: Path
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "running"})

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="invalid job id"):
            if operation == "status":
                client.status("fl2va:a/b")
            else:
                client.download("fl2va:a/b", tmp_path / "video.mp4")

    assert seen == []


def test_submit_and_status_raise_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "no"})

    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.submit({"prompt": "A lake"})
        with pytest.raises(httpx.HTTPStatusError):
            client.status("fl2va:j")


def test_wait_returns_completed_job_and_uses_poll_interval() -> None:
    client = H3Client("http://gateway.test", "secret")
    with (
        patch.object(
            client,
            "status",
            side_effect=[{"status": "queued"}, {"status": "completed", "id": "job"}],
        ) as status,
        patch("minimax_h3_headless.client.time.sleep") as sleep,
    ):
        result = client.wait("job", poll_interval=0.25, timeout=5)

    assert result == {"status": "completed", "id": "job"}
    assert status.call_count == 2
    sleep.assert_called_once_with(0.25)
    client.close()


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_wait_raises_for_failed_terminal_states(terminal: str) -> None:
    client = H3Client("http://gateway.test", "secret")
    with patch.object(
        client, "status", return_value={"status": terminal, "error": "reason"}
    ):
        with pytest.raises(RuntimeError, match=f"{terminal}: reason"):
            client.wait("job", poll_interval=0, timeout=5)
    client.close()


def test_wait_times_out_without_sleeping_after_deadline() -> None:
    client = H3Client("http://gateway.test", "secret")
    with (
        patch("minimax_h3_headless.client.time.monotonic", side_effect=[0, 0, 6]),
        patch.object(client, "status", return_value={"status": "queued"}) as status,
        patch("minimax_h3_headless.client.time.sleep") as sleep,
    ):
        with pytest.raises(TimeoutError, match="within 5 seconds"):
            client.wait("job", poll_interval=2, timeout=5)

    status.assert_called_once_with("job")
    sleep.assert_called_once_with(2)
    client.close()


def test_download_streams_to_created_parent(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"video-data")

    destination = tmp_path / "nested" / "output.mp4"
    with make_client(httpx.MockTransport(handler)) as client:
        result = client.download("fl2va:j", destination)

    assert result == destination
    assert destination.read_bytes() == b"video-data"


def test_download_http_error_does_not_leave_empty_output(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"missing")

    destination = tmp_path / "output.mp4"
    with make_client(httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            client.download("fl2va:missing", destination)

    assert not destination.exists()


def test_generate_submits_waits_then_downloads(tmp_path: Path) -> None:
    client = H3Client("http://gateway.test", "secret")
    destination = tmp_path / "output.mp4"
    with (
        patch.object(client, "submit", return_value={"id": "fl2va:j"}) as submit,
        patch.object(client, "wait", return_value={"status": "completed"}) as wait,
        patch.object(client, "download", return_value=destination) as download,
    ):
        result = client.generate(
            {"prompt": "A lake"}, destination, poll_interval=1, timeout=7
        )

    assert result == destination
    submit.assert_called_once_with({"prompt": "A lake"})
    wait.assert_called_once_with("fl2va:j", poll_interval=1, timeout=7)
    download.assert_called_once_with("fl2va:j", destination)
    client.close()


def test_context_manager_closes_transport() -> None:
    client = H3Client("http://gateway.test", "secret")
    with client as entered:
        assert entered is client
        assert not client.client.is_closed

    assert client.client.is_closed

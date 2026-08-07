from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
GENERATE = REPO / "scripts" / "generate.sh"


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def _sandbox(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the launcher into an isolated repository and install a fake curl."""
    root = tmp_path / "repo with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "generate.sh"
    shutil.copy2(GENERATE, script)

    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    state_dir = tmp_path / "curl-state"
    state_dir.mkdir()
    _write_executable(
        fake_bin / "curl",
        f"#!{sys.executable}\n"
        + r'''
import json
import os
import pathlib
import sys

args = sys.argv[1:]
state_dir = pathlib.Path(os.environ["FAKE_CURL_STATE_DIR"])
log = state_dir / "calls.jsonl"
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")

def value_after(flag):
    return args[args.index(flag) + 1]

output = pathlib.Path(value_after("--output"))
url = next((arg for arg in args if arg.startswith("http://") or arg.startswith("https://")), "")
phase = "download" if url.endswith("/content") else ("submit" if "--request" in args else "status")
if os.environ.get("FAKE_CURL_FAIL") == phase:
    if phase == "download" and os.environ.get("FAKE_CURL_PARTIAL") == "1":
        output.write_bytes(b"partial-or-error-body")
    print(f"fake {phase} HTTP error", file=sys.stderr)
    raise SystemExit(22)

if phase == "submit":
    data_arg = value_after("--data-binary")
    request = pathlib.Path(data_arg.removeprefix("@")).read_text(encoding="utf-8")
    (state_dir / "request.json").write_text(request, encoding="utf-8")
    body = os.environ.get("FAKE_SUBMIT_BODY", '{"id":"job/quoted id"}')
    output.write_text(body, encoding="utf-8")
elif phase == "status":
    count_file = state_dir / "status-count"
    count = int(count_file.read_text() if count_file.exists() else "0")
    statuses = os.environ.get("FAKE_STATUSES", "queued,running,completed").split(",")
    status = statuses[min(count, len(statuses) - 1)]
    count_file.write_text(str(count + 1))
    body = {"status": status}
    if status in ("failed", "cancelled"):
        body["error"] = os.environ.get("FAKE_ERROR", "backend exploded")
    output.write_text(json.dumps(body), encoding="utf-8")
else:
    output.write_bytes(b"fake-mp4\x00payload")
''',
    )
    return root, script, state_dir


def _run(
    tmp_path: Path,
    *args: str,
    stdin: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    root, script, state_dir = _sandbox(tmp_path)
    env = os.environ.copy()
    for key in (
        "H3_GATEWAY_API_KEY",
        "H3_GATEWAY_URL",
        "H3_POLL_INTERVAL_SECONDS",
        "H3_GENERATION_TIMEOUT_SECONDS",
        "H3_DURATION_SECONDS",
        "H3_ASPECT_RATIO",
        "H3_SEED",
    ):
        env.pop(key, None)
    env.update(
        H3_GATEWAY_API_KEY="env-secret",
        H3_POLL_INTERVAL_SECONDS="0",
        FAKE_CURL_STATE_DIR=str(state_dir),
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    if env_overrides:
        for key, value in env_overrides.items():
            if value == "<UNSET>":
                env.pop(key, None)
            else:
                env[key] = value
    result = subprocess.run(
        [str(script), *args],
        input=stdin,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result, root, state_dir


def _request(state_dir: Path) -> dict[str, object]:
    return json.loads((state_dir / "request.json").read_text(encoding="utf-8"))


def _calls(state_dir: Path) -> list[list[str]]:
    return [json.loads(line) for line in (state_dir / "calls.jsonl").read_text().splitlines()]


def test_positional_prompt_polls_and_downloads_to_requested_path(tmp_path: Path) -> None:
    output = tmp_path / "nested output" / "movie name.mp4"
    result, _, state = _run(tmp_path, "A moonlit fox", str(output))
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"fake-mp4\x00payload"
    assert _request(state)["prompt"] == "A moonlit fox"
    assert result.stdout.count("Status: queued") == 1
    assert result.stdout.count("Status: running") == 1
    assert result.stdout.count("Status: completed") == 1
    assert f"Saved video: {output}" in result.stdout


def test_interactive_prompt_and_default_output(tmp_path: Path) -> None:
    result, root, state = _run(tmp_path, stdin="An interactive otter\n")
    assert result.returncode == 0, result.stderr
    assert _request(state)["prompt"] == "An interactive otter"
    outputs = list((root / "outputs").glob("h3-*.mp4"))
    assert len(outputs) == 1
    assert outputs[0].read_bytes() == b"fake-mp4\x00payload"


def test_prompt_is_json_encoded_without_shell_interpolation(tmp_path: Path) -> None:
    prompt = '你好 "quoted"\nsecond line $HOME `touch NEVER` \\ end'
    output = tmp_path / "unicode.mp4"
    result, _, state = _run(tmp_path, prompt, str(output))
    assert result.returncode == 0, result.stderr
    assert _request(state)["prompt"] == prompt
    assert not (tmp_path / "NEVER").exists()


def test_environment_controls_request_and_gateway_url(tmp_path: Path) -> None:
    output = tmp_path / "custom.mp4"
    result, _, state = _run(
        tmp_path,
        "portrait video",
        str(output),
        env_overrides={
            "H3_GATEWAY_URL": "https://gateway.example/base",
            "H3_DURATION_SECONDS": "10.5",
            "H3_ASPECT_RATIO": "9:16",
            "H3_SEED": "123",
        },
    )
    assert result.returncode == 0, result.stderr
    request = _request(state)
    assert request["target"] == {
        "short_edge": 768,
        "aspect_ratio": "9:16",
        "duration_seconds": 10.5,
    }
    assert request["seed"] == 123
    calls = _calls(state)
    assert calls[0][calls[0].index("--request") + 1] == "POST"
    assert "https://gateway.example/base/v1/generations" in calls[0]
    assert any("https://gateway.example/base/v1/generations/job/quoted id/content" in call for call in calls)


def test_env_file_key_is_loaded_and_environment_takes_precedence(tmp_path: Path) -> None:
    root, script, state = _sandbox(tmp_path)
    (root / ".env").write_text("H3_GATEWAY_API_KEY=file-secret\n")
    env = os.environ.copy()
    env.update(
        FAKE_CURL_STATE_DIR=str(state),
        H3_POLL_INTERVAL_SECONDS="0",
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    env.pop("H3_GATEWAY_API_KEY", None)
    output = tmp_path / "file-key.mp4"
    result = subprocess.run(
        [str(script), "prompt", str(output)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert any("Authorization: Bearer file-secret" in call for call in _calls(state))

    # A non-empty environment key must override the file on every request.
    prior_call_count = len(_calls(state))
    other = tmp_path / "env-key.mp4"
    env["H3_GATEWAY_API_KEY"] = "environment-secret"
    result = subprocess.run(
        [str(script), "prompt", str(other)], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    recent = _calls(state)[prior_call_count:]
    assert recent
    assert all("Authorization: Bearer environment-secret" in call for call in recent)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_does_not_create_output(tmp_path: Path, status: str) -> None:
    output = tmp_path / "should-not-exist.mp4"
    result, _, _ = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={"FAKE_STATUSES": status, "FAKE_ERROR": "GPU unavailable"},
    )
    assert result.returncode == 1
    assert f"Generation {status}: GPU unavailable" in result.stderr
    assert not output.exists()


def test_timeout_does_not_download(tmp_path: Path) -> None:
    output = tmp_path / "timeout.mp4"
    result, _, state = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={
            "FAKE_STATUSES": "queued",
            "H3_GENERATION_TIMEOUT_SECONDS": "0",
        },
    )
    assert result.returncode == 1
    assert "Generation timed out after 0 seconds." in result.stderr
    assert not output.exists()
    assert not any(call[-2].endswith("/content") if len(call) > 1 else False for call in _calls(state))


@pytest.mark.parametrize("phase", ["submit", "status", "download"])
def test_http_errors_propagate_and_never_report_success(tmp_path: Path, phase: str) -> None:
    output = tmp_path / f"{phase}.mp4"
    result, _, _ = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={"FAKE_CURL_FAIL": phase, "FAKE_STATUSES": "completed"},
    )
    assert result.returncode == 22
    assert f"fake {phase} HTTP error" in result.stderr
    assert "Saved video:" not in result.stdout


def test_failed_partial_download_preserves_existing_output_and_cleans_temp_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing video.mp4"
    destination.write_bytes(b"valuable-existing-video")
    result, _, _ = _run(
        tmp_path,
        "prompt",
        str(destination),
        env_overrides={
            "FAKE_CURL_FAIL": "download",
            "FAKE_CURL_PARTIAL": "1",
            "FAKE_STATUSES": "completed",
        },
    )
    assert result.returncode == 22
    assert destination.read_bytes() == b"valuable-existing-video"
    assert list(tmp_path.glob("existing video.mp4.partial.*")) == []
    assert "Saved video:" not in result.stdout


def test_missing_job_id_fails_before_polling(tmp_path: Path) -> None:
    output = tmp_path / "missing-id.mp4"
    result, _, state = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={"FAKE_SUBMIT_BODY": '{"status":"queued"}'},
    )
    assert result.returncode != 0
    assert "Gateway response did not include a job id" in result.stderr
    assert len(_calls(state)) == 1
    assert not output.exists()


@pytest.mark.parametrize("prompt", ["", "   ", "\t"])
def test_empty_prompt_is_rejected_before_network(tmp_path: Path, prompt: str) -> None:
    result, _, state = _run(tmp_path, prompt, stdin="\n")
    assert result.returncode == 2
    assert "Prompt cannot be empty." in result.stderr
    assert not (state / "calls.jsonl").exists()


def test_missing_key_is_actionable_and_skips_network(tmp_path: Path) -> None:
    result, root, state = _run(
        tmp_path,
        "prompt",
        env_overrides={"H3_GATEWAY_API_KEY": "<UNSET>"},
    )
    assert result.returncode == 2
    assert f"Set H3_GATEWAY_API_KEY or create {root}/.env first." in result.stderr
    assert not (state / "calls.jsonl").exists()


@pytest.mark.parametrize("missing", ["curl", "python3"])
def test_missing_required_tool_is_reported(tmp_path: Path, missing: str) -> None:
    root, script, _ = _sandbox(tmp_path)
    minimal_bin = tmp_path / f"without-{missing}"
    minimal_bin.mkdir()
    needed = {"bash", "date", "dirname", "sed", "tail", "curl", "python3"} - {missing}
    for name in needed:
        target = (tmp_path / "fake bin" / name) if name == "curl" else shutil.which(name)
        assert target
        (minimal_bin / name).symlink_to(target)
    env = {
        "PATH": str(minimal_bin),
        "H3_GATEWAY_API_KEY": "secret",
        "FAKE_CURL_STATE_DIR": str(tmp_path / "curl-state"),
    }
    result = subprocess.run(
        [str(script), "prompt"], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert f"{missing} is required." in result.stderr


def test_temporary_directory_is_cleaned_after_success(tmp_path: Path) -> None:
    temp_parent = tmp_path / "temp-parent"
    temp_parent.mkdir()
    output = tmp_path / "clean.mp4"
    result, _, _ = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={"TMPDIR": str(temp_parent), "FAKE_STATUSES": "completed"},
    )
    assert result.returncode == 0, result.stderr
    assert list(temp_parent.iterdir()) == []


def test_script_is_executable_and_has_valid_bash_syntax() -> None:
    assert GENERATE.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        ["bash", "-n", str(GENERATE)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr

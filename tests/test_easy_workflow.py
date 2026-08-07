from __future__ import annotations

import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "setup.sh"
DOWNLOAD = REPO / "download_models.sh"
SERVER = REPO / "run_server.sh"
GENERATE = REPO / "generate.sh"


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _copy_script(source: Path, root: Path) -> Path:
    destination = root / source.name
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _fake_command(bin_dir: Path, name: str, log: Path, *, exit_code: int = 0) -> Path:
    return _executable(
        bin_dir / name,
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' {name!r}\" $*\" >> {str(log)!r}\n"
        f"exit {exit_code}\n",
    )


def _setup_sandbox(tmp_path: Path, *, include_uv: bool = True) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo with spaces"
    script = _copy_script(SETUP, root)
    log = tmp_path / "calls.log"
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()

    _executable(bin_dir / "uname", "#!/usr/bin/env bash\necho Linux\n")
    for name in ("curl", "git", "ffmpeg", "apt-get", "sudo"):
        _fake_command(bin_dir, name, log)
    if include_uv:
        _fake_command(bin_dir, "uv", log)

    _executable(
        root / "scripts/bootstrap_gateway.sh",
        "#!/usr/bin/env bash\nprintf 'gateway cwd=%s\\n' \"$PWD\" >> \"$CALL_LOG\"\n",
    )
    _executable(
        root / "scripts/bootstrap_sglang.sh",
        "#!/usr/bin/env bash\nprintf 'sglang cwd=%s\\n' \"$PWD\" >> \"$CALL_LOG\"\n",
    )
    return root, script, log


def test_setup_delegates_from_its_repo_and_is_repeatable(tmp_path: Path) -> None:
    root, script, log = _setup_sandbox(tmp_path)
    env = os.environ.copy()
    env.update(PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}", CALL_LOG=str(log))

    for _ in range(2):
        result = subprocess.run(
            [str(script)], cwd=tmp_path, env=env, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "Setup complete. Next run: ./download_models.sh" in result.stdout

    calls = log.read_text().splitlines()
    assert calls.count(f"gateway cwd={root}") == 2
    assert calls.count(f"sglang cwd={root}") == 2
    for directory in ("outputs", "logs", ".run", "models"):
        assert (root / directory).is_dir()


def test_setup_installs_only_missing_ubuntu_packages_with_sudo(tmp_path: Path) -> None:
    root, script, log = _setup_sandbox(tmp_path)
    (tmp_path / "fake bin/git").unlink()
    (tmp_path / "fake bin/ffmpeg").unlink()
    # Do not inherit the host's git/ffmpeg through PATH during command discovery.
    isolated = tmp_path / "isolated"
    shutil.copytree(tmp_path / "fake bin", isolated)
    for command in ("bash", "dirname", "mkdir", "python3"):
        target = shutil.which(command)
        assert target
        (isolated / command).symlink_to(target)
    env = {
        "PATH": str(isolated),
        "CALL_LOG": str(log),
        "HOME": str(tmp_path / "home"),
    }
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert "sudo apt-get update" in calls
    assert "sudo apt-get install -y git ffmpeg" in calls


def test_setup_reports_missing_package_manager(tmp_path: Path) -> None:
    _, script, _ = _setup_sandbox(tmp_path)
    isolated = tmp_path / "minimal"
    isolated.mkdir()
    for command in ("bash", "dirname", "uname"):
        source = tmp_path / "fake bin" / command
        target = source if source.exists() else Path(shutil.which(command) or "")
        assert target.exists()
        (isolated / command).symlink_to(target)
    result = subprocess.run(
        [str(script)], env={"PATH": str(isolated)}, text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "Install these system packages, then rerun setup.sh:" in result.stderr


def test_setup_installs_uv_and_uses_new_binary_in_same_run(tmp_path: Path) -> None:
    root, script, log = _setup_sandbox(tmp_path, include_uv=False)
    home = tmp_path / "home with spaces"
    curl = tmp_path / "fake bin/curl"
    _executable(
        curl,
        "#!/usr/bin/env bash\n"
        "printf 'curl:%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "cat <<'INSTALLER'\n"
        "mkdir -p \"$HOME/.local/bin\"\n"
        "printf '#!/usr/bin/env bash\\nexit 0\\n' > \"$HOME/.local/bin/uv\"\n"
        "chmod +x \"$HOME/.local/bin/uv\"\n"
        "INSTALLER\n",
    )
    env = os.environ.copy()
    isolated = tmp_path / "uv-install-bin"
    shutil.copytree(tmp_path / "fake bin", isolated)
    for command in ("bash", "cat", "chmod", "dirname", "mkdir", "sh"):
        target = shutil.which(command)
        assert target
        (isolated / command).symlink_to(target)
    env.update(
        PATH=str(isolated),
        CALL_LOG=str(log),
        HOME=str(home),
    )
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "Installing uv ..." in result.stdout
    assert (home / ".local/bin/uv").is_file()
    assert f"gateway cwd={root}" in log.read_text()


def test_setup_requires_sudo_for_missing_packages_as_non_root(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("non-root branch cannot be exercised as root")
    _, script, _ = _setup_sandbox(tmp_path)
    isolated = tmp_path / "without-sudo"
    isolated.mkdir()
    for command in ("bash", "dirname", "uname", "apt-get", "curl", "uv"):
        source = tmp_path / "fake bin" / command
        target = source if source.exists() else Path(shutil.which(command) or "")
        assert target.exists()
        (isolated / command).symlink_to(target)
    result = subprocess.run(
        [str(script)], env={"PATH": str(isolated)}, text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "sudo is required to install: git ffmpeg" in result.stderr


def _download_sandbox(tmp_path: Path, auth_exit: int = 0) -> tuple[Path, Path, Path]:
    root = tmp_path / "download repo with spaces"
    script = _copy_script(DOWNLOAD, root)
    log = tmp_path / "download.log"
    _executable(
        root / ".venv-sglang/bin/hf",
        "#!/usr/bin/env bash\n"
        "printf 'hf:%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        f"[[ \"$*\" == 'auth whoami' ]] && exit {auth_exit}\n"
        "exit 0\n",
    )
    _executable(
        root / "scripts/download_model.sh",
        "#!/usr/bin/env bash\n"
        "printf 'download:%s:model=%s:path=%s\\n' \"$1\" \"$H3_MODEL_DIR\" \"$PATH\" >> \"$CALL_LOG\"\n",
    )
    return root, script, log


@pytest.mark.parametrize("variant", ["fl2va", "ref2va", "both"])
def test_download_delegates_variant_and_default_model_directory(
    tmp_path: Path, variant: str
) -> None:
    root, script, log = _download_sandbox(tmp_path)
    env = os.environ.copy()
    env["CALL_LOG"] = str(log)
    result = subprocess.run(
        [str(script), variant], cwd=tmp_path, env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert f"download:{variant}:model={root / 'models/MiniMax-H3'}:" in log.read_text()
    assert str(root / ".venv-sglang/bin") in log.read_text()


def test_download_logs_in_only_when_needed_and_honors_custom_model_dir(tmp_path: Path) -> None:
    _, script, log = _download_sandbox(tmp_path, auth_exit=1)
    model_dir = tmp_path / "shared models/MiniMax-H3"
    env = os.environ.copy()
    env.update(CALL_LOG=str(log), H3_MODEL_DIR=str(model_dir))
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "hf:auth whoami" in calls
    assert "hf:auth login" in calls
    assert f"download:fl2va:model={model_dir}:" in calls


def test_download_rejects_bad_variant_and_missing_setup(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    script = _copy_script(DOWNLOAD, root)
    bad = subprocess.run([str(script), "wrong"], text=True, capture_output=True, check=False)
    assert bad.returncode == 2
    assert "Usage:" in bad.stderr
    missing = subprocess.run([str(script)], text=True, capture_output=True, check=False)
    assert missing.returncode == 1
    assert "Run ./setup.sh first" in missing.stderr


def _server_sandbox(
    tmp_path: Path, *, sglang_exits: bool = False, gateway_exits: bool = False
) -> tuple[Path, Path, Path]:
    root = tmp_path / "server repo with spaces"
    script = _copy_script(SERVER, root)
    state = tmp_path / "server-state"
    state.mkdir()
    model = root / "models/MiniMax-H3"
    (model / "FL2VA").mkdir(parents=True)
    (model / "model_index.json").write_text("{}")
    _executable(root / ".venv-sglang/bin/sglang", "#!/usr/bin/env bash\nexit 0\n")
    gateway_body = "#!/usr/bin/env bash\nexit 19\n" if gateway_exits else (
        "#!/usr/bin/env bash\n"
        "printf 'port=%s fl2va=%s\\n' \"${H3_GATEWAY_PORT:-}\" \"${H3_FL2VA_URL:-}\" > \"$SERVER_STATE/gateway-env\"\n"
        "while :; do sleep 1; done\n"
    )
    _executable(root / ".venv/bin/h3-gateway", gateway_body)
    if sglang_exits:
        launch = "#!/usr/bin/env bash\nexit 17\n"
    else:
        launch = (
            "#!/usr/bin/env bash\n"
            "printf 'model=%s profile=%s cuda=%s args=%s\\n' \"$H3_MODEL_PATH\" \"$H3_PROFILE\" "
            "\"$CUDA_VISIBLE_DEVICES\" \"$*\" > \"$SERVER_STATE/sglang-env\"\n"
            "while :; do sleep 1; done\n"
        )
    _executable(root / "deploy/start_sglang.sh", launch)
    fake_bin = tmp_path / "server-bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$SERVER_STATE/curl-calls\"\n"
        "[[ \"${FAKE_CURL_EXIT:-0}\" != 0 ]] && exit \"$FAKE_CURL_EXIT\"\n"
        "if [[ \"$*\" == */healthz* ]]; then [[ -f \"$SERVER_STATE/gateway-env\" ]]; exit; fi\n"
        "if [[ \"$*\" == */health* ]]; then [[ -f \"$SERVER_STATE/sglang-env\" ]]; exit; fi\n"
        "exit 22\n",
    )
    return root, script, state


def _server_env(tmp_path: Path, state: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        PATH=f"{tmp_path / 'server-bin'}{os.pathsep}{env['PATH']}",
        SERVER_STATE=str(state),
        H3_STARTUP_TIMEOUT_SECONDS="1",
    )
    env.update(overrides)
    return env


def test_server_start_status_stop_and_restart_with_defaults(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state)
    try:
        started = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
        assert started.returncode == 0, started.stderr
        assert "Server is ready. Run: ./generate.sh" in started.stdout
        assert (root / "logs/sglang.log").is_file()
        assert (root / "logs/gateway.log").is_file()
        for _ in range(100):
            if (state / "sglang-env").exists():
                break
            time.sleep(0.01)
        assert "profile=auto cuda=0 args=fl2va" in (state / "sglang-env").read_text()
        status = subprocess.run(
            [str(script), "status"], env=env, text=True, capture_output=True, check=False
        )
        assert status.returncode == 0
        assert "SGLang: running" in status.stdout
        assert "Gateway: running" in status.stdout

        duplicate = subprocess.run(
            [str(script), "start"], env=env, text=True, capture_output=True, check=False
        )
        assert duplicate.returncode == 1
        assert "Server is already running" in duplicate.stderr

        restarted = subprocess.run(
            [str(script), "restart"], env=env, text=True, capture_output=True, check=False
        )
        assert restarted.returncode == 0, restarted.stderr
        assert "Server is ready" in restarted.stdout
    finally:
        subprocess.run([str(script), "stop"], env=env, capture_output=True, check=False)

    stopped = subprocess.run(
        [str(script), "status"], env=env, text=True, capture_output=True, check=False
    )
    assert stopped.stdout == "SGLang: stopped\nGateway: stopped\n"
    assert not list((root / ".run").glob("*.pid"))


def test_server_honors_model_gpu_and_profile_environment(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    custom = tmp_path / "custom model path"
    (custom / "FL2VA").mkdir(parents=True)
    (custom / "model_index.json").write_text("{}")
    env = _server_env(
        tmp_path,
        state,
        H3_MODEL_PATH=str(custom),
        H3_PROFILE="h100x4",
        CUDA_VISIBLE_DEVICES="GPU-one,GPU-two,GPU-three,GPU-four",
    )
    try:
        result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        for _ in range(100):
            if (state / "sglang-env").exists():
                break
            time.sleep(0.01)
        launched = (state / "sglang-env").read_text()
        assert f"model={custom}" in launched
        assert "profile=h100x4" in launched
        assert "cuda=GPU-one,GPU-two,GPU-three,GPU-four" in launched
    finally:
        subprocess.run([str(script), "stop"], env=env, capture_output=True, check=False)


def test_server_propagates_custom_inference_and_gateway_ports(tmp_path: Path) -> None:
    _, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state, H3_INFERENCE_PORT="39101", H3_GATEWAY_PORT="39102")
    try:
        result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "39101/health" in (state / "curl-calls").read_text()
        assert "39102/healthz" in (state / "curl-calls").read_text()
        gateway_env = (state / "gateway-env").read_text()
        assert "port=39102" in gateway_env
        assert "fl2va=http://127.0.0.1:39101" in gateway_env
    finally:
        subprocess.run([str(script), "stop"], env=env, capture_output=True, check=False)


def test_server_rejects_missing_environment_and_model(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state)
    (root / ".venv/bin/h3-gateway").unlink()
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "Run ./setup.sh first" in result.stderr

    _executable(root / ".venv/bin/h3-gateway", "#!/usr/bin/env bash\nexit 0\n")
    shutil.rmtree(root / "models/MiniMax-H3/FL2VA")
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "Run ./download_models.sh first" in result.stderr


def test_server_cleans_pid_files_when_sglang_dies_at_startup(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path, sglang_exits=True)
    env = _server_env(tmp_path, state, FAKE_CURL_EXIT="22")
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "SGLang stopped during startup" in result.stderr
    assert not list((root / ".run").glob("*.pid"))


def test_server_cleans_both_processes_when_gateway_dies_at_startup(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path, gateway_exits=True)
    env = _server_env(tmp_path, state)
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "Gateway" in result.stderr
    assert not list((root / ".run").glob("*.pid"))


def test_server_timeout_stops_spawned_process_and_cleans_pid_files(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state, FAKE_CURL_EXIT="22", H3_STARTUP_TIMEOUT_SECONDS="0")
    result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "SGLang startup timed out" in result.stderr
    assert not list((root / ".run").glob("*.pid"))


def test_server_removes_nonnumeric_and_dead_stale_pid_files(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state)
    (root / ".run").mkdir()
    (root / ".run/sglang.pid").write_text("not-a-pid\n")
    (root / ".run/gateway.pid").write_text("99999999\n")
    try:
        result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
    finally:
        subprocess.run([str(script), "stop"], env=env, capture_output=True, check=False)


@pytest.mark.parametrize("port_variable", ["H3_INFERENCE_PORT", "H3_GATEWAY_PORT"])
def test_server_rejects_an_existing_service_without_owned_pid(
    tmp_path: Path, port_variable: str
) -> None:
    root, script, state = _server_sandbox(tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    env = _server_env(tmp_path, state, **{port_variable: str(port)})
    try:
        result = subprocess.run([str(script)], env=env, text=True, capture_output=True, check=False)
        assert result.returncode == 1
        assert "already" in result.stderr.lower() or "in use" in result.stderr.lower()
        assert not (state / "sglang-env").exists(), "launcher ran despite an existing service"
        assert not list((root / ".run").glob("*.pid"))
    finally:
        listener.close()


def test_server_does_not_kill_unrelated_process_from_stale_pid_file(tmp_path: Path) -> None:
    root, script, state = _server_sandbox(tmp_path)
    env = _server_env(tmp_path, state)
    (root / ".run").mkdir()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (root / ".run/sglang.pid").write_text(f"{unrelated.pid}\n")
        result = subprocess.run(
            [str(script), "stop"], env=env, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert unrelated.poll() is None, "stale PID file caused an unrelated live process to be killed"
    finally:
        if unrelated.poll() is None:
            unrelated.terminate()
        unrelated.wait(timeout=5)


def test_root_generate_preserves_all_arguments_and_exit_status(tmp_path: Path) -> None:
    root = tmp_path / "generate repo with spaces"
    script = _copy_script(GENERATE, root)
    output = tmp_path / "delegated args"
    _executable(
        root / "scripts/generate.sh",
        "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\nexit \"${DELEGATE_EXIT:-0}\"\n",
    )
    result = subprocess.run(
        [str(script), "prompt with spaces", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["<prompt with spaces>", f"<{output}>"]
    failed = subprocess.run(
        [str(script)],
        env={**os.environ, "DELEGATE_EXIT": "23"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 23


@pytest.mark.parametrize("script", [SETUP, DOWNLOAD, SERVER, GENERATE])
def test_easy_scripts_are_executable_and_valid_bash(script: Path) -> None:
    assert script.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr

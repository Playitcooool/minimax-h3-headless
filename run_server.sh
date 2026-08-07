#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [[ -n "${H3_MODULES_FILE:-}" && -f "${H3_MODULES_FILE}" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
  # shellcheck disable=SC1090
  source "${H3_MODULES_FILE}"
fi

action=${1:-start}
runtime_dir="${repo_dir}/.run"
log_dir="${repo_dir}/logs"
sglang_pid_file="${runtime_dir}/sglang.pid"
gateway_pid_file="${runtime_dir}/gateway.pid"

mkdir -p "${runtime_dir}" "${log_dir}"

read_pid() {
  local file=$1
  [[ -f "${file}" ]] && awk 'NR == 1 {print $1}' "${file}" || true
}

read_start_token() {
  local file=$1
  [[ -f "${file}" ]] && awk 'NR == 1 {print $2}' "${file}" || true
}

process_start_token() {
  local pid=$1
  if [[ -r "/proc/${pid}/stat" ]]; then
    python3 - "${pid}" <<'PY'
from pathlib import Path
import sys

value = Path(f"/proc/{sys.argv[1]}/stat").read_text()
fields = value.rsplit(") ", 1)[1].split()
print(fields[19])
PY
  else
    # Portable fallback for development/testing hosts without Linux /proc.
    ps -o lstart= -p "${pid}" 2>/dev/null | awk '{$1=$1; gsub(/ /, "_"); print}'
  fi
}

write_pid_record() {
  local file=$1
  local pid=$2
  local token
  token=$(process_start_token "${pid}")
  [[ -n "${token}" ]] || return 1
  printf '%s %s\n' "${pid}" "${token}" >"${file}"
}

is_running() {
  local pid=${1:-}
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

port_in_use() {
  local port=$1
  python3 - "${port}" <<'PY'
import socket
import sys

with socket.socket() as client:
    client.settimeout(0.5)
    raise SystemExit(0 if client.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

is_owned_process() {
  local file=$1
  local pid recorded_token live_token
  pid=$(read_pid "${file}")
  recorded_token=$(read_start_token "${file}")
  is_running "${pid}" || return 1
  [[ -n "${recorded_token}" ]] || return 1
  live_token=$(process_start_token "${pid}" 2>/dev/null || true)
  [[ -n "${live_token}" && "${live_token}" == "${recorded_token}" ]]
}

stop_process() {
  local name=$1
  local file=$2
  local pid
  pid=$(read_pid "${file}")
  if is_owned_process "${file}"; then
    echo "Stopping ${name} (${pid}) ..."
    kill "${pid}"
    for _ in {1..30}; do
      is_owned_process "${file}" || break
      sleep 1
    done
    is_owned_process "${file}" && kill -KILL "${pid}"
  elif is_running "${pid}"; then
    echo "Ignoring stale ${name} PID record for unrelated process ${pid}." >&2
  fi
  rm -f -- "${file}"
}

stop_all() {
  stop_process gateway "${gateway_pid_file}"
  stop_process sglang "${sglang_pid_file}"
}

show_status() {
  local sglang_pid gateway_pid
  sglang_pid=$(read_pid "${sglang_pid_file}")
  gateway_pid=$(read_pid "${gateway_pid_file}")
  if is_owned_process "${sglang_pid_file}"; then
    echo "SGLang: running (PID ${sglang_pid})"
  else
    echo "SGLang: stopped"
  fi
  if is_owned_process "${gateway_pid_file}"; then
    echo "Gateway: running (PID ${gateway_pid})"
  else
    echo "Gateway: stopped"
  fi
}

case "${action}" in
  stop) stop_all; exit 0 ;;
  restart) stop_all ;;
  status) show_status; exit 0 ;;
  start) ;;
  *) echo "Usage: $0 [start|stop|restart|status]" >&2; exit 2 ;;
esac

sglang_pid=$(read_pid "${sglang_pid_file}")
gateway_pid=$(read_pid "${gateway_pid_file}")
if is_owned_process "${sglang_pid_file}" || is_owned_process "${gateway_pid_file}"; then
  echo "Server is already running. Use './run_server.sh status' or './run_server.sh restart'." >&2
  exit 1
fi
rm -f -- "${sglang_pid_file}" "${gateway_pid_file}"

sglang_bin=${H3_SGLANG_BIN:-"${repo_dir}/.venv-sglang/bin/sglang"}
gateway_bin=${H3_GATEWAY_BIN:-"${repo_dir}/.venv/bin/h3-gateway"}
[[ -x "${sglang_bin}" && -x "${gateway_bin}" ]] || {
  echo "Environment not found. Run ./setup.sh first." >&2
  exit 1
}
command -v python3 >/dev/null 2>&1 || { echo "python3 is required. Run ./setup.sh." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "No NVIDIA GPU is visible. Run this inside your Nibi H100 allocation." >&2
  exit 1
}
if ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -Eqi 'H100'; then
  echo "No full H100 is visible. Run this inside your allocated Nibi H100 job." >&2
  exit 1
fi

model_path=${H3_MODEL_PATH:-"${repo_dir}/models/MiniMax-H3"}
[[ -f "${model_path}/model_index.json" && -d "${model_path}/FL2VA" ]] || {
  echo "FL2VA model not found at ${model_path}. Run ./download_models.sh first." >&2
  exit 1
}

inference_port=${H3_INFERENCE_PORT:-30010}
gateway_port=${H3_GATEWAY_PORT:-8080}
if port_in_use "${inference_port}"; then
  echo "Port ${inference_port} is already in use by a process not owned by this launcher." >&2
  echo "Stop that process or choose H3_INFERENCE_PORT before starting." >&2
  exit 1
fi
if port_in_use "${gateway_port}"; then
  echo "Port ${gateway_port} is already in use by a process not owned by this launcher." >&2
  echo "Stop that process or choose H3_GATEWAY_PORT before starting." >&2
  exit 1
fi

export H3_MODEL_PATH="${model_path}"
export H3_SGLANG_BIN="${sglang_bin}"
export H3_PROFILE=${H3_PROFILE:-auto}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export H3_INFERENCE_PORT="${inference_port}"
export H3_FL2VA_URL="http://127.0.0.1:${inference_port}"

echo "Starting MiniMax H3 on GPU ${CUDA_VISIBLE_DEVICES} ..."
nohup deploy/start_sglang.sh fl2va >"${log_dir}/sglang.log" 2>&1 &
sglang_pid=$!
if ! write_pid_record "${sglang_pid_file}" "${sglang_pid}"; then
  echo "SGLang exited before its process record could be created." >&2
  wait "${sglang_pid}" 2>/dev/null || true
  exit 1
fi

echo "Waiting for the model to load. This can take several minutes ..."
model_deadline=$((SECONDS + ${H3_STARTUP_TIMEOUT_SECONDS:-1800}))
while ! curl --fail --silent "http://127.0.0.1:${inference_port}/health" >/dev/null 2>&1; do
  if ! is_owned_process "${sglang_pid_file}"; then
    echo "SGLang stopped during startup. See ${log_dir}/sglang.log" >&2
    stop_all
    exit 1
  fi
  if ((SECONDS >= model_deadline)); then
    echo "SGLang startup timed out. See ${log_dir}/sglang.log" >&2
    stop_all
    exit 1
  fi
  sleep 5
done

echo "Starting the gateway ..."
nohup "${gateway_bin}" >"${log_dir}/gateway.log" 2>&1 &
gateway_pid=$!
if ! write_pid_record "${gateway_pid_file}" "${gateway_pid}"; then
  echo "Gateway exited before its process record could be created." >&2
  wait "${gateway_pid}" 2>/dev/null || true
  stop_all
  exit 1
fi

gateway_deadline=$((SECONDS + 60))
while ! curl --fail --silent "http://127.0.0.1:${gateway_port}/healthz" >/dev/null 2>&1; do
  if ! is_owned_process "${gateway_pid_file}"; then
    echo "Gateway stopped during startup. See ${log_dir}/gateway.log" >&2
    stop_all
    exit 1
  fi
  if ((SECONDS >= gateway_deadline)); then
    echo "Gateway startup timed out. See ${log_dir}/gateway.log" >&2
    stop_all
    exit 1
  fi
  sleep 1
done

echo
echo "Server is ready. Run: ./generate.sh"
echo "Logs: ${log_dir}/sglang.log and ${log_dir}/gateway.log"
echo "Stop everything with: ./run_server.sh stop"

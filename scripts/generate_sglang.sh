#!/usr/bin/env bash
set -euo pipefail

# Submit the official SGLang H3 T2VA payload, wait for completion, and download
# the MP4.  This deliberately talks to SGLang directly: there is no ComfyUI or
# project gateway in the default headless workflow.

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
port=${H3_PORT:-${H3_INFERENCE_PORT:-30010}}
sglang_url=${H3_SGLANG_URL:-"http://127.0.0.1:${port}"}
poll_interval=${H3_POLL_INTERVAL_SECONDS:-2}
generation_timeout=${H3_GENERATION_TIMEOUT_SECONDS:-7200}
duration=${H3_DURATION_SECONDS:-5}
aspect_ratio=${H3_ASPECT_RATIO:-16:9}
seed=${H3_SEED:-42}
steps=${H3_NUM_INFERENCE_STEPS:-50}
flow_shift=${H3_FLOW_SHIFT:-12.0}
audio_flow_shift=${H3_AUDIO_FLOW_SHIFT:-3.0}
model_id=${H3_MODEL_ID:-MiniMaxAI/MiniMax-H3}

usage() {
  cat <<'EOF'
Usage: scripts/generate_sglang.sh [PROMPT] [OUTPUT.mp4]

Submits a text-to-video-and-audio request to the local SGLang server.
If PROMPT is omitted, the script asks for it interactively.

Optional environment variables:
  H3_SGLANG_URL                  Direct SGLang endpoint (default localhost:30010)
  H3_DURATION_SECONDS             4 through 15 (default 5)
  H3_ASPECT_RATIO                 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16
  H3_SEED                         Integer seed (default 42)
  H3_NUM_INFERENCE_STEPS          Sampling steps (default 50)
  H3_GENERATION_TIMEOUT_SECONDS   Maximum wait time (default 7200)
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

(( $# <= 2 )) || { usage >&2; exit 2; }

prompt=${1:-}
if [[ -z "${prompt}" ]]; then
  read -r -p "Video prompt: " prompt
fi
[[ -n "${prompt//[[:space:]]/}" ]] || { echo "Prompt cannot be empty." >&2; exit 2; }

timestamp=$(date +%Y%m%d-%H%M%S)
output=${2:-"${repo_dir}/outputs/h3-${timestamp}.mp4"}

command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required." >&2; exit 1; }

work_dir=$(mktemp -d)
download_file=
cleanup() {
  rm -rf -- "${work_dir}"
  [[ -z "${download_file}" ]] || rm -f -- "${download_file}"
}
trap cleanup EXIT

request_file="${work_dir}/request.json"
response_file="${work_dir}/response.json"

python3 - \
  "${prompt}" "${duration}" "${aspect_ratio}" "${seed}" "${steps}" \
  "${flow_shift}" "${audio_flow_shift}" "${model_id}" >"${request_file}" <<'PY'
import json
import math
import sys

prompt, duration, aspect_ratio, seed, steps, flow_shift, audio_flow_shift, model_id = sys.argv[1:]

try:
    duration_value = float(duration)
    seed_value = int(seed)
    steps_value = int(steps)
    flow_shift_value = float(flow_shift)
    audio_flow_shift_value = float(audio_flow_shift)
except ValueError as exc:
    raise SystemExit(f"Invalid numeric generation setting: {exc}") from exc

if not math.isfinite(duration_value) or not 4 <= duration_value <= 15:
    raise SystemExit("H3_DURATION_SECONDS must be between 4 and 15.")
if aspect_ratio not in {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}:
    raise SystemExit("H3_ASPECT_RATIO must be one of 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16.")
if seed_value < 0:
    raise SystemExit("H3_SEED must be non-negative.")
if not 1 <= steps_value <= 100:
    raise SystemExit("H3_NUM_INFERENCE_STEPS must be between 1 and 100.")
if not math.isfinite(flow_shift_value) or not math.isfinite(audio_flow_shift_value):
    raise SystemExit("Flow-shift values must be finite numbers.")

# `seconds` and `target.duration_seconds` are both included to match the
# published SGLang request examples for MiniMax H3.
json.dump(
    {
        "model": model_id,
        "prompt": prompt,
        "seconds": duration_value,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_value,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps_value,
        "flow_shift": flow_shift_value,
        "audio_flow_shift": audio_flow_shift_value,
        "seed": seed_value,
    },
    sys.stdout,
)
PY

echo "Submitting generation to ${sglang_url} ..."
curl --fail-with-body --silent --show-error \
  --request POST "${sglang_url%/}/v1/videos" \
  --header 'Content-Type: application/json' \
  --data-binary "@${request_file}" \
  --output "${response_file}"

job_id=$(python3 - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
job_id = payload.get("id")
if not isinstance(job_id, str) or not job_id:
    raise SystemExit("SGLang response did not include a job id")
print(job_id)
PY
)
job_path=$(python3 - "${job_id}" <<'PY'
from urllib.parse import quote
import sys

print(quote(sys.argv[1], safe=""))
PY
)

echo "Job: ${job_id}"
started_at=${SECONDS}
last_status=

while true; do
  curl --fail-with-body --silent --show-error \
    "${sglang_url%/}/v1/videos/${job_path}" \
    --output "${response_file}"

  status=$(python3 - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("status", "unknown"))
PY
)

  if [[ "${status}" != "${last_status}" ]]; then
    echo "Status: ${status}"
    last_status=${status}
  fi

  case "${status}" in
    completed|succeeded) break ;;
    failed|cancelled)
      python3 - "${response_file}" <<'PY' >&2
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(f"Generation {payload.get('status')}: {payload.get('error') or 'no error details'}")
PY
      exit 1
      ;;
  esac

  if ((SECONDS - started_at >= generation_timeout)); then
    echo "Generation timed out after ${generation_timeout} seconds." >&2
    exit 1
  fi
  sleep "${poll_interval}"
done

mkdir -p -- "$(dirname -- "${output}")"
download_file=$(mktemp "${output}.partial.XXXXXX")
curl --fail-with-body --silent --show-error \
  --location \
  "${sglang_url%/}/v1/videos/${job_path}/content" \
  --output "${download_file}"

mv -f -- "${download_file}" "${output}"
download_file=

echo "Saved video: ${output}"

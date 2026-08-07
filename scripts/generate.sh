#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gateway_url=${H3_GATEWAY_URL:-http://127.0.0.1:8080}
poll_interval=${H3_POLL_INTERVAL_SECONDS:-2}
generation_timeout=${H3_GENERATION_TIMEOUT_SECONDS:-3600}
duration=${H3_DURATION_SECONDS:-5}
aspect_ratio=${H3_ASPECT_RATIO:-16:9}
seed=${H3_SEED:-42}

prompt=${1:-}
if [[ -z "${prompt}" ]]; then
  read -r -p "Video prompt: " prompt
fi
[[ -n "${prompt//[[:space:]]/}" ]] || { echo "Prompt cannot be empty." >&2; exit 2; }

timestamp=$(date +%Y%m%d-%H%M%S)
output=${2:-"${repo_dir}/outputs/h3-${timestamp}.mp4"}

api_key=${H3_GATEWAY_API_KEY:-}
if [[ -z "${api_key}" && -f "${repo_dir}/.env" ]]; then
  api_key=$(sed -n 's/^H3_GATEWAY_API_KEY=//p' "${repo_dir}/.env" | tail -n 1)
fi
[[ -n "${api_key}" ]] || {
  echo "Set H3_GATEWAY_API_KEY or create ${repo_dir}/.env first." >&2
  exit 2
}

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

python3 - "${prompt}" "${duration}" "${aspect_ratio}" "${seed}" >"${request_file}" <<'PY'
import json
import sys

prompt, duration, aspect_ratio, seed = sys.argv[1:]
json.dump(
    {
        "task": "t2va",
        "prompt": prompt,
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": float(duration),
        },
        "seed": int(seed),
    },
    sys.stdout,
)
PY

echo "Submitting generation to ${gateway_url} ..."
curl --fail-with-body --silent --show-error \
  --request POST "${gateway_url}/v1/generations" \
  --header "Authorization: Bearer ${api_key}" \
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
    raise SystemExit("Gateway response did not include a job id")
print(job_id)
PY
)

echo "Job: ${job_id}"
started_at=${SECONDS}
last_status=

while true; do
  curl --fail-with-body --silent --show-error \
    --header "Authorization: Bearer ${api_key}" \
    "${gateway_url}/v1/generations/${job_id}" \
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
    completed) break ;;
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
  --header "Authorization: Bearer ${api_key}" \
  "${gateway_url}/v1/generations/${job_id}/content" \
  --output "${download_file}"

mv -f -- "${download_file}" "${output}"
download_file=

echo "Saved video: ${output}"

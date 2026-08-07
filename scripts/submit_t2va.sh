#!/usr/bin/env bash
set -euo pipefail

gateway_url=${H3_GATEWAY_URL:-http://127.0.0.1:8080}
: "${H3_GATEWAY_API_KEY:?Export H3_GATEWAY_API_KEY first}"

curl --fail-with-body --silent --show-error \
  --request POST "${gateway_url}/v1/generations" \
  --header "Authorization: Bearer ${H3_GATEWAY_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "task": "t2va",
  "prompt": "A red panda makes tea in a quiet cabin during gentle rain, cinematic close shot, synchronized kettle and rain sounds, no music.",
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
  "seed": 42
}
JSON
echo

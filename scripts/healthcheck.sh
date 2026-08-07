#!/usr/bin/env bash
set -euo pipefail

gateway_url=${H3_GATEWAY_URL:-http://127.0.0.1:8080}
curl --fail-with-body --silent --show-error "${gateway_url}/healthz"
echo

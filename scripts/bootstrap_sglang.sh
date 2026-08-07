#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_dir}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

UV_LINK_MODE=copy uv sync --frozen --extra inference --python 3.11

echo "Unified MiniMax H3 environment ready. Activate it with: source .venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_dir}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

UV_LINK_MODE=copy uv sync --frozen --extra inference

if [[ ! -f .env ]]; then
  cp .env.example .env
  secret=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  python3 - "${secret}" <<'PY'
from pathlib import Path
import sys

path = Path(".env")
path.write_text(path.read_text().replace("change-me-to-a-long-random-secret", sys.argv[1]))
PY
  chmod 600 .env
  echo "Created .env with a random API key."
fi

echo "Unified MiniMax H3 environment ready. Activate it with: source .venv/bin/activate"

#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_dir}"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
}

uv venv --python 3.11 .venv-sglang
UV_LINK_MODE=copy uv pip install \
  --python .venv-sglang/bin/python \
  --prerelease=allow \
  "sglang[diffusion]" \
  "huggingface_hub[cli]"

echo "SGLang environment ready at ${repo_dir}/.venv-sglang"

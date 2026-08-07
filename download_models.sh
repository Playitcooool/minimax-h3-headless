#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"

variant=${1:-fl2va}
case "${variant}" in
  fl2va|ref2va|both) ;;
  *) echo "Usage: $0 [fl2va|ref2va|both]" >&2; exit 2 ;;
esac

hf_bin="${repo_dir}/.venv-sglang/bin/hf"
[[ -x "${hf_bin}" ]] || {
  echo "Environment not found. Run ./setup.sh first." >&2
  exit 1
}

if ! "${hf_bin}" auth whoami >/dev/null 2>&1; then
  echo "Log in to Hugging Face to download MiniMax H3."
  "${hf_bin}" auth login
fi

export PATH="${repo_dir}/.venv-sglang/bin:${PATH}"
export H3_MODEL_DIR=${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}
scripts/download_model.sh "${variant}"

echo
echo "Model ready. Next run: ./run_server.sh"

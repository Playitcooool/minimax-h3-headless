#!/usr/bin/env bash
set -euo pipefail

model_dir=${H3_MODEL_DIR:-"${PWD}/models/MiniMax-H3"}
variant=${1:-both}

command -v hf >/dev/null 2>&1 || {
  echo "The Hugging Face CLI is required. Run scripts/bootstrap_sglang.sh first." >&2
  exit 1
}

case "${variant}" in
  fl2va) include=("model_index.json" "FL2VA/*") ;;
  ref2va) include=("model_index.json" "Ref2VA/*") ;;
  both) include=("model_index.json" "FL2VA/*" "Ref2VA/*") ;;
  *) echo "Usage: $0 [fl2va|ref2va|both]" >&2; exit 2 ;;
esac

mkdir -p "${model_dir}"
hf download MiniMaxAI/MiniMax-H3 --include "${include[@]}" --local-dir "${model_dir}"
echo "Model downloaded to ${model_dir}"

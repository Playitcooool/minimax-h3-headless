#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
variant=${1:-fl2va}
profile=${H3_PROFILE:-auto}
model_path=${H3_MODEL_PATH:-MiniMaxAI/MiniMax-H3}
host=${H3_INFERENCE_HOST:-127.0.0.1}

case "${variant}" in
  fl2va) port=${H3_INFERENCE_PORT:-30010} ;;
  ref2va) port=${H3_INFERENCE_PORT:-30011} ;;
  *) echo "Usage: $0 [fl2va|ref2va]" >&2; exit 2 ;;
esac

if [[ "${profile}" == "auto" ]]; then
  profile=$("${repo_dir}/deploy/detect_profile.sh" sglang)
  echo "Auto-selected SGLang profile: ${profile}" >&2
fi

case "${profile}" in
  h100x1|genericx1)
    resident_layers=${H3_DIT_RESIDENT_LAYERS:-20}
    [[ "${resident_layers}" =~ ^[0-9]+$ ]] || {
      echo "H3_DIT_RESIDENT_LAYERS must be a non-negative integer." >&2
      exit 2
    }
    topology=(
      --num-gpus 1 --tp-size 1 --ulysses-degree 1 --performance-mode memory
      --layerwise-offload-components "dit,text_encoder,vae"
      --dit-offload-prefetch-size 1 --dit-layerwise-resident-layers "${resident_layers}"
      --enable-torch-compile false
    )
    ;;
  h100x4)
    topology=(--num-gpus 4 --tp-size 2 --ulysses-degree 2 --performance-mode speed)
    ;;
  h100x4_memory)
    topology=(--num-gpus 4 --tp-size 4 --ulysses-degree 1 --performance-mode speed)
    ;;
  h100x4_fsdp)
    topology=(--num-gpus 4 --ulysses-degree 4 --performance-mode speed --use-fsdp-inference true)
    ;;
  h200x4)
    topology=(--num-gpus 4 --ulysses-degree 4 --performance-mode speed)
    ;;
  rtx5090x2)
    topology=(
      --num-gpus 2 --tp-size 2 --ulysses-degree 1 --performance-mode memory
      --layerwise-offload-components "dit,text_encoder,vae"
      --dit-offload-prefetch-size 1 --dit-layerwise-resident-layers 20
      --enable-torch-compile false
    )
    ;;
  *)
    echo "Unknown H3_PROFILE=${profile}" >&2
    echo "Choose auto, h100x1, h100x4, h100x4_memory, h100x4_fsdp, h200x4, rtx5090x2, or genericx1." >&2
    exit 2
    ;;
esac

sglang_bin=${H3_SGLANG_BIN:-"${repo_dir}/.venv/bin/sglang"}
[[ -x "${sglang_bin}" ]] || {
  echo "SGLang not found at ${sglang_bin}; run scripts/bootstrap_sglang.sh." >&2
  exit 1
}

exec "${sglang_bin}" serve \
  --model-path "${model_path}" \
  --model-variant "${variant}" \
  "${topology[@]}" \
  --host "${host}" \
  --port "${port}"

#!/usr/bin/env bash
set -euo pipefail

variant=${1:-FL2VA}
profile=${H3_PROFILE:-b300x4}
model_root=${H3_MODEL_DIR:?Set H3_MODEL_DIR to the absolute MiniMax-H3 download directory}
host=${H3_INFERENCE_HOST:-127.0.0.1}

case "${variant}" in
  FL2VA) port=${H3_INFERENCE_PORT:-30010} ;;
  Ref2VA) port=${H3_INFERENCE_PORT:-30011} ;;
  *) echo "Usage: $0 [FL2VA|Ref2VA]" >&2; exit 2 ;;
esac

model_dir="${model_root}/${variant}"
[[ -d "${model_dir}" ]] || { echo "Missing model partition: ${model_dir}" >&2; exit 1; }

case "${profile}" in
  b300x4)
    gpu_count=4
    args=(--num-gpus 4 --usp 4 --ring 1 --vae-patch-parallel-size 4
      --vae-parallel-mode tile --vae-use-tiling --diffusion-attention-backend FLASH_ATTN)
    ;;
  rtx5090x2)
    gpu_count=2
    args=(--num-gpus 2 --tensor-parallel-size 2 --usp 1 --ring 1
      --text-encoder-tp-size 2 --vae-patch-parallel-size 2 --vae-parallel-mode tile
      --vae-use-tiling --enable-distributed-layerwise-offload --dlo-no-use-allgather
      --dlo-resident-layers 20 --enforce-eager --diffusion-attention-backend CUDNN_ATTN)
    ;;
  rtx4090x2)
    gpu_count=2
    args=(--num-gpus 2 --tensor-parallel-size 2 --usp 1 --ring 1
      --text-encoder-tp-size 2 --vae-patch-parallel-size 2 --vae-parallel-mode tile
      --vae-use-tiling --enable-distributed-layerwise-offload --dlo-no-use-allgather
      --dlo-resident-layers 12 --enforce-eager --diffusion-attention-backend CUDNN_ATTN)
    ;;
  single_offload)
    gpu_count=1
    args=(--num-gpus 1 --enable-cpu-offload --diffusion-attention-backend FLASH_ATTN)
    ;;
  *) echo "Unknown H3_PROFILE=${profile}" >&2; exit 2 ;;
esac

mkdir -p "${model_root}/vllm-storage/${variant}"

exec docker run --rm --gpus "${gpu_count}" \
  --ipc=host \
  -p "${host}:${port}:${port}" \
  -v "${model_dir}:${model_dir}:ro" \
  -v "${model_root}/vllm-storage/${variant}:/var/tmp/vllm-omni-videos" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  -e VLLM_OMNI_SERVER_STORAGE__PATH=/var/tmp/vllm-omni-videos \
  --entrypoint vllm \
  vllm/vllm-omni:minimax-h3 \
  serve "${model_dir}" --omni --host 0.0.0.0 --port "${port}" \
  --trust-remote-code "${args[@]}"

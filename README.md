# MiniMax H3 — single-H100, headless deployment

This repository now has one recommended path for an SSH-only Linux server:
**SGLang directly on `127.0.0.1`, with no ComfyUI and no extra gateway.**

`MiniMax H3` produces a 768p video and synchronized stereo audio in one request.
The public release splits its weights into two partitions:

- `FL2VA` — text-to-video-and-audio (`t2va`) and first/last-frame generation.
- `Ref2VA` — image, audio, and video reference generation.

On a single H100 80 GB, run one partition at a time. The default is `FL2VA`,
which is enough for prompt-only generation.

## Important limitation

The full H3 weights are larger than one H100. This project therefore uses
SGLang's BF16/FP32 **layerwise CPU offload** mode. It does not quantize the
model and does not depend on ComfyUI, but it is much slower than a resident
multi-GPU deployment.

MiniMax and SGLang publish measured H100 recipes for four H100 80 GB GPUs; the
single-H100 offload topology is a practical capacity configuration, not a
published H100 benchmark. Budget several minutes per first 5-second clip and
validate the speed on your own server.

The locally available model is H3-Base at a 768-pixel short edge. The hosted
H3-Context-IR prompt-preprocessing stage and 2K regeneration are not included
in the open release.

## Server requirements

- Linux with one NVIDIA H100 80 GB visible to `nvidia-smi`
- At least 128 GiB host RAM; 256 GiB is recommended for CPU offload
- At least 180 GiB free disk for one checkpoint partition (more for both)
- A CUDA driver compatible with the SGLang version locked in this repository
- `curl`, `git`, `python3`, and `ffmpeg` already available on the server

No system packages are installed by these scripts. `uv` is installed into your
user account only when it is missing.

## Four commands

Run these on the H100 server after cloning the repository:

```bash
./h3.sh setup
./h3.sh download
./h3.sh start
./h3.sh generate "A red panda makes tea in a quiet cabin during gentle rain, cinematic close shot, synchronized kettle and rain sounds."
```

Before `download`, open the [MiniMax H3 model page](https://huggingface.co/MiniMaxAI/MiniMax-H3), accept its license, and use a Hugging Face account with access. The command opens the Hugging Face login flow if needed.

The generated MP4 is written to `outputs/` and the server log to
`logs/sglang.log`.

## Day-to-day commands

```bash
./h3.sh status
./h3.sh logs
./h3.sh stop
./h3.sh restart
```

The server binds only to `127.0.0.1:30010`, so it is not exposed to the public
network. To call SGLang's native API from your laptop, use an SSH tunnel:

```bash
ssh -N -L 30010:127.0.0.1:30010 USER@GPU_SERVER
```

## Optional settings

The defaults create a 5-second, 16:9, 768p clip with 50 inference steps.

```bash
H3_DURATION_SECONDS=10 H3_ASPECT_RATIO=9:16 H3_SEED=123 \
  ./h3.sh generate "A slow vertical tracking shot through a neon night market."
```

If startup or generation runs out of memory, reduce the number of resident DiT
blocks and restart. This trades speed for capacity without changing the model's
BF16/FP32 weights:

```bash
H3_DIT_RESIDENT_LAYERS=4 ./h3.sh restart
```

The default is `20`. Keep `H3_DIT_RESIDENT_LAYERS` at a non-negative integer;
increase it only after measuring available GPU and host memory.

To put model weights on a larger mounted volume, set the same environment
variable for setup, download, and server start:

```bash
export H3_MODEL_DIR=/data/models/MiniMax-H3
./h3.sh setup
./h3.sh download
./h3.sh start
```

To install and switch to the reference partition, stop `FL2VA` first:

```bash
./h3.sh stop
./h3.sh download ref2va
./h3.sh start ref2va
```

`./h3.sh generate` intentionally handles the simple text-only `FL2VA` request.
For first/last-frame or Ref2VA inputs, call the native SGLang endpoint with the
official request schema. The local server accepts `file://` input URIs only for
files visible on that same server.

## What the launcher does

`./h3.sh start` launches this direct SGLang configuration:

```bash
sglang serve \
  --model-path /path/to/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 1 \
  --tp-size 1 \
  --ulysses-degree 1 \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 20 \
  --enable-torch-compile false \
  --host 127.0.0.1 \
  --port 30010
```

This is the conservative SGLang memory policy: it avoids non-reference
quantization and leaves the default attention backend unchanged. The direct
client sends the documented `/v1/videos` payload, polls its status, then
downloads the completed MP4 atomically.

## Troubleshooting

- **Download denied:** accept the MiniMax H3 license in Hugging Face first, then
  rerun `./h3.sh download`.
- **Server exits during startup:** run `./h3.sh logs`. Most often this means
  insufficient host RAM, GPU memory, free disk, or a CUDA/driver mismatch.
- **GPU OOM:** restart with `H3_DIT_RESIDENT_LAYERS=4`; do not enable arbitrary
  quantization as a first response.
- **Slow first request:** expected. The model is loading and CPU-offloaded
  blocks traverse PCIe during denoising.
- **Need 2K output or official Context-IR quality:** use MiniMax's hosted API;
  those stages are not open-sourced.

The older FastAPI gateway, Slurm, and vLLM-Omni files remain in the repository
for existing users, but they are not part of the recommended single-H100 path.

This repository's MIT license covers only its deployment code. MiniMax H3
weights remain subject to the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3).

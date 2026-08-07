# MiniMax H3 Headless

An easy-to-deploy, SSH-first wrapper around the official
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) open-weight release. It
runs without ComfyUI on a Linux GPU server or Slurm cluster and exposes a small
authenticated API for agent pipelines.

## Four-command workflow

On your Linux H100 server:

```bash
git clone https://github.com/Playitcooool/minimax-h3-headless.git
cd minimax-h3-headless

./setup.sh
./download_models.sh
./run_server.sh
./generate.sh
```

That is the complete basic workflow:

1. `setup.sh` installs missing Ubuntu packages, uv, the gateway environment,
   and SGLang.
2. `download_models.sh` logs in to Hugging Face when needed and downloads the
   FL2VA checkpoint for prompt-to-video generation.
3. `run_server.sh` automatically selects the visible GPU, starts SGLang and the
   authenticated gateway in the background, waits until both are ready, and
   writes logs under `logs/`.
4. `generate.sh` asks for a prompt, waits for generation, and saves the MP4
   under `outputs/`.

Stop or inspect the server with:

```bash
./run_server.sh status
./run_server.sh stop
./run_server.sh restart
```

Everything below is optional detail for remote access, clusters, alternative
models, and advanced configuration.

## What this repo provides

- Official `t2va`, `fl2va`, and `ref2va` request contracts.
- Automatic GPU detection, defaulting to a single-H100 offload deployment.
- SGLang profiles, including the measured 4×H100 TP2 + Ulysses2 fast path.
- vLLM-Omni Docker profiles for its currently documented hardware paths.
- An authenticated FastAPI gateway that routes FL2VA and Ref2VA separately.
- A Python client, curl examples, Slurm template, systemd unit, and GitHub CI.
- A GPU-free mocked test suite for the entire gateway.

## Important model facts

MiniMax H3 is not a conventional text LLM. The open H3-Base release jointly
generates 24 FPS H.264 video and 32 kHz stereo AAC audio at a 768-pixel short
edge. Duration is 4–15 seconds. FL2VA and Ref2VA are separate checkpoint
partitions and therefore separate server processes. H3-Context-IR and the 2K
regeneration stage are hosted services and are not part of the open release.

## Default path: one H100 with automatic selection

Prerequisites: Linux, one H100 80 GB GPU, at least 256 GB host RAM, CUDA/driver compatible with current
SGLang, `ffmpeg`, `git`, and [uv](https://docs.astral.sh/uv/).

```bash
git clone YOUR_GITHUB_URL minimax-h3-headless
cd minimax-h3-headless

scripts/bootstrap_gateway.sh
scripts/bootstrap_sglang.sh

# Optional but recommended for clusters/shared storage:
source .venv-sglang/bin/activate
hf auth login
export H3_MODEL_DIR=/data/models/MiniMax-H3
scripts/download_model.sh fl2va
export H3_MODEL_PATH="$H3_MODEL_DIR"

# Terminal/tmux 1: detects the visible H100 and selects CPU/layerwise offload
export CUDA_VISIBLE_DEVICES=0
export H3_PROFILE=auto
deploy/start_sglang.sh fl2va

# Terminal/tmux 2
uv run h3-gateway
```

For Ref2VA on the same single H100, stop FL2VA first, then run:

```bash
export CUDA_VISIBLE_DEVICES=0
deploy/start_sglang.sh ref2va
```

You can skip the manual model download and leave `H3_MODEL_PATH` unset; SGLang
will use `MiniMaxAI/MiniMax-H3` and download through the Hugging Face cache.

The single-H100 SGLang profile is a conservative project-provided offload
configuration, not an upstream benchmarked topology. It will be much slower
than four H100s. If four H100s are visible, `auto` selects the official measured
TP2 + Ulysses2 profile automatically.

## Connect from your laptop

Create an SSH tunnel. The remote services stay private:

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@GPU_SERVER
```

Copy the generated API key from the server's `.env`, then on the laptop generate
a video with one command:

```bash
export H3_GATEWAY_API_KEY='the-server-key'
scripts/generate.sh "A panda walking through Shanghai at night" output.mp4
```

Omit both arguments for an interactive prompt and an automatically named file:

```bash
scripts/generate.sh
```

The script submits the job, displays status changes, waits for completion, and
downloads the MP4. Optional environment controls:

```bash
export H3_DURATION_SECONDS=10       # 4 through 15
export H3_ASPECT_RATIO=9:16         # 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16
export H3_SEED=123
export H3_GENERATION_TIMEOUT_SECONDS=7200
```

For Python/agent integration, see [`examples/agent_client.py`](examples/agent_client.py).

## vLLM-Omni

vLLM H3 support lives in vLLM-Omni, not the ordinary vLLM text-serving path.
Download both model partitions locally, install NVIDIA Container Toolkit, then:

```bash
export H3_MODEL_DIR=/data/models/MiniMax-H3
export H3_PROFILE=auto
deploy/docker/start_vllm_omni.sh FL2VA
```

Set `H3_BACKEND=vllm_omni` in `.env` before starting the gateway. Current
vLLM-Omni reference serving accepts a narrower set of Ref2VA combinations than
the model itself; the gateway returns HTTP 422 for unsupported combinations.

## Slurm and production

- [`docs/SLURM.md`](docs/SLURM.md): site-neutral batch submission and tunnels.
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md): profiles, security, and failures.
- [`deploy/systemd/h3-gateway.service`](deploy/systemd/h3-gateway.service): a
  hardened template for a dedicated `/opt/minimax-h3-headless` install.

## Local development

```bash
uv sync --extra dev --frozen
uv run pytest -q
```

This repository's MIT license covers only its own gateway and deployment code.
MiniMax H3 weights and upstream code remain subject to MiniMax's own license.

# Slurm generation and serving

## One-shot generation job

The recommended batch path starts FL2VA, waits for it to become healthy,
generates one video, and stops the server automatically. Run setup and download
the FL2VA weights once on the login node, then submit:

```bash
./scripts/submit_slurm_generation.sh \
  "A cinematic tracking shot through a lantern-lit forest, with synchronized rain and footsteps." \
  outputs/forest.mp4
```

The helper loads `H3_SLURM_ACCOUNT` from `.env`, requests one H100 by default,
and prints the Slurm job ID. If `OUTPUT.mp4` is omitted, the result is written to
`outputs/h3-JOB_ID.mp4`. SGLang's per-job log is `logs/sglang-JOB_ID.log`, while
Slurm stdout and stderr are `slurm-h3-generate-JOB_ID.out` and
`slurm-h3-generate-JOB_ID.err`.

Generation settings are ordinary exported variables:

```bash
H3_DURATION_SECONDS=10 H3_ASPECT_RATIO=9:16 H3_SEED=123 \
  ./scripts/submit_slurm_generation.sh "A vertical shot through a night market."
```

For a different site GPU syntax, pass the exact option through the helper:

```bash
H3_SLURM_GPU_OPTION=--gres=gpu:h100:1 \
  ./scripts/submit_slurm_generation.sh "A quiet mountain lake at dawn."
```

You can also call the batch file directly. It intentionally omits account,
partition, and GPU directives because those names differ between clusters:

```bash
cd /path/to/minimax-h3-headless
export H3_REPO_DIR="$PWD"
export H3_MODEL_PATH=/path/to/models/MiniMax-H3

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  --gpus-per-node=h100:1 \
  --export=ALL \
  deploy/slurm/h3-generate.sbatch \
  "A red panda makes tea in a quiet cabin." \
  outputs/red-panda.mp4
```

The one-shot file binds SGLang only to loopback, chooses a per-job port, uses
the speed-first single-H100 profile, and always stops its child server on normal
exit, failure, cancellation, or time-limit warning. The job requests 32 CPU
cores, 256 GB host RAM, and four hours by default; command-line `sbatch` options
override these headers.

## Long-running server job

For several interactive requests in the same allocation, submit the existing
server-only job instead:

```bash
export H3_REPO_DIR="$PWD"
export H3_MODEL_PATH=/path/to/models/MiniMax-H3
export H3_PROFILE=auto

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  --gpus-per-node=h100:1 \
  --export=ALL \
  deploy/slurm/h3-sglang.sbatch fl2va
```

Use your site's equivalent of `--gres=gpu:h100:1` if it does not support
`--gpus-per-node`. Keep both model partitions on one node only if the node has
enough suitable GPUs. With one H100, run FL2VA and Ref2VA as separate jobs.
The auto-selected `h100x1` SGLang profile uses layerwise CPU offload, so keep the
256 GB host-memory request and expect substantially higher latency than the
official four-H100 resident profile.

Find the compute node and inference port:

```bash
squeue -j JOB_ID -o '%.18i %.20N %.10T'
```

If the gateway runs on the login node and the site allows login-to-compute
traffic, set `H3_FL2VA_URL=http://COMPUTE_NODE:30010`. Otherwise run the gateway
inside the same allocation and create a two-hop SSH tunnel according to your
site policy. Never bind an unauthenticated inference port to a public interface.

The job log contains the selected node, variant, profile, and SGLang startup
output. A successful server reports its health endpoint before accepting video
jobs.

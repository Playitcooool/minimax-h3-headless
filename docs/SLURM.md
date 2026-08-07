# Slurm deployment

The batch file intentionally omits account, partition, and GPU directives
because their names differ between clusters. Pass those site-specific resources
to `sbatch`.

```bash
cd /path/to/minimax-h3-headless
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

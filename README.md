# MiniMax H3 Headless

An SSH-first deployment kit for running
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) without ComfyUI on a
Linux GPU server or Slurm cluster.

The repository is being assembled in verified stages. Its public request model
tracks H3's official `t2va`, `fl2va`, and `ref2va` modes, 4–15 second duration,
768-pixel short edge, and asynchronous video job workflow.

Reference counts are validated by the gateway. Reference media format, file
size, and actual 2–15 second clip duration are validated by the inference
server, which can inspect the referenced file; they cannot be proven from a URL
string alone.

## Architecture

```text
Agent / laptop
    │  SSH tunnel + API key
    ▼
H3 gateway (this repo)
    ├── FL2VA backend: t2va + first/last-frame generation
    └── Ref2VA backend: image/video/audio reference generation
             │
             ▼
       H.264 + stereo AAC MP4
```

The gateway defaults to loopback. Do not expose the raw inference ports or the
gateway directly to the public internet.

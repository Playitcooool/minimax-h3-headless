"""Minimal blocking client example for an agent pipeline."""

import os

from minimax_h3_headless.client import H3Client


payload = {
    "task": "t2va",
    "prompt": (
        "A cinematic close shot of a red panda making tea in a quiet cabin; "
        "soft rain and kettle sounds, no music."
    ),
    "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
    "seed": 42,
}

with H3Client("http://127.0.0.1:8080", os.environ["H3_GATEWAY_API_KEY"]) as client:
    result = client.generate(payload, "outputs/red-panda.mp4", timeout=3600)
    print(result)

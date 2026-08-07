"""Small synchronous client intended for scripts and agent tools."""

import time
from pathlib import Path
from typing import Any

import httpx

from .backends import RoutedJob


class H3Client:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30) -> None:
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "H3Client":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post("/v1/generations", json=payload)
        response.raise_for_status()
        return response.json()

    def status(self, job_id: str) -> dict[str, Any]:
        safe_job_id = RoutedJob.parse(job_id).public_id
        response = self.client.get(f"/v1/generations/{safe_job_id}")
        response.raise_for_status()
        return response.json()

    def wait(self, job_id: str, poll_interval: float = 2, timeout: float = 3600) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.status(job_id)
            if job["status"] == "completed":
                return job
            if job["status"] in {"failed", "cancelled"}:
                raise RuntimeError(f"generation {job_id} ended with {job['status']}: {job.get('error')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"generation {job_id} did not finish within {timeout} seconds")

    def download(self, job_id: str, output: str | Path) -> Path:
        safe_job_id = RoutedJob.parse(job_id).public_id
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.client.stream("GET", f"/v1/generations/{safe_job_id}/content") as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return destination

    def generate(self, payload: dict[str, Any], output: str | Path, **wait_options: Any) -> Path:
        job = self.submit(payload)
        self.wait(job["id"], **wait_options)
        return self.download(job["id"], output)

"""Public request/response contracts based on MiniMax H3's official Video API."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Task(StrEnum):
    T2VA = "t2va"
    FL2VA = "fl2va"
    REF2VA = "ref2va"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Condition(StrictModel):
    type: Literal["image", "video", "video_audio", "audio"]
    uri: str = Field(min_length=1)
    role: Literal["keyframe", "reference"]
    frame_index: Literal[0, -1] | None = None
    start_time_seconds: Annotated[float, Field(ge=0)] | None = None


class Target(StrictModel):
    short_edge: Literal[768] = 768
    aspect_ratio: Literal["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "16:9"
    duration_seconds: Annotated[float, Field(ge=4, le=15)] = 5


class GenerateRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=7000)
    task: Task = Task.T2VA
    conditions: list[Condition] = Field(default_factory=list, max_length=12)
    target: Target = Field(default_factory=Target)
    seed: int = Field(default=0, ge=0)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    flow_shift: float = 12.0
    audio_flow_shift: float = 3.0
    num_outputs_per_prompt: int = Field(default=1, ge=1, le=10)

    @model_validator(mode="after")
    def validate_task_conditions(self) -> "GenerateRequest":
        if self.task is Task.T2VA and self.conditions:
            raise ValueError("t2va does not accept media conditions")
        if self.task is Task.FL2VA:
            if not 1 <= len(self.conditions) <= 2:
                raise ValueError("fl2va requires one or two keyframe images")
            if any(c.type != "image" or c.role != "keyframe" for c in self.conditions):
                raise ValueError("fl2va conditions must be keyframe images")
            indices = {c.frame_index for c in self.conditions}
            if None in indices or len(indices) != len(self.conditions):
                raise ValueError("fl2va frame_index must be unique and either 0 or -1")
        if self.task is Task.REF2VA:
            if not self.conditions:
                raise ValueError("ref2va requires at least one reference")
            if any(c.role != "reference" for c in self.conditions):
                raise ValueError("ref2va conditions must use role=reference")
            has_audio = any(c.type == "audio" for c in self.conditions)
            has_visual = any(c.type in {"image", "video", "video_audio"} for c in self.conditions)
            if has_audio and not has_visual:
                raise ValueError("audio cannot be the sole ref2va reference")
            image_count = sum(c.type == "image" for c in self.conditions)
            video_count = sum(c.type in {"video", "video_audio"} for c in self.conditions)
            audio_count = sum(c.type == "audio" for c in self.conditions)
            if image_count > 9:
                raise ValueError("ref2va accepts at most 9 images")
            if video_count > 3:
                raise ValueError("ref2va accepts at most 3 videos")
            if audio_count > 3:
                raise ValueError("ref2va accepts at most 3 standalone audio clips")
        return self


class JobAccepted(StrictModel):
    id: str
    status: str = "queued"
    backend: str


class JobStatus(StrictModel):
    id: str
    status: str
    error: str | None = None
    content_url: HttpUrl | str | None = None

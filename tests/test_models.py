"""Contract tests for the public MiniMax H3 gateway models."""

import pytest
from pydantic import ValidationError

from minimax_h3_headless.models import (
    Condition,
    GenerateRequest,
    JobAccepted,
    JobStatus,
    Target,
    Task,
)


def condition(
    media_type: str = "image",
    *,
    role: str = "reference",
    frame_index: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "type": media_type,
        "uri": f"file:///references/{media_type}",
        "role": role,
    }
    if frame_index is not None:
        value["frame_index"] = frame_index
    return value


def assert_invalid(**values: object) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="A valid prompt", **values)


def test_default_request_is_t2va() -> None:
    request = GenerateRequest(prompt="A quiet lake at sunrise")

    assert request.task is Task.T2VA
    assert request.conditions == []
    assert request.target == Target()


@pytest.mark.parametrize("duration", [4, 4.5, 15])
def test_duration_accepts_official_inclusive_range(duration: float) -> None:
    request = GenerateRequest(
        prompt="A quiet lake at sunrise",
        target={"duration_seconds": duration},
    )

    assert request.target.duration_seconds == duration


@pytest.mark.parametrize("duration", [0, 3.99, 15.01, 100])
def test_duration_rejects_values_outside_official_range(duration: float) -> None:
    assert_invalid(target={"duration_seconds": duration})


def test_t2va_rejects_all_media_conditions() -> None:
    assert_invalid(task="t2va", conditions=[condition()])


@pytest.mark.parametrize("frame_index", [0, -1])
def test_fl2va_accepts_a_first_or_last_keyframe(frame_index: int) -> None:
    request = GenerateRequest(
        prompt="Continue this frame",
        task="fl2va",
        conditions=[
            condition("image", role="keyframe", frame_index=frame_index)
        ],
    )

    assert request.conditions[0].frame_index == frame_index


def test_fl2va_accepts_both_distinct_keyframes() -> None:
    request = GenerateRequest(
        prompt="Move naturally between the supplied frames",
        task="fl2va",
        conditions=[
            condition("image", role="keyframe", frame_index=0),
            condition("image", role="keyframe", frame_index=-1),
        ],
    )

    assert {item.frame_index for item in request.conditions} == {0, -1}


@pytest.mark.parametrize(
    "conditions",
    [
        [],
        [condition("image", role="keyframe")],
        [condition("video", role="keyframe", frame_index=0)],
        [condition("image", role="reference", frame_index=0)],
        [
            condition("image", role="keyframe", frame_index=0),
            condition("image", role="keyframe", frame_index=0),
        ],
        [
            condition("image", role="keyframe", frame_index=0),
            condition("image", role="keyframe", frame_index=-1),
            condition("image", role="keyframe", frame_index=0),
        ],
    ],
    ids=[
        "no-keyframe",
        "missing-index",
        "non-image",
        "wrong-role",
        "duplicate-index",
        "too-many-keyframes",
    ],
)
def test_fl2va_rejects_invalid_condition_shapes(
    conditions: list[dict[str, object]],
) -> None:
    assert_invalid(task="fl2va", conditions=conditions)


@pytest.mark.parametrize("media_type", ["image", "video", "video_audio"])
def test_ref2va_accepts_a_visual_reference(media_type: str) -> None:
    request = GenerateRequest(
        prompt="Use the supplied reference",
        task="ref2va",
        conditions=[condition(media_type)],
    )

    assert request.conditions[0].type == media_type


def test_ref2va_accepts_mixed_visual_and_audio_references() -> None:
    request = GenerateRequest(
        prompt="Use all supplied references",
        task="ref2va",
        conditions=[condition(), condition("video"), condition("audio")],
    )

    assert len(request.conditions) == 3


@pytest.mark.parametrize(
    "conditions",
    [
        [],
        [condition("image", role="keyframe", frame_index=0)],
        [condition("audio")],
        [condition("audio"), condition("audio")],
    ],
    ids=["empty", "wrong-role", "audio-only", "multiple-audio-only"],
)
def test_ref2va_rejects_missing_visual_reference(
    conditions: list[dict[str, object]],
) -> None:
    assert_invalid(task="ref2va", conditions=conditions)


def test_ref2va_accepts_official_per_modality_maxima() -> None:
    conditions = (
        [condition("image") for _ in range(6)]
        + [condition("video") for _ in range(3)]
        + [condition("audio") for _ in range(3)]
    )

    request = GenerateRequest(
        prompt="Use all supplied references",
        task="ref2va",
        conditions=conditions,
    )

    assert len(request.conditions) == 12


@pytest.mark.parametrize(
    "conditions",
    [
        [condition("image") for _ in range(9)],
        [condition("video") for _ in range(3)],
        [condition("image")] + [condition("audio") for _ in range(3)],
    ],
    ids=["nine-images", "three-videos", "three-audios-with-visual"],
)
def test_ref2va_accepts_each_official_per_modality_boundary(
    conditions: list[dict[str, object]],
) -> None:
    request = GenerateRequest(
        prompt="Use all supplied references",
        task="ref2va",
        conditions=conditions,
    )

    assert len(request.conditions) == len(conditions)


@pytest.mark.parametrize(
    "conditions",
    [
        [condition("image") for _ in range(10)],
        [condition("video") for _ in range(4)],
        [condition("image")] + [condition("audio") for _ in range(4)],
        [condition("image") for _ in range(9)]
        + [condition("video") for _ in range(3)]
        + [condition("audio")],
    ],
    ids=["ten-images", "four-videos", "four-audios", "thirteen-files"],
)
def test_ref2va_rejects_official_input_limit_violations(
    conditions: list[dict[str, object]],
) -> None:
    assert_invalid(task="ref2va", conditions=conditions)


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_prompt_rejects_empty_or_whitespace_only_values(value: str) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt=value)


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_condition_uri_rejects_empty_or_whitespace_only_values(value: str) -> None:
    with pytest.raises(ValidationError):
        Condition(type="image", uri=value, role="reference")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_outputs_per_prompt", 0),
        ("num_outputs_per_prompt", 11),
        ("num_inference_steps", 0),
        ("num_inference_steps", 101),
        ("seed", -1),
    ],
)
def test_sampling_control_bounds(field: str, value: int) -> None:
    assert_invalid(**{field: value})


def test_official_ten_output_limit_is_accepted() -> None:
    request = GenerateRequest(prompt="Ten variants", num_outputs_per_prompt=10)

    assert request.num_outputs_per_prompt == 10


@pytest.mark.parametrize(
    ("model", "values"),
    [
        (GenerateRequest, {"prompt": "valid", "duration_second": 5}),
        (Condition, {"type": "image", "uri": "file:///x", "role": "reference", "frame_indx": 0}),
        (Target, {"short_edge": 768, "duration_second": 5}),
        (JobAccepted, {"id": "1", "backend": "sglang", "unexpected": True}),
        (JobStatus, {"id": "1", "status": "queued", "unexpected": True}),
    ],
)
def test_public_contracts_forbid_unknown_fields(
    model: type[object], values: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model(**values)

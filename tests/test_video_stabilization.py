from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from utils.video_stabilization import (
    StabilizationConfig,
    StabilizationError,
    estimate_trajectory,
    stabilize_frames,
    stabilize_video,
)


def _config(**overrides) -> StabilizationConfig:
    values = {
        "smoothing_sec": 0.1,
        "min_inliers": 20,
        "min_inlier_ratio": 0.5,
        "max_reprojection_error": 5.0,
        "protect_center_width": 0.2,
        "protect_center_height": 0.2,
        "border_margin": 4,
        "max_crop_ratio": 0.4,
    }
    values.update(overrides)
    return StabilizationConfig(**values)


def _background() -> np.ndarray:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:] = (35, 50, 70)
    for y in range(10, 230, 20):
        cv2.line(image, (8, y), (312, y), (180, 190, 200), 1)
    for x in range(10, 310, 20):
        cv2.line(image, (x, 8), (x, 232), (180, 190, 200), 1)
    for x, y in [(25, 35), (70, 180), (130, 55), (190, 200), (270, 40), (295, 160)]:
        cv2.circle(image, (x, y), 6, (20, 230, 100), -1)
    return image


def _translated_frames(shifts: list[tuple[float, float]]) -> list[np.ndarray]:
    base = _background()
    height, width = base.shape[:2]
    return [
        cv2.warpAffine(
            base,
            np.asarray([[1.0, 0.0, x], [0.0, 1.0, y]], dtype=np.float32),
            (width, height),
            borderMode=cv2.BORDER_REFLECT,
        )
        for x, y in shifts
    ]


def test_estimate_trajectory_recovers_known_translation() -> None:
    frames = _translated_frames([(0, 0), (2, 1), (4, 2), (6, 3)])
    transforms, smoothed, diagnostics = estimate_trajectory(frames, 30.0, _config())

    assert transforms.shape == (4, 3, 3)
    assert smoothed.shape == (4, 3, 3)
    assert len(diagnostics) == 3
    np.testing.assert_allclose(transforms[-1, :2, 2], [6, 3], atol=0.5)
    assert min(item["inliers"] for item in diagnostics) >= 20
    assert min(item["inlier_ratio"] for item in diagnostics) >= 0.5


def test_identity_frames_are_not_changed() -> None:
    frame = _background()
    stabilized, diagnostics = stabilize_frames([frame, frame.copy(), frame.copy()], 30.0, _config())

    assert diagnostics["safe_crop_margin_px"] == 0
    assert all(np.array_equal(frame, output) for output in stabilized)


def test_featureless_video_fails_quality_gate() -> None:
    frame = np.full((120, 160, 3), 80, dtype=np.uint8)

    with pytest.raises(StabilizationError, match="features"):
        stabilize_frames([frame, frame.copy()], 30.0, _config())


def test_mp4_output_preserves_video_contract_and_writes_diagnostics(tmp_path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "stabilized.mp4"
    diagnostics_path = tmp_path / "stabilized.json"
    frames = _translated_frames([(0, 0), (1, 0), (2, 0), (3, 0)])
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (320, 240),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()

    diagnostics = stabilize_video(input_path, output_path, diagnostics_path, _config())

    assert diagnostics["frame_count"] == len(frames)
    assert diagnostics_path.is_file()
    assert output_path.is_file()
    saved = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert saved["format"] == "sam-body4d-stabilization-v1"
    capture = cv2.VideoCapture(str(output_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
    capture.release()

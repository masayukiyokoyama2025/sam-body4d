"""Offline 2D video stabilization for the SAM-Body4D input path."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class StabilizationError(RuntimeError):
    """Raised when a video cannot pass the stabilization quality gate."""


@dataclass(frozen=True)
class StabilizationConfig:
    smoothing_sec: float
    min_inliers: int = 40
    min_inlier_ratio: float = 0.5
    max_reprojection_error: float = 20.0
    max_features: int = 800
    quality_level: float = 0.01
    min_distance: float = 8.0
    border_margin: int = 12
    protect_center_width: float = 0.45
    protect_center_height: float = 0.65
    max_crop_ratio: float = 0.25
    max_processing_dimension: int = 960

    def validate(self) -> None:
        if self.smoothing_sec <= 0:
            raise ValueError("smoothing_sec must be greater than zero")
        if self.min_inliers < 3:
            raise ValueError("min_inliers must be at least 3")
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in (0, 1]")
        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be greater than zero")
        if not 0 <= self.max_crop_ratio < 0.5:
            raise ValueError("max_crop_ratio must be in [0, 0.5)")


def _as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    result[:2] = matrix
    return result


def _to_affine(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:2], dtype=np.float64)


def _resize_for_estimation(gray: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = gray.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale == 1.0:
        return gray, scale
    resized = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _feature_mask(shape: tuple[int, int], config: StabilizationConfig) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    margin = min(config.border_margin, max(0, min(height, width) // 4))
    if margin:
        mask[:margin] = 0
        mask[-margin:] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
    center_width = int(width * config.protect_center_width)
    center_height = int(height * config.protect_center_height)
    x0 = max(0, (width - center_width) // 2)
    y0 = max(0, (height - center_height) // 2)
    mask[y0 : min(height, y0 + center_height), x0 : min(width, x0 + center_width)] = 0
    return mask


def _scale_affine_translation(matrix: np.ndarray, scale: float) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    if scale != 1.0:
        result[:, 2] /= scale
    return result


def _estimate_pair(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    config: StabilizationConfig,
) -> tuple[np.ndarray, dict]:
    previous_small, scale = _resize_for_estimation(
        previous_gray, config.max_processing_dimension
    )
    current_small, _ = _resize_for_estimation(
        current_gray, config.max_processing_dimension
    )
    points = cv2.goodFeaturesToTrack(
        previous_small,
        maxCorners=config.max_features,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance,
        mask=_feature_mask(previous_small.shape, config),
        blockSize=7,
    )
    detected_count = 0 if points is None else len(points)
    if points is None or detected_count < 3:
        raise StabilizationError(
            f"insufficient background features: detected={detected_count}"
        )

    tracked, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_small,
        current_small,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        raise StabilizationError("optical flow returned no tracked points")
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= np.isfinite(errors.reshape(-1))
    previous_points = points.reshape(-1, 2)[valid]
    current_points = tracked.reshape(-1, 2)[valid]
    tracked_count = len(previous_points)
    if tracked_count < 3:
        raise StabilizationError(f"insufficient tracked points: tracked={tracked_count}")

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        raise StabilizationError("RANSAC could not estimate a similarity transform")
    matrix = _scale_affine_translation(matrix, scale)
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / tracked_count
    predicted = cv2.transform(previous_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - current_points, axis=1)
    inlier_residuals = residuals[inliers]
    reprojection_error = float(np.median(inlier_residuals)) if len(inlier_residuals) else float("inf")
    if inlier_count < config.min_inliers:
        raise StabilizationError(
            f"RANSAC inliers below threshold: inliers={inlier_count}, "
            f"required={config.min_inliers}"
        )
    if inlier_ratio < config.min_inlier_ratio:
        raise StabilizationError(
            f"RANSAC inlier ratio below threshold: ratio={inlier_ratio:.3f}, "
            f"required={config.min_inlier_ratio:.3f}"
        )
    if reprojection_error > config.max_reprojection_error:
        raise StabilizationError(
            f"RANSAC reprojection error above threshold: error={reprojection_error:.3f}, "
            f"required<={config.max_reprojection_error:.3f}"
        )

    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale_value = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return matrix, {
        "detected_features": detected_count,
        "tracked_features": tracked_count,
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "reprojection_error_px": reprojection_error,
        "translation_px": [float(matrix[0, 2]), float(matrix[1, 2])],
        "rotation_deg": rotation,
        "scale": scale_value,
    }


def _trajectory_parameters(transforms: np.ndarray) -> np.ndarray:
    parameters = np.empty((len(transforms), 4), dtype=np.float64)
    parameters[:, 0] = transforms[:, 0, 2]
    parameters[:, 1] = transforms[:, 1, 2]
    parameters[:, 2] = np.unwrap(
        np.arctan2(transforms[:, 1, 0], transforms[:, 0, 0])
    )
    scales = np.hypot(transforms[:, 0, 0], transforms[:, 1, 0])
    parameters[:, 3] = np.log(np.maximum(scales, 1e-12))
    return parameters


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    half = window // 2
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])],
        axis=1,
    )


def _parameters_to_transforms(parameters: np.ndarray) -> np.ndarray:
    transforms = np.repeat(np.eye(3, dtype=np.float64)[None], len(parameters), axis=0)
    scales = np.exp(parameters[:, 3])
    cosines = np.cos(parameters[:, 2]) * scales
    sines = np.sin(parameters[:, 2]) * scales
    transforms[:, 0, 0] = cosines
    transforms[:, 0, 1] = -sines
    transforms[:, 1, 0] = sines
    transforms[:, 1, 1] = cosines
    transforms[:, 0, 2] = parameters[:, 0]
    transforms[:, 1, 2] = parameters[:, 1]
    return transforms


def estimate_trajectory(
    frames: Iterable[np.ndarray],
    fps: float,
    config: StabilizationConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    config.validate()
    frame_list = list(frames)
    if len(frame_list) < 2:
        raise StabilizationError("at least two frames are required")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    shape = frame_list[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frame_list):
        raise StabilizationError("all frames must have the same dimensions")

    transforms = [np.eye(3, dtype=np.float64)]
    diagnostics = []
    previous_gray = cv2.cvtColor(frame_list[0], cv2.COLOR_BGR2GRAY)
    for frame_index, frame in enumerate(frame_list[1:], start=1):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pair_transform, pair_diagnostic = _estimate_pair(previous_gray, current_gray, config)
        transforms.append(_as_homogeneous(pair_transform) @ transforms[-1])
        pair_diagnostic["frame_index"] = frame_index
        diagnostics.append(pair_diagnostic)
        previous_gray = current_gray

    transform_array = np.asarray(transforms)
    parameters = _trajectory_parameters(transform_array)
    window = max(1, int(round(config.smoothing_sec * fps)))
    if window % 2 == 0:
        window += 1
    smoothed = _moving_average(parameters, window)
    smoothed_array = _parameters_to_transforms(smoothed)
    return transform_array, smoothed_array, diagnostics


def _warp_matrices(transforms: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    result = []
    for transform, smooth in zip(transforms, smoothed):
        result.append(transform @ np.linalg.inv(smooth))
    return np.asarray(result)


def _crop_margin(warp_matrices: np.ndarray, width: int, height: int, config: StabilizationConfig) -> int:
    max_margin = min(width, height) // 2 - 2
    if max_margin < 0:
        raise StabilizationError("video dimensions are too small for safe cropping")
    for margin in range(max_margin + 1):
        corners = np.asarray(
            [
                [margin, margin],
                [width - 1 - margin, margin],
                [margin, height - 1 - margin],
                [width - 1 - margin, height - 1 - margin],
            ],
            dtype=np.float64,
        )
        valid = True
        for matrix in warp_matrices:
            source_corners = cv2.transform(corners.reshape(-1, 1, 2), _to_affine(matrix)).reshape(-1, 2)
            if np.any(source_corners[:, 0] < 0) or np.any(source_corners[:, 0] >= width):
                valid = False
                break
            if np.any(source_corners[:, 1] < 0) or np.any(source_corners[:, 1] >= height):
                valid = False
                break
        if valid:
            crop_ratio = margin / min(width, height)
            if crop_ratio > config.max_crop_ratio:
                raise StabilizationError(
                    f"safe crop exceeds threshold: ratio={crop_ratio:.3f}, "
                    f"required<={config.max_crop_ratio:.3f}"
                )
            return margin
    raise StabilizationError("no common valid crop exists for the estimated transforms")


def stabilize_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    config: StabilizationConfig,
) -> tuple[list[np.ndarray], dict]:
    frame_list = list(frames)
    if not frame_list:
        raise StabilizationError("input contains no frames")
    height, width = frame_list[0].shape[:2]
    transforms, smoothed, pair_diagnostics = estimate_trajectory(frame_list, fps, config)
    warp_matrices = _warp_matrices(transforms, smoothed)
    margin = _crop_margin(warp_matrices, width, height, config)
    stabilized = []
    for frame, matrix in zip(frame_list, warp_matrices):
        warped = cv2.warpAffine(
            frame,
            _to_affine(matrix),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if margin:
            warped = warped[margin : height - margin, margin : width - margin]
            warped = cv2.resize(warped, (width, height), interpolation=cv2.INTER_LINEAR)
        stabilized.append(warped)
    diagnostics = {
        "format": "sam-body4d-stabilization-v1",
        "fps": float(fps),
        "frame_count": len(frame_list),
        "width": width,
        "height": height,
        "parameters": asdict(config),
        "safe_crop_margin_px": margin,
        "safe_crop_ratio": margin / min(width, height),
        "pairs": pair_diagnostics,
    }
    return stabilized, diagnostics


def _assert_output_outside_repo(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        path.relative_to(repo_root)
    except ValueError:
        return
    raise ValueError(f"stabilization output must be outside the public repository: {path}")


def stabilize_video(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    diagnostics_path: str | os.PathLike[str],
    config: StabilizationConfig,
) -> dict:
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    diagnostics_file = Path(diagnostics_path).expanduser().resolve()
    if input_file.suffix.lower() != ".mp4":
        raise ValueError("stabilization input must be an .mp4 file")
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if input_file == output_file:
        raise ValueError("stabilization output must differ from the input video")
    _assert_output_outside_repo(output_file)
    _assert_output_outside_repo(diagnostics_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_file.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_file))
    if not capture.isOpened():
        raise StabilizationError(f"could not open input video: {input_file}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if fps <= 0:
        raise StabilizationError("input video does not provide a valid FPS")
    stabilized, diagnostics = stabilize_frames(frames, fps, config)

    height, width = stabilized[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise StabilizationError(f"could not open output video: {output_file}")
    try:
        for frame in stabilized:
            writer.write(frame)
    finally:
        writer.release()

    output_capture = cv2.VideoCapture(str(output_file))
    output_count = int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(output_capture.get(cv2.CAP_PROP_FPS))
    output_width = int(output_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(output_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_capture.release()
    if output_count != len(frames) or output_width != width or output_height != height:
        raise StabilizationError("output video metadata does not match input")
    diagnostics.update(
        {
            "input_path": str(input_file),
            "output_path": str(output_file),
            "diagnostics_path": str(diagnostics_file),
            "output_fps": output_fps,
            "processing_seconds": time.perf_counter() - started,
        }
    )
    with diagnostics_file.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return diagnostics
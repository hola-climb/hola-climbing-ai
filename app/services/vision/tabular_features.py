"""Deterministic tabular pose features for dynamic/static baselines."""

from __future__ import annotations

from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

TabularFeatureVariant = Literal["exact", "normalized", "velocity_only"]

_LANDMARKS: Final[int] = 33
_CHANNELS: Final[int] = 4
_FLAT_CHANNELS: Final[int] = _LANDMARKS * _CHANNELS
_XYZ_CHANNELS: Final[int] = _LANDMARKS * 3
_LEFT_SHOULDER: Final[int] = 11
_RIGHT_SHOULDER: Final[int] = 12
_LEFT_HIP: Final[int] = 23
_RIGHT_HIP: Final[int] = 24
_EPS: Final[float] = 1e-6


def pose_json_frames_to_array(frames: list[dict[str, object]]) -> NDArray[np.float32]:
    """Convert `/hola_ind` pose JSON frames into `(T, 33, 4)` arrays."""
    if not frames:
        raise ValueError("frames must not be empty")

    rows: list[list[list[float]]] = []
    for frame in frames:
        keypoints = frame.get("keypoints")
        if not isinstance(keypoints, list) or len(keypoints) != _LANDMARKS:
            raise ValueError("each frame must contain 33 keypoints")

        landmarks: list[list[float]] = []
        for keypoint in keypoints:
            if not isinstance(keypoint, dict):
                raise ValueError("each keypoint must be an object")
            landmarks.append(
                [
                    float(keypoint["x"]),
                    float(keypoint["y"]),
                    float(keypoint["z"]),
                    float(keypoint["v"]),
                ]
            )
        rows.append(landmarks)

    return cast(NDArray[np.float32], np.asarray(rows, dtype=np.float32))


def extract_tabular_pose_features(
    pose: NDArray[np.float32],
    *,
    variant: TabularFeatureVariant,
) -> NDArray[np.float32]:
    """Extract fixed-width tabular features from `(T, 33, 4)` pose frames."""
    arr = _validate_pose(pose)
    if variant == "velocity_only":
        return _velocity_summary(_true_xyz(arr))

    feature_pose = arr
    if variant == "normalized":
        feature_pose = _normalize_pose(arr)
        velocity_source = _true_xyz(feature_pose)
    elif variant == "exact":
        flat = feature_pose.reshape(feature_pose.shape[0], _FLAT_CHANNELS)
        velocity_source = flat[:, :_XYZ_CHANNELS]
    else:
        raise ValueError(f"unsupported tabular feature variant: {variant}")

    flat = feature_pose.reshape(feature_pose.shape[0], _FLAT_CHANNELS)
    features = np.concatenate([_position_summary(flat), _velocity_summary(velocity_source)])
    return cast(NDArray[np.float32], features.astype(np.float32))


def _validate_pose(pose: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(pose, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (_LANDMARKS, _CHANNELS):
        raise ValueError(f"pose must have shape (T, 33, 4), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("pose must contain at least one frame")
    return arr


def _position_summary(flat_pose: NDArray[np.float32]) -> NDArray[np.float32]:
    summary = np.concatenate(
        [
            np.mean(flat_pose, axis=0),
            np.std(flat_pose, axis=0),
            np.min(flat_pose, axis=0),
            np.max(flat_pose, axis=0),
        ]
    )
    return cast(NDArray[np.float32], summary.astype(np.float32))


def _velocity_summary(coords: NDArray[np.float32]) -> NDArray[np.float32]:
    if coords.shape[0] < 2:
        return np.zeros(8, dtype=np.float32)

    velocity_vec = np.diff(coords, axis=0)
    velocity_mag = np.linalg.norm(velocity_vec, axis=1)
    if velocity_mag.size == 0:
        return np.zeros(8, dtype=np.float32)

    sorted_velocity = np.sort(velocity_mag)
    top_5_count = int(len(velocity_mag) * 0.05)
    top_10_count = max(1, int(len(velocity_mag) * 0.1))
    threshold = np.percentile(velocity_mag, 70)
    is_move = velocity_mag > threshold

    max_streak = 0
    current_streak = 0
    for moving in is_move:
        if bool(moving):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    features = np.asarray(
        [
            np.mean(velocity_mag),
            np.std(velocity_mag),
            np.max(velocity_mag),
            np.mean(sorted_velocity[-top_5_count:]),
            np.max(velocity_mag),
            np.mean(sorted_velocity[-top_10_count:]),
            np.sum(is_move) / len(velocity_mag),
            max_streak / len(velocity_mag),
        ],
        dtype=np.float32,
    )
    return features


def _true_xyz(pose: NDArray[np.float32]) -> NDArray[np.float32]:
    return pose[:, :, :3].reshape(pose.shape[0], _XYZ_CHANNELS)


def _normalize_pose(pose: NDArray[np.float32]) -> NDArray[np.float32]:
    coords = pose[:, :, :3]
    visibility = pose[:, :, 3:4]

    hip_center = coords[:, [_LEFT_HIP, _RIGHT_HIP], :].mean(axis=1, keepdims=True)
    shoulder_center = coords[:, [_LEFT_SHOULDER, _RIGHT_SHOULDER], :].mean(axis=1, keepdims=True)
    torso = np.linalg.norm((shoulder_center - hip_center)[:, :, :2], axis=2, keepdims=True)
    torso = np.maximum(torso, _EPS)

    normalized = (coords - hip_center) / torso
    return cast(
        NDArray[np.float32],
        np.concatenate([normalized, visibility], axis=2).astype(np.float32),
    )

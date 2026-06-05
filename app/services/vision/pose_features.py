"""Feature transforms for learned pose sequence classifiers."""

from __future__ import annotations

from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

FeatureSet = Literal["raw", "motion"]

_LANDMARKS: Final[int] = 33
_RAW_CHANNELS: Final[int] = 4
_MOTION_CHANNELS: Final[int] = 11
_LEFT_SHOULDER: Final[int] = 11
_RIGHT_SHOULDER: Final[int] = 12
_LEFT_HIP: Final[int] = 23
_RIGHT_HIP: Final[int] = 24
_EPS: Final[float] = 1e-6


def feature_size(feature_set: str) -> int:
    """Return flattened per-frame feature size for a feature set."""
    if feature_set == "raw":
        return _LANDMARKS * _RAW_CHANNELS
    if feature_set == "motion":
        return _LANDMARKS * _MOTION_CHANNELS
    raise ValueError(f"unsupported feature_set: {feature_set}")


def prepare_pose_features(
    x: NDArray[np.float32],
    *,
    feature_set: str,
) -> NDArray[np.float32]:
    """Transform raw flattened pose sequence into model-ready features."""
    raw = _validate_raw_input(x)
    if feature_set == "raw":
        return raw.copy()
    if feature_set != "motion":
        raise ValueError(f"unsupported feature_set: {feature_set}")

    pose = raw.reshape(raw.shape[0], _LANDMARKS, _RAW_CHANNELS)
    coords = pose[:, :, :3]
    visibility = pose[:, :, 3:4]

    hip_center = coords[:, [_LEFT_HIP, _RIGHT_HIP], :].mean(axis=1, keepdims=True)
    shoulder_center = coords[:, [_LEFT_SHOULDER, _RIGHT_SHOULDER], :].mean(axis=1, keepdims=True)
    torso = np.linalg.norm((shoulder_center - hip_center)[:, :, :2], axis=2, keepdims=True)
    torso = np.maximum(torso, _EPS)

    normalized = (coords - hip_center) / torso
    velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    speed = np.linalg.norm(velocity[:, :, :2], axis=2, keepdims=True)

    features = np.concatenate(
        [normalized, visibility, velocity, acceleration, speed],
        axis=2,
    )
    return cast(NDArray[np.float32], features.reshape(raw.shape[0], feature_size("motion")).astype(np.float32))


def _validate_raw_input(x: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != feature_size("raw"):
        raise ValueError(f"x must have shape (T, 132), got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("x must contain at least one frame")
    return arr

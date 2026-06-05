"""Tabular pose feature extraction for dynamic/static baselines."""

from __future__ import annotations

import numpy as np

from app.services.vision.tabular_features import (
    extract_tabular_pose_features,
    pose_json_frames_to_array,
)


def test_exact_feature_shape_is_hola_ind_compatible() -> None:
    pose = np.zeros((5, 33, 4), dtype=np.float32)
    pose[:, :, 0] = np.arange(33, dtype=np.float32)
    pose[:, :, 1] = np.arange(33, dtype=np.float32) + 100
    pose[:, :, 2] = np.arange(33, dtype=np.float32) + 200
    pose[:, :, 3] = 1.0

    features = extract_tabular_pose_features(pose, variant="exact")

    assert features.shape == (536,)


def test_velocity_only_removes_position_summary() -> None:
    pose = np.zeros((6, 33, 4), dtype=np.float32)
    pose[:, :, :3] = np.linspace(0.0, 1.0, num=6, dtype=np.float32)[:, None, None]
    pose[:, :, 3] = 1.0

    features = extract_tabular_pose_features(pose, variant="velocity_only")

    assert features.shape == (8,)
    assert features[2] > 0.0


def test_pose_json_frames_to_array_reads_keypoints() -> None:
    frames = [
        {"keypoints": [{"x": 1.0, "y": 2.0, "z": 3.0, "v": 0.5} for _ in range(33)]},
        {"keypoints": [{"x": 2.0, "y": 3.0, "z": 4.0, "v": 0.6} for _ in range(33)]},
    ]

    arr = pose_json_frames_to_array(frames)

    assert arr.shape == (2, 33, 4)
    assert arr[0, 0].tolist() == [1.0, 2.0, 3.0, 0.5]

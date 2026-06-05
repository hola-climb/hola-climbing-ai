"""Feature transforms for learned pose sequence classifiers."""

from __future__ import annotations

import numpy as np

from app.services.vision.pose_features import feature_size, prepare_pose_features


def _sequence(*, offset: float, scale: float) -> np.ndarray:
    pose = np.zeros((3, 33, 4), dtype=np.float32)
    for t in range(3):
        hip_x = offset + scale * float(t)
        hip_y = offset
        pose[t, :, :3] = (hip_x, hip_y, 0.0)
        pose[t, 23, :3] = (hip_x - scale, hip_y, 0.0)
        pose[t, 24, :3] = (hip_x + scale, hip_y, 0.0)
        pose[t, 11, :3] = (hip_x - scale, hip_y + 2.0 * scale, 0.0)
        pose[t, 12, :3] = (hip_x + scale, hip_y + 2.0 * scale, 0.0)
        pose[t, 0, :3] = (hip_x, hip_y + 3.0 * scale, 0.0)
        pose[t, :, 3] = 1.0
    return pose.reshape(3, 132)


def test_raw_feature_set_returns_float32_passthrough() -> None:
    x = np.arange(2 * 132, dtype=np.float32).reshape(2, 132)

    out = prepare_pose_features(x, feature_set="raw")

    assert out.shape == (2, feature_size("raw"))
    assert out.dtype == np.float32
    assert np.array_equal(out, x)


def test_motion_feature_set_adds_normalized_motion_channels() -> None:
    x = _sequence(offset=10.0, scale=2.0)

    out = prepare_pose_features(x, feature_set="motion")

    assert out.shape == (3, feature_size("motion"))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert feature_size("motion") == 363


def test_motion_features_are_invariant_to_translation_and_scale() -> None:
    base = prepare_pose_features(_sequence(offset=10.0, scale=2.0), feature_set="motion")
    shifted = prepare_pose_features(_sequence(offset=-50.0, scale=8.0), feature_set="motion")

    assert np.allclose(base, shifted, atol=1e-5)


def test_motion_features_start_with_zero_velocity_and_acceleration() -> None:
    out = prepare_pose_features(_sequence(offset=10.0, scale=2.0), feature_set="motion")
    per_landmark = out.reshape(3, 33, 11)

    assert np.allclose(per_landmark[0, :, 4:7], 0.0)
    assert np.allclose(per_landmark[0, :, 7:10], 0.0)

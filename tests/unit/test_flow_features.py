"""Optical-flow feature extraction for dynamic/static baselines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.services.vision.flow_features import (
    FLOW_FEATURE_DIM,
    V3_FLOW_FEATURE_DIM,
    extract_flow_series,
    extract_flow_stats,
    extract_flow_stats_legacy,
    extract_flow_stats_v3,
    remove_fall_end,
    trim_fall_segment,
)


def test_trim_fall_segment_trims_contiguous_terminal_burst() -> None:
    signal = np.asarray([1.0] * 40 + [4.8, 5.1, 5.4, 5.2], dtype=np.float32)
    trimmed = trim_fall_segment(signal, spike_multiplier=3.5, max_trim_ratio=0.25)
    assert trimmed.tolist() == [1.0] * 40


def test_trim_fall_segment_ignores_single_terminal_spike() -> None:
    signal = np.asarray([1.0] * 40 + [5.5], dtype=np.float32)
    trimmed = trim_fall_segment(signal, spike_multiplier=3.5, min_tail_frames=2)
    assert trimmed is signal


def test_trim_fall_segment_respects_max_trim_ratio() -> None:
    signal = np.asarray([1.0] * 75 + [5.0] * 25, dtype=np.float32)
    trimmed = trim_fall_segment(signal, spike_multiplier=3.5, max_trim_ratio=0.10)
    assert len(signal) - len(trimmed) == 10


def test_remove_fall_end_keeps_backward_compatible_wrapper() -> None:
    signal = np.asarray([1.0] * 40 + [5.0, 5.2, 5.1], dtype=np.float32)
    trimmed = remove_fall_end(signal)
    assert len(trimmed) < len(signal)


def test_extract_flow_series_tracks_vertical_direction(tmp_path: Path) -> None:
    down_video = tmp_path / "down.avi"
    up_video = tmp_path / "up.avi"
    _write_translated_texture_video(down_video, step_y=1)
    _write_translated_texture_video(up_video, step_y=-1)

    down_series, _fps, _duration = extract_flow_series(down_video, resize=(64, 64), target_fps=10)
    up_series, _fps, _duration = extract_flow_series(up_video, resize=(64, 64), target_fps=10)

    # OpenCV image coordinates use +y downward.
    assert float(down_series[:, 1].mean()) > 0.0
    assert float(up_series[:, 1].mean()) < 0.0


def test_extract_flow_stats_returns_v4_features() -> None:
    magnitude = np.linspace(0.1, 1.0, num=90, dtype=np.float32)
    vy = np.linspace(-0.2, 0.2, num=90, dtype=np.float32)
    signal = np.stack([magnitude, vy], axis=1)
    features = extract_flow_stats(signal)
    assert features.shape == (FLOW_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_extract_flow_stats_v3_returns_46_features() -> None:
    signal = np.linspace(0.1, 1.0, num=90, dtype=np.float32)
    features = extract_flow_stats_v3(signal)
    assert features.shape == (V3_FLOW_FEATURE_DIM,)
    assert np.isfinite(features).all()


def test_extract_flow_stats_legacy_returns_42_features() -> None:
    signal = np.linspace(0.1, 1.0, num=90, dtype=np.float32)
    features = extract_flow_stats_legacy(signal)
    assert features.shape == (42,)
    assert np.isfinite(features).all()


def test_extract_flow_stats_includes_burst_aware_features() -> None:
    signal = np.ones(260, dtype=np.float32)
    signal[30:90] = 5.0
    signal[170:230] = 5.0

    features = extract_flow_stats_v3(signal)

    max_window_mean, top3_window_mean, burst_count, burst_duration_ratio = features[-4:]
    assert max_window_mean > 4.0
    assert top3_window_mean > 4.0
    assert burst_count >= 2.0
    assert 0.0 < burst_duration_ratio < 1.0


def test_extract_flow_stats_includes_vy_direction_features() -> None:
    magnitude = np.ones(260, dtype=np.float32)
    vy = np.zeros(260, dtype=np.float32)
    vy[30:90] = -3.0
    vy[170:230] = 3.0
    features = extract_flow_stats(np.stack([magnitude, vy], axis=1))

    (
        _vy_mean,
        _vy_std,
        vy_min,
        vy_max,
        _vy_p10,
        _vy_p90,
        upward_count,
        upward_ratio,
        downward_count,
        downward_ratio,
        max_upward_window_mean,
        max_downward_window_mean,
    ) = features[-12:]
    assert vy_min < 0.0
    assert vy_max > 0.0
    assert upward_count >= 1.0
    assert downward_count >= 1.0
    assert upward_ratio > 0.0
    assert downward_ratio > 0.0
    assert max_upward_window_mean > 0.0
    assert max_downward_window_mean > 0.0


def _write_translated_texture_video(path: Path, *, step_y: int) -> None:
    rng = np.random.default_rng(42)
    base = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    assert writer.isOpened()
    try:
        for idx in range(12):
            matrix = np.float32([[1, 0, 0], [0, 1, step_y * idx]])
            shifted = cv2.warpAffine(base, matrix, (64, 64), borderMode=cv2.BORDER_REPLICATE)
            frame = cv2.cvtColor(shifted, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
    finally:
        writer.release()

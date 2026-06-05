"""Optical-flow feature extraction for dynamic/static baselines."""

from __future__ import annotations

import numpy as np

from app.services.vision.flow_features import extract_flow_stats, remove_fall_end


def test_remove_fall_end_trims_tail_spike() -> None:
    signal = np.asarray([1.0] * 20 + [100.0, 120.0], dtype=np.float32)
    trimmed = remove_fall_end(signal, tail_ratio=0.1)
    assert len(trimmed) < len(signal)


def test_extract_flow_stats_returns_42_features() -> None:
    signal = np.linspace(0.1, 1.0, num=90, dtype=np.float32)
    features = extract_flow_stats(signal)
    assert features.shape == (42,)
    assert np.isfinite(features).all()

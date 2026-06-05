"""Learned pose sequence classifier helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.services.vision.model_classifier import (  # noqa: E402
    PoseSequenceClassifier,
    load_checkpoint,
    predict_dynamic_probability,
    save_checkpoint,
)


def test_pose_sequence_classifier_forward_shape() -> None:
    model = PoseSequenceClassifier(input_size=132, hidden_size=8, num_layers=1)
    x = torch.zeros((4, 16, 132), dtype=torch.float32)

    logits = model(x)

    assert tuple(logits.shape) == (4,)


def test_checkpoint_roundtrip_predicts_probability(tmp_path: Path) -> None:
    model = PoseSequenceClassifier(input_size=132, hidden_size=8, num_layers=1)
    checkpoint = tmp_path / "pose_dynamic_static.pt"

    save_checkpoint(
        checkpoint,
        model=model,
        target_frames=16,
        hidden_size=8,
        num_layers=1,
        metrics={"accuracy": 0.5},
    )
    loaded = load_checkpoint(checkpoint)
    x = np.zeros((16, 132), dtype=np.float32)

    prob = predict_dynamic_probability(loaded.model, x)

    assert 0.0 <= prob <= 1.0
    assert loaded.target_frames == 16
    assert loaded.metrics == {"accuracy": 0.5}


def test_checkpoint_roundtrip_preserves_input_size_and_feature_set(tmp_path: Path) -> None:
    model = PoseSequenceClassifier(input_size=363, hidden_size=8, num_layers=1)
    checkpoint = tmp_path / "pose_dynamic_static_motion.pt"

    save_checkpoint(
        checkpoint,
        model=model,
        target_frames=16,
        hidden_size=8,
        num_layers=1,
        input_size=363,
        feature_set="motion",
        metrics={"accuracy": 0.6},
    )
    loaded = load_checkpoint(checkpoint)

    assert loaded.input_size == 363
    assert loaded.feature_set == "motion"

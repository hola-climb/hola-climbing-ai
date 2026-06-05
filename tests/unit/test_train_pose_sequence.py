"""Training helper behavior for pose sequence classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_pose_sequence import (
    binary_metrics,
    load_npz_dataset,
    stratified_kfold,
    write_reports,
)


def _write_sample(path: Path, *, label: int, raw_pose_frames: int) -> None:
    x = np.zeros((4, 132), dtype=np.float32)
    x[:, 23 * 4 : 23 * 4 + 4] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    x[:, 24 * 4 : 24 * 4 + 4] = np.array([2.0, 0.0, 0.0, 1.0], dtype=np.float32)
    x[:, 11 * 4 : 11 * 4 + 4] = np.array([0.0, 2.0, 0.0, 1.0], dtype=np.float32)
    x[:, 12 * 4 : 12 * 4 + 4] = np.array([2.0, 2.0, 0.0, 1.0], dtype=np.float32)
    np.savez_compressed(
        path,
        x=x,
        label=np.asarray(label, dtype=np.int64),
        stem=np.asarray(path.stem),
        raw_pose_frames=np.asarray(raw_pose_frames, dtype=np.int64),
    )


def test_load_npz_dataset_filters_low_pose_frame_samples(tmp_path: Path) -> None:
    _write_sample(tmp_path / "keep.npz", label=1, raw_pose_frames=40)
    _write_sample(tmp_path / "skip.npz", label=0, raw_pose_frames=4)

    dataset = load_npz_dataset(tmp_path, feature_set="motion", min_raw_pose_frames=30)

    assert dataset.x.shape == (1, 4, 363)
    assert dataset.y.tolist() == [1.0]
    assert dataset.stems == ["keep"]
    assert dataset.skipped_low_pose_frames == [("skip", 4)]


def test_stratified_kfold_covers_every_sample_once_per_validation() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32)

    splits = stratified_kfold(labels, folds=3, seed=7)

    valid_indices = sorted(idx for split in splits for idx in split.valid_idx)
    assert valid_indices == [0, 1, 2, 3, 4, 5]
    assert all({int(labels[i]) for i in split.valid_idx} == {0, 1} for split in splits)


def test_binary_metrics_reports_specificity_and_balanced_accuracy() -> None:
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1], dtype=np.float32),
        np.asarray([0.1, 0.8, 0.7, 0.2], dtype=np.float32),
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision_dynamic"] == 0.5
    assert metrics["recall_dynamic"] == 0.5
    assert metrics["specificity_static"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_write_reports_creates_predictions_csv_and_metrics_json(tmp_path: Path) -> None:
    rows = [
        {
            "fold": 1,
            "split": "valid",
            "stem": "sample",
            "label": 1,
            "prob_dynamic": 0.7,
            "pred": 1,
            "correct": True,
            "raw_pose_frames": 42,
        }
    ]
    summary = {"samples": 1, "feature_set": "motion"}

    outputs = write_reports(tmp_path, run_name="run", prediction_rows=rows, summary=summary)

    assert outputs.predictions_csv.read_text(encoding="utf-8").splitlines()[0].startswith("fold,split")
    assert '"feature_set": "motion"' in outputs.metrics_json.read_text(encoding="utf-8")

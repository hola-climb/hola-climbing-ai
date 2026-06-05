"""Training helpers for tabular dynamic/static baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_tabular_dynamic_static import (
    canonical_group,
    evaluate_predictions,
    load_tabular_dataset,
    stratified_group_splits,
)


def test_canonical_group_strips_duplicate_suffix() -> None:
    assert canonical_group("IMG_3445 (1)") == "IMG_3445"
    assert canonical_group("IMG_3445") == "IMG_3445"


def test_evaluate_predictions_reports_balanced_accuracy() -> None:
    metrics = evaluate_predictions(
        y_true=np.asarray([0, 0, 1, 1]),
        y_prob=np.asarray([0.1, 0.8, 0.7, 0.2]),
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision_dynamic"] == 0.5
    assert metrics["recall_dynamic"] == 0.5


def test_stratified_group_splits_keep_duplicate_groups_together() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    groups = ["A", "A (1)", "B", "B (1)", "C", "D"]

    splits = stratified_group_splits(labels, groups, folds=2, seed=42)

    assert len(splits) == 2
    for split in splits:
        train_groups = {canonical_group(groups[i]) for i in split.train_idx}
        valid_groups = {canonical_group(groups[i]) for i in split.valid_idx}
        assert train_groups.isdisjoint(valid_groups)
        assert split.train_idx
        assert split.valid_idx


def test_load_tabular_dataset_prefers_manifest_order(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    for stem, label in [("A", 0), ("B", 1)]:
        np.savez_compressed(
            data_dir / f"{stem}.npz",
            x=np.asarray([float(label)], dtype=np.float32),
            label=np.asarray(label, dtype=np.int64),
            stem=np.asarray(stem),
            source_path=np.asarray(f"{stem}.json"),
        )
    manifest = tmp_path / "dataset_manifest.csv"
    manifest.write_text(
        "stem,label,variant,feature_dim,source_path,out_path\n"
        f"B,1,exact,1,B.json,{data_dir / 'B.npz'}\n"
        f"A,0,exact,1,A.json,{data_dir / 'A.npz'}\n",
        encoding="utf-8",
    )

    dataset = load_tabular_dataset(data_dir)

    assert dataset.stems == ["B", "A"]

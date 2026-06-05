"""Pose/flow fusion dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.build_fusion_dataset import build_fusion_dataset


def _write_npz(root: Path, stem: str, label: int, values: list[float]) -> None:
    np.savez_compressed(
        root / f"{stem}.npz",
        x=np.asarray(values, dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        stem=np.asarray(stem),
        source_path=np.asarray(f"{stem}.json"),
        variant=np.asarray("test"),
    )


def test_build_fusion_dataset_joins_matching_stems(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    out = tmp_path / "out"
    left.mkdir()
    right.mkdir()
    _write_npz(left, "A", 1, [1.0, 2.0])
    _write_npz(left, "B", 0, [3.0, 4.0])
    _write_npz(right, "A", 1, [5.0])
    _write_npz(right, "C", 0, [6.0])

    result = build_fusion_dataset(left_dir=left, right_dir=right, out_dir=out)

    assert result.written == 1
    assert result.missing_right == ["B"]
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert data["x"].tolist() == [1.0, 2.0, 5.0]
        assert int(data["label"]) == 1

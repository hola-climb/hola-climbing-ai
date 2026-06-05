"""Pose JSON to tabular dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_pose_tabular_dataset import build_tabular_dataset


def _write_pose_json(path: Path, frames: int = 4) -> None:
    payload = [
        {
            "keypoints": [
                {"x": float(i), "y": float(i + 1), "z": float(i + 2), "v": 1.0}
                for i in range(33)
            ]
        }
        for _ in range(frames)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_tabular_dataset_from_pose_json(tmp_path: Path) -> None:
    pose_dir = tmp_path / "pose_json"
    pose_dir.mkdir()
    _write_pose_json(pose_dir / "A.json")
    _write_pose_json(pose_dir / "B.json")
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nA.json,0\nB.json,1\nC.json,\n", encoding="utf-8")
    out = tmp_path / "out"

    result = build_tabular_dataset(
        labels_csv=labels,
        pose_json_dir=pose_dir,
        out_dir=out,
        variant="exact",
    )

    assert result.written == 2
    assert result.missing == []
    assert sorted(p.name for p in out.glob("*.npz")) == ["A.npz", "B.npz"]
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert data["x"].shape == (536,)
        assert int(data["label"]) == 0

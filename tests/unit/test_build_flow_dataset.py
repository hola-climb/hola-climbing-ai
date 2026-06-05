"""Optical-flow dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_flow_dataset import build_flow_dataset


def test_build_flow_dataset_from_labeled_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "A.mp4").write_bytes(b"video")
    (videos / "B.mov").write_bytes(b"video")
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nA.json,0\nB.json,1\nC.json,\n", encoding="utf-8")
    out = tmp_path / "out"

    def fake_extract_flow_magnitude(
        video_path: Path,
        *,
        resize: tuple[int, int] = (320, 240),
        target_fps: int = 30,
    ) -> tuple[np.ndarray, float, float]:
        assert video_path.name in {"A.mp4", "B.mov"}
        assert resize == (320, 240)
        assert target_fps == 30
        return np.linspace(0.1, 1.0, num=90, dtype=np.float32), 30.0, 3.0

    monkeypatch.setattr(
        "scripts.build_flow_dataset.extract_flow_magnitude",
        fake_extract_flow_magnitude,
    )

    result = build_flow_dataset(labels_csv=labels, videos_dir=videos, out_dir=out)

    assert result.written == 2
    assert result.missing == []
    assert sorted(p.name for p in out.glob("*.npz")) == ["A.npz", "B.npz"]
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert data["x"].shape == (42,)
        assert int(data["label"]) == 0

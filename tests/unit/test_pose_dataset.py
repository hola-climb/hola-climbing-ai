"""Pose dataset preprocessing for learned dynamic/static classifier."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.services.vision.pose import PoseFrame
from app.services.vision.pose_dataset import (
    build_model_input,
    load_label_rows,
    match_labeled_videos,
    normalize_label,
    pose_frames_to_array,
    resample_pose_array,
)


def _write_labels(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)


def _pose_frame(frame_idx: int, value: float) -> PoseFrame:
    landmarks = np.full((33, 4), value, dtype=np.float32)
    return PoseFrame(frame_idx=frame_idx, timestamp_ms=frame_idx * 100, landmarks=landmarks)


def test_normalize_label_supports_binary_and_text_values() -> None:
    assert normalize_label("0") == 0
    assert normalize_label("static") == 0
    assert normalize_label("s") == 0
    assert normalize_label("1") == 1
    assert normalize_label("dynamic") == 1
    assert normalize_label("d") == 1
    assert normalize_label("") is None
    assert normalize_label("unknown") is None


def test_load_label_rows_ignores_unlabeled_rows(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(
        labels,
        [
            ("IMG_0001.json", "0"),
            ("IMG_0002.json", ""),
            ("IMG_0003.json", "1"),
        ],
    )

    rows = load_label_rows(labels)

    assert rows == [("IMG_0001", 0), ("IMG_0003", 1)]


def test_match_labeled_videos_by_stem_and_supported_extension(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    videos = tmp_path / "videos"
    videos.mkdir()
    _write_labels(labels, [("IMG_0001.json", "0"), ("IMG_0002.json", "1")])
    (videos / "IMG_0001.mov").write_bytes(b"video")
    (videos / "IMG_0002.MOV").write_bytes(b"video")

    matched, missing = match_labeled_videos(labels, videos)

    assert missing == []
    assert [(m.stem, m.label, m.video_path.name) for m in matched] == [
        ("IMG_0001", 0, "IMG_0001.mov"),
        ("IMG_0002", 1, "IMG_0002.MOV"),
    ]


def test_pose_frames_to_array_preserves_pose_shape() -> None:
    arr = pose_frames_to_array([_pose_frame(0, 1.0), _pose_frame(1, 2.0)])

    assert arr.shape == (2, 33, 4)
    assert arr.dtype == np.float32
    assert float(arr[0, 0, 0]) == 1.0
    assert float(arr[1, 0, 0]) == 2.0


def test_resample_pose_array_interpolates_to_target_frames() -> None:
    source = np.stack(
        [
            np.full((33, 4), 0.0, dtype=np.float32),
            np.full((33, 4), 10.0, dtype=np.float32),
        ],
        axis=0,
    )

    out = resample_pose_array(source, target_frames=5)

    assert out.shape == (5, 33, 4)
    assert np.allclose(out[:, 0, 0], np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32))


def test_resample_pose_array_repeats_single_frame() -> None:
    source = np.full((1, 33, 4), 3.0, dtype=np.float32)

    out = resample_pose_array(source, target_frames=4)

    assert out.shape == (4, 33, 4)
    assert np.allclose(out, 3.0)


def test_build_model_input_returns_flattened_sequence() -> None:
    x = build_model_input([_pose_frame(0, 1.0), _pose_frame(1, 2.0)], target_frames=8)

    assert x.shape == (8, 132)
    assert x.dtype == np.float32

"""QA review queue generation for dynamic/static classifier outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.build_dynamic_static_review_queue import build_review_rows, write_review_csv


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["fold", "split", "stem", "label", "prob_dynamic", "pred", "correct", "raw_pose_frames"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(stem: str, label: int, prob: float, pred: int, correct: bool, frames: int = 100) -> dict[str, object]:
    return {
        "fold": 1,
        "split": "valid",
        "stem": stem,
        "label": label,
        "prob_dynamic": prob,
        "pred": pred,
        "correct": correct,
        "raw_pose_frames": frames,
    }


def _write_cached_sample(path: Path, *, label: int, raw_pose_frames: int) -> None:
    np.savez_compressed(
        path,
        x=np.zeros((4, 132), dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        stem=np.asarray(path.stem),
        raw_pose_frames=np.asarray(raw_pose_frames, dtype=np.int64),
    )


def test_build_review_rows_prioritizes_failures_common_misses_and_review_samples(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    motion_csv = tmp_path / "motion.csv"
    data_dir = tmp_path / "data"
    videos_dir = tmp_path / "videos"
    data_dir.mkdir()
    videos_dir.mkdir()
    (videos_dir / "both.mov").write_bytes(b"video")
    (videos_dir / "raw_high.mov").write_bytes(b"video")
    (videos_dir / "correct.mov").write_bytes(b"video")
    (videos_dir / "low_pose.mov").write_bytes(b"video")
    (videos_dir / "pose_fail.mov").write_bytes(b"video")
    _write_cached_sample(data_dir / "low_pose.npz", label=0, raw_pose_frames=4)

    _write_predictions(
        raw_csv,
        [
            _row("both", 1, 0.2, 0, False),
            _row("raw_high", 0, 0.9, 1, False),
            _row("correct", 1, 0.8, 1, True),
        ],
    )
    _write_predictions(
        motion_csv,
        [
            _row("both", 1, 0.3, 0, False),
            _row("raw_high", 0, 0.2, 0, True),
            _row("correct", 1, 0.7, 1, True),
        ],
    )

    rows = build_review_rows(
        raw_predictions=raw_csv,
        motion_predictions=motion_csv,
        data_dir=data_dir,
        videos_dir=videos_dir,
        high_confidence_limit=5,
        correct_sample_count=1,
        known_failures=["pose_fail:no_pose_detected"],
        label_map={"pose_fail": 1},
    )

    assert [(row["priority"], row["review_group"], row["stem"]) for row in rows] == [
        ("P0", "pose_failure", "pose_fail"),
        ("P0", "low_pose_frames", "low_pose"),
        ("P1", "both_models_missed", "both"),
        ("P2", "raw_high_confidence_miss", "raw_high"),
        ("P4", "raw_correct_sample", "correct"),
    ]
    assert rows[0]["current_label"] == "1"
    assert rows[1]["reason"] == "video_too_short"
    assert rows[2]["suggested_status"] == "review"
    assert rows[3]["wrong_confidence"] == "0.900000"
    assert rows[4]["video_path"].endswith("correct.mov")


def test_write_review_csv_has_stable_columns(tmp_path: Path) -> None:
    out = tmp_path / "review.csv"

    write_review_csv(
        out,
        [
            {
                "priority": "P1",
                "review_group": "both_models_missed",
                "stem": "sample",
                "current_label": "1",
                "raw_prob": "0.2",
                "raw_pred": "0",
                "raw_correct": "False",
                "motion_prob": "0.3",
                "motion_pred": "0",
                "motion_correct": "False",
                "wrong_confidence": "0.8",
                "raw_pose_frames": "100",
                "suggested_status": "review",
                "new_label": "",
                "reason": "",
                "notes": "",
                "video_path": "/tmp/sample.mov",
            }
        ],
    )

    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("priority,review_group,stem,current_label")

"""High-confidence flow miss review queue."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_flow_miss_review_queue import build_flow_miss_review_rows


def _write_predictions(path: Path) -> None:
    rows = [
        {
            "model": "rf",
            "split": "group-kfold",
            "fold": "0",
            "stem": "STATIC_WRONG",
            "group": "STATIC_WRONG",
            "label": "0",
            "prob_dynamic": "0.91",
            "pred": "1",
            "correct": "False",
            "source_path": "/videos/STATIC_WRONG.mp4",
        },
        {
            "model": "rf",
            "split": "group-kfold",
            "fold": "1",
            "stem": "DYNAMIC_WRONG",
            "group": "DYNAMIC_WRONG",
            "label": "1",
            "prob_dynamic": "0.12",
            "pred": "0",
            "correct": "False",
            "source_path": "/videos/DYNAMIC_WRONG.mp4",
        },
        {
            "model": "rf",
            "split": "group-kfold",
            "fold": "2",
            "stem": "CORRECT",
            "group": "CORRECT",
            "label": "1",
            "prob_dynamic": "0.88",
            "pred": "1",
            "correct": "True",
            "source_path": "/videos/CORRECT.mp4",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_flow_miss_review_rows_sorts_by_wrong_confidence(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions)

    rows = build_flow_miss_review_rows(predictions_csv=predictions, model="rf", split="group-kfold")

    assert [row["stem"] for row in rows] == ["STATIC_WRONG", "DYNAMIC_WRONG"]
    assert rows[0]["miss_type"] == "false_positive_static"
    assert rows[0]["wrong_confidence"] == "0.910000"
    assert rows[1]["miss_type"] == "false_negative_dynamic"
    assert rows[1]["wrong_confidence"] == "0.880000"

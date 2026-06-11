"""Flow regression comparison."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.compare_flow_regression import compare_regression


def test_compare_regression_counts_recovered_and_new_misses(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_predictions(
        baseline,
        [
            ("A", 1, 0, 0.05),  # high-confidence miss, recovered
            ("B", 0, 1, 0.90),  # high-confidence miss, still wrong
            ("C", 1, 1, 0.80),  # correct, becomes wrong
        ],
    )
    _write_predictions(
        candidate,
        [
            ("A", 1, 1, 0.65),
            ("B", 0, 1, 0.70),
            ("C", 1, 0, 0.40),
        ],
    )

    summary = compare_regression(
        baseline_predictions=baseline,
        candidate_predictions=candidate,
        high_confidence_threshold=0.85,
    )

    assert summary.baseline_miss_count == 2
    assert summary.candidate_miss_count == 2
    assert summary.recovered_stems == ["A"]
    assert summary.newly_wrong_stems == ["C"]
    assert summary.high_confidence_baseline_miss_count == 2
    assert summary.high_confidence_recovered_stems == ["A"]


def _write_predictions(path: Path, rows: list[tuple[str, int, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "fold",
                "stem",
                "group",
                "label",
                "prob_dynamic",
                "pred",
                "correct",
                "source_path",
            ],
        )
        writer.writeheader()
        for stem, label, pred, prob_dynamic in rows:
            writer.writerow(
                {
                    "model": "rf",
                    "split": "group-kfold",
                    "fold": "0",
                    "stem": stem,
                    "group": stem,
                    "label": str(label),
                    "prob_dynamic": str(prob_dynamic),
                    "pred": str(pred),
                    "correct": str(label == pred),
                    "source_path": f"/videos/{stem}.mp4",
                }
            )

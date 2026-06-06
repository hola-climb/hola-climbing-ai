"""Build a high-confidence review queue from flow dynamic/static predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REVIEW_COLUMNS = [
    "priority",
    "stem",
    "label",
    "pred",
    "prob_dynamic",
    "wrong_confidence",
    "miss_type",
    "model",
    "split",
    "fold",
    "suggested_status",
    "new_label",
    "reason",
    "notes",
    "video_path",
]


def build_flow_miss_review_rows(
    *,
    predictions_csv: Path,
    model: str = "rf",
    split: str = "group-kfold",
    limit: int = 0,
) -> list[dict[str, str]]:
    """Return high-confidence wrong predictions sorted for human review."""
    rows: list[dict[str, str]] = []
    with predictions_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != model or row.get("split") != split:
                continue
            if row.get("correct") == "True":
                continue
            label = int(row["label"])
            pred = int(row["pred"])
            prob_dynamic = float(row["prob_dynamic"])
            wrong_confidence = prob_dynamic if label == 0 else 1.0 - prob_dynamic
            rows.append(
                {
                    "priority": "P1",
                    "stem": row["stem"],
                    "label": str(label),
                    "pred": str(pred),
                    "prob_dynamic": f"{prob_dynamic:.6f}",
                    "wrong_confidence": f"{wrong_confidence:.6f}",
                    "miss_type": _miss_type(label, pred),
                    "model": model,
                    "split": split,
                    "fold": row.get("fold", ""),
                    "suggested_status": "review",
                    "new_label": "",
                    "reason": "",
                    "notes": "",
                    "video_path": row.get("source_path", ""),
                }
            )
    rows.sort(key=lambda item: (-float(item["wrong_confidence"]), item["stem"]))
    return rows[:limit] if limit > 0 else rows


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REVIEW_COLUMNS})


def _miss_type(label: int, pred: int) -> str:
    if label == 0 and pred == 1:
        return "false_positive_static"
    if label == 1 and pred == 0:
        return "false_negative_dynamic"
    return "other_miss"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="rf")
    parser.add_argument("--split", default="group-kfold")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = build_flow_miss_review_rows(
        predictions_csv=args.predictions,
        model=args.model,
        split=args.split,
        limit=args.limit,
    )
    write_review_csv(args.out, rows)
    print(f"[summary] rows={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

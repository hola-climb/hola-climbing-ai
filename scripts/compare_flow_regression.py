"""Compare flow prediction regressions against a baseline miss set."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    stem: str
    label: int
    pred: int
    prob_dynamic: float

    @property
    def correct(self) -> bool:
        return self.label == self.pred

    @property
    def wrong_confidence(self) -> float:
        return self.prob_dynamic if self.label == 0 else 1.0 - self.prob_dynamic


@dataclass(frozen=True)
class RegressionSummary:
    baseline_miss_count: int
    candidate_miss_count: int
    recovered_count: int
    newly_wrong_count: int
    still_wrong_count: int
    high_confidence_baseline_miss_count: int
    high_confidence_recovered_count: int
    high_confidence_still_wrong_count: int
    recovered_stems: list[str]
    newly_wrong_stems: list[str]
    high_confidence_recovered_stems: list[str]


def compare_regression(
    *,
    baseline_predictions: Path,
    candidate_predictions: Path,
    model: str = "rf",
    split: str = "group-kfold",
    high_confidence_threshold: float = 0.85,
) -> RegressionSummary:
    """Compare baseline misses with candidate predictions by stem."""
    baseline = _load_predictions(baseline_predictions, model=model, split=split)
    candidate = _load_predictions(candidate_predictions, model=model, split=split)

    baseline_misses = {stem for stem, pred in baseline.items() if not pred.correct}
    candidate_misses = {stem for stem, pred in candidate.items() if not pred.correct}
    recovered = sorted(stem for stem in baseline_misses - candidate_misses if stem in candidate)
    newly_wrong = sorted(stem for stem in candidate_misses - baseline_misses if stem in baseline)
    still_wrong = sorted(baseline_misses & candidate_misses)

    high_confidence_misses = {
        stem
        for stem in baseline_misses
        if baseline[stem].wrong_confidence >= high_confidence_threshold
    }
    high_confidence_recovered = sorted(
        stem for stem in high_confidence_misses - candidate_misses if stem in candidate
    )
    high_confidence_still_wrong = sorted(high_confidence_misses & candidate_misses)

    return RegressionSummary(
        baseline_miss_count=len(baseline_misses),
        candidate_miss_count=len(candidate_misses),
        recovered_count=len(recovered),
        newly_wrong_count=len(newly_wrong),
        still_wrong_count=len(still_wrong),
        high_confidence_baseline_miss_count=len(high_confidence_misses),
        high_confidence_recovered_count=len(high_confidence_recovered),
        high_confidence_still_wrong_count=len(high_confidence_still_wrong),
        recovered_stems=recovered,
        newly_wrong_stems=newly_wrong,
        high_confidence_recovered_stems=high_confidence_recovered,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", default="rf")
    parser.add_argument("--split", default="group-kfold")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = compare_regression(
        baseline_predictions=args.baseline,
        candidate_predictions=args.candidate,
        model=args.model,
        split=args.split,
        high_confidence_threshold=args.high_confidence_threshold,
    )
    payload = asdict(summary)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


def _load_predictions(path: Path, *, model: str, split: str) -> dict[str, Prediction]:
    rows: dict[str, Prediction] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") != model or row.get("split") != split:
                continue
            stem = row["stem"]
            rows[stem] = Prediction(
                stem=stem,
                label=int(row["label"]),
                pred=int(row["pred"]),
                prob_dynamic=float(row["prob_dynamic"]),
            )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())

"""Propagate completed review decisions into a newer review queue."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REVIEW_FIELDS = ["suggested_status", "new_label", "reason", "notes"]
VALID_COMPLETED_STATUSES = {"keep", "fix_label", "exclude", "ambiguous"}


@dataclass(frozen=True)
class AppliedDecision:
    stem: str
    source_path: str
    suggested_status: str
    new_label: str
    reason: str
    notes: str


@dataclass(frozen=True)
class PropagateResult:
    rows: list[dict[str, str]]
    applied: list[AppliedDecision]
    skipped_completed: list[str]
    conflict_count: int
    source_counts: dict[str, int]

    @property
    def applied_count(self) -> int:
        return len(self.applied)


def propagate_review_decisions(
    target_csv: Path,
    source_csvs: list[Path],
    *,
    overwrite_completed: bool = False,
) -> PropagateResult:
    """Copy completed review decisions from source CSVs into target review rows.

    Earlier source CSVs have higher precedence. By default, target rows already
    marked as a completed decision are left untouched.
    """
    target_rows = _read_csv(target_csv)
    source_by_stem, conflicts = _load_source_decisions(source_csvs)
    applied: list[AppliedDecision] = []
    skipped_completed: list[str] = []
    source_counter: Counter[str] = Counter()

    out_rows: list[dict[str, str]] = []
    for row in target_rows:
        out_row = dict(row)
        stem = (out_row.get("stem") or "").strip()
        decision = source_by_stem.get(stem)
        if decision is None:
            out_rows.append(out_row)
            continue

        current_status = (out_row.get("suggested_status") or "").strip().lower()
        if not overwrite_completed and current_status and current_status != "review":
            skipped_completed.append(stem)
            out_rows.append(out_row)
            continue

        for field in REVIEW_FIELDS:
            out_row[field] = getattr(decision, field)
        applied.append(decision)
        source_counter[decision.source_path] += 1
        out_rows.append(out_row)

    return PropagateResult(
        rows=out_rows,
        applied=applied,
        skipped_completed=skipped_completed,
        conflict_count=conflicts,
        source_counts=dict(source_counter),
    )


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write review rows preserving the target CSV column order."""
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, applied: list[AppliedDecision]) -> None:
    """Write an audit CSV for rows filled from previous review decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stem", "source_path", *REVIEW_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for decision in applied:
            writer.writerow(
                {
                    "stem": decision.stem,
                    "source_path": decision.source_path,
                    "suggested_status": decision.suggested_status,
                    "new_label": decision.new_label,
                    "reason": decision.reason,
                    "notes": decision.notes,
                }
            )


def _load_source_decisions(source_csvs: list[Path]) -> tuple[dict[str, AppliedDecision], int]:
    decisions: dict[str, AppliedDecision] = {}
    conflicts = 0
    for source in source_csvs:
        for row in _read_csv(source):
            stem = (row.get("stem") or "").strip()
            status = (row.get("suggested_status") or "").strip().lower()
            if not stem or status not in VALID_COMPLETED_STATUSES:
                continue
            decision = AppliedDecision(
                stem=stem,
                source_path=str(source),
                suggested_status=status,
                new_label=(row.get("new_label") or "").strip(),
                reason=(row.get("reason") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
            existing = decisions.get(stem)
            if existing is None:
                decisions[stem] = decision
            elif _decision_key(existing) != _decision_key(decision):
                conflicts += 1
    return decisions, conflicts


def _decision_key(decision: AppliedDecision) -> tuple[str, str, str, str]:
    return (
        decision.suggested_status,
        decision.new_label,
        decision.reason,
        decision.notes,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--overwrite-completed", action="store_true")
    args = parser.parse_args()

    result = propagate_review_decisions(
        args.target,
        args.source,
        overwrite_completed=args.overwrite_completed,
    )
    write_review_csv(args.out, result.rows)
    if args.summary_out:
        write_summary_csv(args.summary_out, result.applied)

    print(
        "[summary] "
        f"target={args.target} applied={result.applied_count} "
        f"skipped_completed={len(result.skipped_completed)} conflicts={result.conflict_count}"
    )
    print(f"[summary] source_counts={result.source_counts}")
    if result.skipped_completed:
        print(f"[summary] skipped_completed_sample={result.skipped_completed[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply completed dynamic/static QA review decisions to labels and cached npz files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VALID_STATUSES = {"keep", "fix_label", "exclude", "ambiguous"}
VALID_LABELS = {"0", "1"}


@dataclass(frozen=True)
class ApplyResult:
    rows: list[dict[str, str]]
    final_labels: dict[str, int | None]
    status_counts: dict[str, int]
    changed_labels: dict[str, tuple[str, str]]
    excluded_stems: set[str]
    missing_label_stems: set[str]
    warnings: list[str]


def apply_review_decisions(labels_csv: Path, review_csv: Path) -> ApplyResult:
    """Apply review decisions to a labels CSV while preserving row order."""
    label_rows = _read_csv(labels_csv)
    review_rows = _read_csv(review_csv)
    reviews = _load_reviews(review_rows)
    label_stems = {_stem_from_filename(row.get("filename", "")) for row in label_rows}
    status_counts = Counter(row["suggested_status"] for row in reviews.values())
    missing_label_stems = set(reviews) - label_stems

    out_rows: list[dict[str, str]] = []
    final_labels: dict[str, int | None] = {}
    changed_labels: dict[str, tuple[str, str]] = {}
    excluded_stems: set[str] = set()
    warnings: list[str] = []

    for row in label_rows:
        out_row = dict(row)
        stem = _stem_from_filename(row.get("filename", ""))
        original = (row.get("label") or "").strip()
        final = original
        review = reviews.get(stem)
        if review is not None:
            status = review["suggested_status"]
            new_label = review["new_label"]
            if status == "fix_label":
                if new_label not in VALID_LABELS:
                    raise ValueError(f"{stem}: fix_label requires new_label 0 or 1")
                final = new_label
            elif status == "keep":
                if original not in VALID_LABELS and new_label in VALID_LABELS:
                    final = new_label
                elif new_label in VALID_LABELS and original in VALID_LABELS and new_label != original:
                    warnings.append(f"{stem}: keep has conflicting new_label={new_label}; kept original={original}")
            elif status in {"exclude", "ambiguous"}:
                final = ""
                excluded_stems.add(stem)

        out_row["label"] = final
        out_rows.append(out_row)
        final_labels[stem] = int(final) if final in VALID_LABELS else None
        if original in VALID_LABELS and final in VALID_LABELS and original != final:
            changed_labels[stem] = (original, final)

    return ApplyResult(
        rows=out_rows,
        final_labels=final_labels,
        status_counts=dict(sorted(status_counts.items())),
        changed_labels=changed_labels,
        excluded_stems=excluded_stems,
        missing_label_stems=missing_label_stems,
        warnings=warnings,
    )


def write_labels_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write labels rows with stable columns from the input rows."""
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_relabelled_cache(
    *,
    cache_in: Path,
    cache_out: Path,
    final_labels: dict[str, int | None],
) -> tuple[int, list[str]]:
    """Copy cached `.npz` files with updated labels, skipping blank labels."""
    cache_out.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: list[str] = []
    for stem, label in final_labels.items():
        source = cache_in / f"{stem}.npz"
        if label is None or not source.exists():
            skipped.append(stem)
            continue
        with np.load(source, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        payload["label"] = np.asarray(label, dtype=np.int64)
        payload.setdefault("stem", np.asarray(stem))
        np.savez_compressed(cache_out / f"{stem}.npz", **payload)
        written += 1
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--cache-in", type=Path)
    parser.add_argument("--cache-out", type=Path)
    args = parser.parse_args()

    result = apply_review_decisions(args.labels, args.review)
    write_labels_csv(args.labels_out, result.rows)
    print(f"[done] labels_out={args.labels_out}")
    print(f"[summary] statuses={result.status_counts}")
    print(f"[summary] changed_labels={len(result.changed_labels)} excluded_or_ambiguous={len(result.excluded_stems)}")
    if result.missing_label_stems:
        print(f"[warning] review stems missing from labels: {sorted(result.missing_label_stems)[:10]}")
    for warning in result.warnings:
        print(f"[warning] {warning}")

    if args.cache_in or args.cache_out:
        if not args.cache_in or not args.cache_out:
            print("[error] --cache-in and --cache-out must be provided together", file=sys.stderr)
            return 2
        if args.cache_out.exists() and any(args.cache_out.glob("*.npz")):
            print(f"[error] cache output already has npz files: {args.cache_out}", file=sys.stderr)
            return 2
        written, skipped = write_relabelled_cache(
            cache_in=args.cache_in,
            cache_out=args.cache_out,
            final_labels=result.final_labels,
        )
        print(f"[done] cache_out={args.cache_out}")
        print(f"[summary] cache_written={written} cache_skipped={len(skipped)}")
        if skipped:
            print(f"[summary] cache_skipped_sample={skipped[:10]}")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_reviews(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    reviews: dict[str, dict[str, str]] = {}
    for row in rows:
        stem = (row.get("stem") or "").strip()
        status = (row.get("suggested_status") or "").strip().lower()
        if not stem or not status:
            continue
        if status not in VALID_STATUSES:
            raise ValueError(f"{stem}: unknown suggested_status={status!r}")
        reviews[stem] = {
            "suggested_status": status,
            "new_label": (row.get("new_label") or "").strip(),
        }
    return reviews


def _stem_from_filename(filename: str) -> str:
    return Path(filename.strip()).stem


if __name__ == "__main__":
    raise SystemExit(main())

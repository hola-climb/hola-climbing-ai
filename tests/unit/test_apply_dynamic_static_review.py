"""Apply completed dynamic/static QA review decisions."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.apply_dynamic_static_review import (
    apply_review_decisions,
    write_labels_csv,
    write_relabelled_cache,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_npz(path: Path, *, label: int, stem: str, raw_pose_frames: int = 100) -> None:
    np.savez_compressed(
        path,
        x=np.ones((2, 4), dtype=np.float32),
        label=np.asarray(label, dtype=np.int64),
        stem=np.asarray(stem),
        raw_pose_frames=np.asarray(raw_pose_frames, dtype=np.int64),
    )


def test_apply_review_decisions_updates_labels_and_blanks_excluded_rows(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    review_csv = tmp_path / "review_complete.csv"
    out_csv = tmp_path / "labels_qa.csv"
    _write_csv(
        labels_csv,
        ["filename", "label"],
        [
            {"filename": "A.json", "label": "1"},
            {"filename": "B.json", "label": "0"},
            {"filename": "C.json", "label": "1"},
            {"filename": "D.json", "label": "0"},
            {"filename": "E.json", "label": ""},
        ],
    )
    _write_csv(
        review_csv,
        ["stem", "current_label", "suggested_status", "new_label", "reason"],
        [
            {"stem": "A", "current_label": "1", "suggested_status": "fix_label", "new_label": "0", "reason": "wrong_label"},
            {"stem": "B", "current_label": "0", "suggested_status": "ambiguous", "new_label": "", "reason": "ambiguous_move"},
            {"stem": "C", "current_label": "1", "suggested_status": "exclude", "new_label": "", "reason": "bad_crop"},
            {"stem": "D", "current_label": "0", "suggested_status": "Keep", "new_label": "0", "reason": "ok"},
        ],
    )

    result = apply_review_decisions(labels_csv, review_csv)
    write_labels_csv(out_csv, result.rows)

    rows = _read_csv(out_csv)
    assert {row["filename"]: row["label"] for row in rows} == {
        "A.json": "0",
        "B.json": "",
        "C.json": "",
        "D.json": "0",
        "E.json": "",
    }
    assert result.changed_labels == {"A": ("1", "0")}
    assert result.excluded_stems == {"B", "C"}
    assert result.status_counts == {"ambiguous": 1, "exclude": 1, "fix_label": 1, "keep": 1}


def test_write_relabelled_cache_updates_npz_labels_and_skips_blank_labels(tmp_path: Path) -> None:
    cache_in = tmp_path / "cache_in"
    cache_out = tmp_path / "cache_out"
    cache_in.mkdir()
    _write_npz(cache_in / "A.npz", label=1, stem="A")
    _write_npz(cache_in / "B.npz", label=0, stem="B")
    _write_npz(cache_in / "D.npz", label=0, stem="D")

    written, skipped = write_relabelled_cache(
        cache_in=cache_in,
        cache_out=cache_out,
        final_labels={"A": 0, "B": None, "D": 0, "missing": 1},
    )

    assert written == 2
    assert skipped == ["B", "missing"]
    assert sorted(path.name for path in cache_out.glob("*.npz")) == ["A.npz", "D.npz"]
    with np.load(cache_out / "A.npz", allow_pickle=False) as data:
        assert int(data["label"]) == 0

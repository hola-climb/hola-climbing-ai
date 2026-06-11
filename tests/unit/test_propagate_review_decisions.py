"""Propagate already completed review decisions into newer review queues."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.propagate_review_decisions import (
    propagate_review_decisions,
    write_review_csv,
    write_summary_csv,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_propagate_review_decisions_prefers_first_source_and_copies_review_fields(tmp_path: Path) -> None:
    flow_source = tmp_path / "flow_complete.csv"
    dynamic_source = tmp_path / "dynamic_complete.csv"
    target = tmp_path / "target.csv"
    out = tmp_path / "target_out.csv"
    summary = tmp_path / "summary.csv"

    _write_csv(
        flow_source,
        ["stem", "suggested_status", "new_label", "reason", "notes"],
        [
            {"stem": "A", "suggested_status": "fix_label", "new_label": "1", "reason": "flow reason", "notes": "flow note"},
        ],
    )
    _write_csv(
        dynamic_source,
        ["stem", "suggested_status", "new_label", "reason", "notes"],
        [
            {"stem": "A", "suggested_status": "keep", "new_label": "0", "reason": "dynamic conflict", "notes": ""},
            {"stem": "B", "suggested_status": "ambiguous", "new_label": "", "reason": "too close", "notes": "needs human"},
        ],
    )
    _write_csv(
        target,
        ["stem", "suggested_status", "new_label", "reason", "notes", "label"],
        [
            {"stem": "A", "suggested_status": "review", "new_label": "", "reason": "", "notes": "", "label": "0"},
            {"stem": "B", "suggested_status": "review", "new_label": "", "reason": "", "notes": "", "label": "1"},
            {"stem": "C", "suggested_status": "review", "new_label": "", "reason": "", "notes": "", "label": "0"},
        ],
    )

    result = propagate_review_decisions(target, [flow_source, dynamic_source])
    write_review_csv(out, result.rows)
    write_summary_csv(summary, result.applied)

    rows = {row["stem"]: row for row in _read_csv(out)}
    assert rows["A"]["suggested_status"] == "fix_label"
    assert rows["A"]["new_label"] == "1"
    assert rows["A"]["reason"] == "flow reason"
    assert rows["A"]["notes"] == "flow note"
    assert rows["B"]["suggested_status"] == "ambiguous"
    assert rows["B"]["reason"] == "too close"
    assert rows["C"]["suggested_status"] == "review"
    assert result.applied_count == 2
    assert result.source_counts == {str(flow_source): 1, str(dynamic_source): 1}
    assert result.conflict_count == 1
    summary_rows = _read_csv(summary)
    assert [row["stem"] for row in summary_rows] == ["A", "B"]
    assert summary_rows[0]["source_path"] == str(flow_source)


def test_propagate_review_decisions_does_not_overwrite_completed_rows_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    _write_csv(
        source,
        ["stem", "suggested_status", "new_label", "reason", "notes"],
        [{"stem": "A", "suggested_status": "fix_label", "new_label": "1", "reason": "source", "notes": ""}],
    )
    _write_csv(
        target,
        ["stem", "suggested_status", "new_label", "reason", "notes"],
        [{"stem": "A", "suggested_status": "keep", "new_label": "0", "reason": "manual", "notes": ""}],
    )

    result = propagate_review_decisions(target, [source])

    assert result.rows[0]["suggested_status"] == "keep"
    assert result.rows[0]["reason"] == "manual"
    assert result.applied_count == 0
    assert result.skipped_completed == ["A"]

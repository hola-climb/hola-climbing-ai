"""Compare worker segment dumps with Spring API JSON or DB CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.callback import AnalysisSegmentPayload


@dataclass(frozen=True)
class LoadedSegments:
    model_version: str | None
    segments: list[AnalysisSegmentPayload]


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"cannot parse boolean: {value!r}")


def _segment_from_mapping(row: Mapping[str, Any]) -> AnalysisSegmentPayload:
    return AnalysisSegmentPayload(
        sequence_index=int(_get(row, "sequence_index", "sequenceIndex")),
        start_time_ms=_to_int(_get(row, "start_time_ms", "startTimeMs")),
        end_time_ms=_to_int(_get(row, "end_time_ms", "endTimeMs")),
        technique=str(_get(row, "technique")),
        is_dynamic=_to_bool(_get(row, "is_dynamic", "isDynamic")),
        confidence=_to_float(_get(row, "confidence")),
    )


def _find_segments_container(data: Any) -> Mapping[str, Any]:
    if isinstance(data, Mapping):
        segments = data.get("segments")
        if isinstance(segments, list):
            return data
        for key in ("data", "result", "analysis", "analysisResult"):
            nested = data.get(key)
            if isinstance(nested, Mapping):
                try:
                    return _find_segments_container(nested)
                except ValueError:
                    pass
    raise ValueError("could not find a JSON object containing a segments array")


def load_segments_from_json(path: Path) -> LoadedSegments:
    """Load worker dump JSON or Spring ApiResponse JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    container = _find_segments_container(data)
    raw_segments = container["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError("segments must be a JSON array")
    model_version = _get(container, "model_version", "modelVersion")
    if model_version is None and isinstance(data, Mapping):
        model_version = _get(data, "model_version", "modelVersion")
    return LoadedSegments(
        model_version=str(model_version) if model_version is not None else None,
        segments=[_segment_from_mapping(row) for row in raw_segments],
    )


def load_segments_from_csv(path: Path) -> LoadedSegments:
    """Load `psql --csv` style DB export from analysis_results."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    model_versions = {
        str(value)
        for row in rows
        if (value := _get(row, "model_version", "modelVersion")) not in (None, "")
    }
    model_version = next(iter(model_versions)) if len(model_versions) == 1 else None
    return LoadedSegments(
        model_version=model_version,
        segments=[_segment_from_mapping(row) for row in rows],
    )


def _compare_value(
    mismatches: list[str],
    *,
    path: str,
    expected: Any,
    actual: Any,
) -> None:
    if expected != actual:
        mismatches.append(f"{path} expected={expected!r} actual={actual!r}")


def compare_loaded_segments(
    expected: LoadedSegments,
    actual: LoadedSegments,
    *,
    compare_model_version: bool = True,
) -> list[str]:
    """Return exact field mismatches between expected and actual segment payloads."""
    mismatches: list[str] = []
    if compare_model_version:
        _compare_value(
            mismatches,
            path="model_version",
            expected=expected.model_version,
            actual=actual.model_version,
        )

    if len(expected.segments) != len(actual.segments):
        mismatches.append(
            f"segments length expected={len(expected.segments)} actual={len(actual.segments)}"
        )

    fields = (
        "sequence_index",
        "start_time_ms",
        "end_time_ms",
        "technique",
        "is_dynamic",
        "confidence",
    )
    for idx, (exp, act) in enumerate(zip(expected.segments, actual.segments, strict=False)):
        exp_data = exp.model_dump(mode="json")
        act_data = act.model_dump(mode="json")
        for field in fields:
            _compare_value(
                mismatches,
                path=f"segment[{idx}].{field}",
                expected=exp_data[field],
                actual=act_data[field],
            )
    return mismatches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expected-dump", type=Path, required=True)
    actual = parser.add_mutually_exclusive_group(required=True)
    actual.add_argument("--actual-json", type=Path, help="Spring GET response or callback JSON.")
    actual.add_argument("--actual-csv", type=Path, help="DB CSV export from analysis_results.")
    parser.add_argument("--ignore-model-version", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected = load_segments_from_json(args.expected_dump)
    actual = (
        load_segments_from_json(args.actual_json)
        if args.actual_json is not None
        else load_segments_from_csv(args.actual_csv)
    )
    mismatches = compare_loaded_segments(
        expected,
        actual,
        compare_model_version=not args.ignore_model_version,
    )
    if mismatches:
        print("[mismatch]")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print(
        f"[ok] segments match count={len(expected.segments)} model_version={expected.model_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

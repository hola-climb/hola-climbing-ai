"""compare_analysis_segments script tests."""

from __future__ import annotations

import csv
from pathlib import Path


def test_load_segments_from_worker_dump_json(tmp_path: Path) -> None:
    from scripts.compare_analysis_segments import load_segments_from_json

    dump = tmp_path / "dump.json"
    dump.write_text(
        """
        {
          "model_version": "rule_v1",
          "segments": [
            {
              "sequence_index": 0,
              "start_time_ms": 100,
              "end_time_ms": 900,
              "technique": "high_step",
              "is_dynamic": false,
              "confidence": 0.8
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    loaded = load_segments_from_json(dump)

    assert loaded.model_version == "rule_v1"
    assert loaded.segments[0].technique == "high_step"


def test_load_segments_from_spring_api_response_with_camel_case(tmp_path: Path) -> None:
    from scripts.compare_analysis_segments import load_segments_from_json

    response = tmp_path / "response.json"
    response.write_text(
        """
        {
          "data": {
            "modelVersion": "rule_v1+flow_rf_v2",
            "segments": [
              {
                "sequenceIndex": 0,
                "startTimeMs": 100,
                "endTimeMs": 900,
                "technique": "dyno",
                "isDynamic": true,
                "confidence": 0.66
              }
            ]
          }
        }
        """,
        encoding="utf-8",
    )

    loaded = load_segments_from_json(response)

    assert loaded.model_version == "rule_v1+flow_rf_v2"
    assert loaded.segments[0].is_dynamic is True


def test_load_segments_from_db_csv(tmp_path: Path) -> None:
    from scripts.compare_analysis_segments import load_segments_from_csv

    path = tmp_path / "db.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence_index",
                "start_time_ms",
                "end_time_ms",
                "technique",
                "is_dynamic",
                "confidence",
                "model_version",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sequence_index": "0",
                "start_time_ms": "100",
                "end_time_ms": "900",
                "technique": "coordination",
                "is_dynamic": "true",
                "confidence": "0.77",
                "model_version": "rule_v1",
            }
        )

    loaded = load_segments_from_csv(path)

    assert loaded.model_version == "rule_v1"
    assert loaded.segments[0].technique == "coordination"


def test_compare_loaded_segments_reports_exact_mismatch() -> None:
    from app.models.callback import AnalysisSegmentPayload
    from scripts.compare_analysis_segments import LoadedSegments, compare_loaded_segments

    expected = LoadedSegments(
        model_version="rule_v1",
        segments=[
            AnalysisSegmentPayload(
                sequence_index=0,
                start_time_ms=100,
                end_time_ms=900,
                technique="dyno",
                is_dynamic=True,
                confidence=0.8,
            )
        ],
    )
    actual = LoadedSegments(
        model_version="rule_v1",
        segments=[
            AnalysisSegmentPayload(
                sequence_index=0,
                start_time_ms=100,
                end_time_ms=901,
                technique="dyno",
                is_dynamic=True,
                confidence=0.8,
            )
        ],
    )

    mismatches = compare_loaded_segments(expected, actual)

    assert mismatches == ["segment[0].end_time_ms expected=900 actual=901"]

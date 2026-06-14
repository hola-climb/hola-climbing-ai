"""render_technique_overlay helper tests."""

from __future__ import annotations

import json
from pathlib import Path

from app.models.callback import AnalysisSegmentPayload


def _seg(
    idx: int,
    technique: str,
    start: int,
    end: int,
    confidence: float,
) -> AnalysisSegmentPayload:
    return AnalysisSegmentPayload(
        sequence_index=idx,
        start_time_ms=start,
        end_time_ms=end,
        technique=technique,
        is_dynamic=technique in {"dyno", "coordination"},
        confidence=confidence,
    )


def test_find_active_segment_uses_half_open_interval() -> None:
    from scripts.render_technique_overlay import find_active_segment

    segments = [_seg(0, "high_step", 100, 500, 0.8), _seg(1, "dyno", 500, 900, 0.7)]

    assert find_active_segment(segments, 99) is None
    assert find_active_segment(segments, 100).technique == "high_step"
    assert find_active_segment(segments, 499).technique == "high_step"
    assert find_active_segment(segments, 500).technique == "dyno"
    assert find_active_segment(segments, 900) is None


def test_format_overlay_label_includes_sequence_confidence_and_dynamic_marker() -> None:
    from scripts.render_technique_overlay import format_overlay_label

    assert format_overlay_label(_seg(2, "coordination", 1000, 1800, 0.678)) == (
        "#2 coordination 0.68 dynamic"
    )


def test_load_segments_from_dump_json(tmp_path: Path) -> None:
    from scripts.render_technique_overlay import load_segments

    dump = tmp_path / "segments.json"
    dump.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "sequence_index": 0,
                        "start_time_ms": 100,
                        "end_time_ms": 500,
                        "technique": "flagging",
                        "is_dynamic": False,
                        "confidence": 0.75,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    segments = load_segments(dump)

    assert len(segments) == 1
    assert segments[0].technique == "flagging"

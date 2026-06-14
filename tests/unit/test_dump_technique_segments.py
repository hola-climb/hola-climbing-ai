"""dump_technique_segments script contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.callback import AnalysisSegmentPayload


def _payload(
    idx: int,
    technique: str,
    *,
    start: int = 0,
    end: int = 900,
    confidence: float = 0.7,
) -> AnalysisSegmentPayload:
    return AnalysisSegmentPayload(
        sequence_index=idx,
        start_time_ms=start,
        end_time_ms=end,
        technique=technique,
        is_dynamic=technique in {"dyno", "coordination"},
        confidence=confidence,
    )


def test_sanity_check_accepts_valid_segments() -> None:
    from scripts.dump_technique_segments import sanity_check_segments

    report = sanity_check_segments(
        [_payload(0, "dyno", start=100, end=800)],
        raw_segment_count=2,
        video_duration_ms=1000,
    )

    assert report.ok is True
    assert report.dropped_by_rules == 1
    assert report.errors == []


def test_sanity_check_rejects_non_continuous_sequence_index() -> None:
    from scripts.dump_technique_segments import sanity_check_segments

    report = sanity_check_segments(
        [_payload(1, "high_step", start=100, end=800)],
        raw_segment_count=1,
        video_duration_ms=1000,
    )

    assert report.ok is False
    assert "sequence_index must be continuous from 0: got [1]" in report.errors


def test_sanity_check_rejects_wrong_dynamic_flag() -> None:
    from scripts.dump_technique_segments import sanity_check_segments

    seg = _payload(0, "coordination", start=100, end=800)
    seg = seg.model_copy(update={"is_dynamic": False})

    report = sanity_check_segments([seg], raw_segment_count=1, video_duration_ms=1000)

    assert report.ok is False
    assert "segment 0 technique=coordination must have is_dynamic=True" in report.errors


def test_find_videos_accepts_directory_and_file_list(tmp_path: Path) -> None:
    from scripts.dump_technique_segments import find_videos

    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    a = video_dir / "IMG_0001.MOV"
    b = video_dir / "IMG_0002.mp4"
    c = video_dir / "notes.txt"
    for p in (a, b, c):
        p.write_bytes(b"x")
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{b}\n# ignored\n", encoding="utf-8")

    assert find_videos([video_dir]) == [a, b]
    assert find_videos([], file_list=file_list) == [b]


def test_select_balanced_videos_uses_label_csv(tmp_path: Path) -> None:
    from scripts.dump_technique_segments import select_balanced_videos

    videos = []
    for name in ("IMG_0001.MOV", "IMG_0002.MOV", "IMG_0003.MOV", "IMG_0004.MOV"):
        path = tmp_path / name
        path.write_bytes(b"video")
        videos.append(path)
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "filename,label\n"
        "IMG_0001.json,1\n"
        "IMG_0002.json,dynamic\n"
        "IMG_0003.json,0\n"
        "IMG_0004.json,static\n",
        encoding="utf-8",
    )

    selected = select_balanced_videos(videos, labels_csv=labels, sample_per_label=1)

    assert [p.name for p in selected] == ["IMG_0001.MOV", "IMG_0003.MOV"]


def test_run_video_writes_json_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dump_technique_segments as script

    video = tmp_path / "IMG_0001.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(script, "iter_frames", lambda video_path, target_fps: iter(["frame"]))
    monkeypatch.setattr(script, "extract_pose_landmarks", lambda frames, **kwargs: ["pose"])
    monkeypatch.setattr(script, "split_segments", lambda pose_frames: [(100, 800), (900, 1600)])
    monkeypatch.setattr(
        script,
        "classify_segments",
        lambda pose_frames, segments: [_payload(0, "dyno", start=100, end=800)],
    )
    monkeypatch.setattr(script, "get_video_duration_ms", lambda video_path: 2000)

    result = script.run_video(
        video,
        output_dir=tmp_path / "out",
        target_fps=15,
        model_complexity=1,
        min_detection_confidence=0.5,
        task_model_path=None,
        model_version="rule_v1",
        flow_gate_model=None,
        flow_gate_static_threshold=0.3,
        flow_gate_dynamic_threshold=0.7,
        flow_gate_demote_confidence=0.55,
        flow_gate_version_suffix="flow_rf_v2",
    )

    assert result.sanity.ok is True
    assert result.summary.technique_counts == {"dyno": 1}
    assert result.summary.dropped_by_rules == 1
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["video"]["filename"] == "IMG_0001.mp4"
    assert payload["model_version"] == "rule_v1"
    assert payload["summary"]["segments"] == 1
    assert payload["segments"][0]["technique"] == "dyno"


def test_run_video_applies_flow_gate_and_records_demote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dump_technique_segments as script

    video = tmp_path / "IMG_0002.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(script, "iter_frames", lambda video_path, target_fps: iter(["frame"]))
    monkeypatch.setattr(script, "extract_pose_landmarks", lambda frames, **kwargs: ["pose"])
    monkeypatch.setattr(script, "split_segments", lambda pose_frames: [(100, 800)])
    monkeypatch.setattr(
        script,
        "classify_segments",
        lambda pose_frames, segments: [_payload(0, "coordination", start=100, end=800)],
    )
    monkeypatch.setattr(script, "get_video_duration_ms", lambda video_path: 1000)
    monkeypatch.setattr(
        script,
        "apply_flow_gate",
        lambda video_path, segments, **kwargs: ([], 0.05),
    )

    result = script.run_video(
        video,
        output_dir=tmp_path / "out",
        target_fps=15,
        model_complexity=1,
        min_detection_confidence=0.5,
        task_model_path=None,
        model_version="rule_v1",
        flow_gate_model=tmp_path / "gate.joblib",
        flow_gate_static_threshold=0.3,
        flow_gate_dynamic_threshold=0.7,
        flow_gate_demote_confidence=0.55,
        flow_gate_version_suffix="flow_rf_v2",
    )

    assert result.summary.segments == 0
    assert result.summary.dropped_by_gate == 1
    assert result.summary.flow_prob_dynamic == 0.05
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "rule_v1+flow_rf_v2"
    assert payload["flow_gate"]["prob_dynamic"] == 0.05

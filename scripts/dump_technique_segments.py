"""Dump rule-based technique segments for local MVP smoke checks.

This script runs the same vision pipeline as the worker:

    iter_frames -> extract_pose_landmarks -> split_segments -> classify_segments

Optionally, it applies the flow dynamic/static gate and writes one JSON file per
video so Spring DB/API results can be compared against deterministic worker
output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from app.core.config import get_settings
from app.models.callback import AnalysisSegmentPayload
from app.services.pipeline.frames import iter_frames
from app.services.vision._thresholds import DYNAMIC_TECHNIQUES
from app.services.vision.classifier import TECHNIQUE_LABELS, classify_segments
from app.services.vision.flow_gate import apply_flow_gate
from app.services.vision.pose import extract_pose_landmarks
from app.services.vision.segmenter import split_segments

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})
LABEL_ALIASES = {
    "1": "dynamic",
    "1.0": "dynamic",
    "dynamic": "dynamic",
    "d": "dynamic",
    "0": "static",
    "0.0": "static",
    "static": "static",
    "s": "static",
}


@dataclass(frozen=True)
class SanityReport:
    ok: bool
    errors: list[str]
    dropped_by_rules: int


@dataclass(frozen=True)
class VideoSummary:
    filename: str
    raw_segments: int
    segments: int
    dropped_by_rules: int
    dropped_by_gate: int
    technique_counts: dict[str, int]
    dynamic_segments: int
    static_segments: int
    flow_prob_dynamic: float | None


@dataclass(frozen=True)
class VideoDumpResult:
    video_path: Path
    output_path: Path
    summary: VideoSummary
    sanity: SanityReport


def _is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def find_videos(inputs: Sequence[Path], *, file_list: Path | None = None) -> list[Path]:
    """Expand video files from positional inputs and an optional newline file list."""
    videos: list[Path] = []

    for input_path in inputs:
        if input_path.is_dir():
            videos.extend(p for p in input_path.iterdir() if _is_video(p))
        elif _is_video(input_path):
            videos.append(input_path)

    if file_list is not None:
        for raw_line in file_list.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if _is_video(path):
                videos.append(path)

    return sorted(dict.fromkeys(videos), key=lambda p: str(p).lower())


def _normalize_binary_label(value: str) -> str | None:
    return LABEL_ALIASES.get(value.strip().lower())


def _load_labels(labels_csv: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with labels_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not filename:
                continue
            label = _normalize_binary_label(raw_label)
            if label is None:
                continue
            labels[Path(filename).stem] = label
    return labels


def select_balanced_videos(
    videos: Sequence[Path],
    *,
    labels_csv: Path,
    sample_per_label: int,
) -> list[Path]:
    """Select N dynamic and N static videos by matching video stem to labels CSV."""
    if sample_per_label < 1:
        return list(videos)
    labels = _load_labels(labels_csv)
    buckets: dict[str, list[Path]] = {"dynamic": [], "static": []}
    for video in videos:
        label = labels.get(video.stem)
        if label in buckets:
            buckets[label].append(video)
    return buckets["dynamic"][:sample_per_label] + buckets["static"][:sample_per_label]


def get_video_duration_ms(video_path: Path) -> int | None:
    """Return OpenCV duration in milliseconds when metadata is available."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or fps > 240 or frames <= 0:
            return None
        return round((frames / fps) * 1000.0)
    finally:
        cap.release()


def sanity_check_segments(
    segments: Sequence[AnalysisSegmentPayload],
    *,
    raw_segment_count: int,
    video_duration_ms: int | None,
) -> SanityReport:
    """Validate callback-segment invariants before treating a smoke run as usable."""
    errors: list[str] = []
    indices = [s.sequence_index for s in segments]
    expected = list(range(len(segments)))
    if indices != expected:
        errors.append(f"sequence_index must be continuous from 0: got {indices}")

    for seg in segments:
        label = f"segment {seg.sequence_index}"
        if seg.technique not in TECHNIQUE_LABELS:
            errors.append(f"{label} has unknown technique={seg.technique}")

        if seg.start_time_ms is None or seg.end_time_ms is None:
            errors.append(f"{label} must include start_time_ms/end_time_ms")
        elif seg.start_time_ms >= seg.end_time_ms:
            errors.append(
                f"{label} must satisfy start_time_ms < end_time_ms: "
                f"{seg.start_time_ms} >= {seg.end_time_ms}"
            )
        elif video_duration_ms is not None and seg.end_time_ms > video_duration_ms:
            errors.append(
                f"{label} end_time_ms={seg.end_time_ms} exceeds video duration {video_duration_ms}"
            )

        if seg.confidence is None:
            errors.append(f"{label} must include confidence")
        elif not 0.0 <= seg.confidence <= 1.0:
            errors.append(f"{label} confidence must be in [0,1]: {seg.confidence}")

        expected_dynamic = seg.technique in DYNAMIC_TECHNIQUES
        if seg.is_dynamic is not expected_dynamic:
            errors.append(
                f"{label} technique={seg.technique} must have is_dynamic={expected_dynamic}"
            )

    return SanityReport(
        ok=not errors,
        errors=errors,
        dropped_by_rules=max(raw_segment_count - len(segments), 0),
    )


def _segments_total_ms(segments: Sequence[AnalysisSegmentPayload]) -> int:
    total = 0
    for seg in segments:
        if seg.start_time_ms is None or seg.end_time_ms is None:
            continue
        total += max(0, seg.end_time_ms - seg.start_time_ms)
    return total


def _build_summary(
    video_path: Path,
    *,
    raw_segments: int,
    classified_count: int,
    final_segments: Sequence[AnalysisSegmentPayload],
    flow_prob_dynamic: float | None,
) -> VideoSummary:
    counts: Counter[str] = Counter(seg.technique for seg in final_segments)
    dynamic_count = sum(1 for seg in final_segments if seg.is_dynamic)
    return VideoSummary(
        filename=video_path.name,
        raw_segments=raw_segments,
        segments=len(final_segments),
        dropped_by_rules=max(raw_segments - classified_count, 0),
        dropped_by_gate=max(classified_count - len(final_segments), 0),
        technique_counts=dict(sorted(counts.items())),
        dynamic_segments=dynamic_count,
        static_segments=len(final_segments) - dynamic_count,
        flow_prob_dynamic=flow_prob_dynamic,
    )


def _dump_payload(
    *,
    video_path: Path,
    video_duration_ms: int | None,
    model_version: str,
    summary: VideoSummary,
    sanity: SanityReport,
    segments: Sequence[AnalysisSegmentPayload],
    flow_gate_model: Path | None,
    flow_gate_static_threshold: float,
    flow_gate_dynamic_threshold: float,
    flow_gate_demote_confidence: float,
) -> dict[str, Any]:
    dynamic_ms = sum(
        (seg.end_time_ms or 0) - (seg.start_time_ms or 0) for seg in segments if seg.is_dynamic
    )
    total_ms = _segments_total_ms(segments)
    return {
        "video": {
            "path": str(video_path),
            "filename": video_path.name,
            "duration_ms": video_duration_ms,
        },
        "model_version": model_version,
        "summary": {
            **asdict(summary),
            "dynamic_time_ratio": round(dynamic_ms / total_ms, 4) if total_ms else None,
        },
        "sanity": asdict(sanity),
        "flow_gate": {
            "enabled": flow_gate_model is not None,
            "model_path": str(flow_gate_model) if flow_gate_model is not None else None,
            "prob_dynamic": summary.flow_prob_dynamic,
            "static_threshold": flow_gate_static_threshold,
            "dynamic_threshold": flow_gate_dynamic_threshold,
            "demote_confidence": flow_gate_demote_confidence,
        },
        "segments": [seg.model_dump(mode="json") for seg in segments],
    }


def run_video(
    video_path: Path,
    *,
    output_dir: Path,
    target_fps: int,
    model_complexity: int,
    min_detection_confidence: float,
    task_model_path: str | None,
    model_version: str,
    flow_gate_model: Path | None,
    flow_gate_static_threshold: float,
    flow_gate_dynamic_threshold: float,
    flow_gate_demote_confidence: float,
    flow_gate_version_suffix: str,
) -> VideoDumpResult:
    """Run one video and write `<stem>.segments.json` under output_dir."""
    frames = iter_frames(str(video_path), target_fps=target_fps)
    pose_frames = extract_pose_landmarks(
        frames,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        task_model_path=task_model_path,
    )
    raw_segments = split_segments(pose_frames)
    classified_segments = classify_segments(pose_frames, raw_segments)
    final_segments = classified_segments
    flow_prob_dynamic: float | None = None
    effective_model_version = model_version

    if flow_gate_model is not None:
        final_segments, flow_prob_dynamic = apply_flow_gate(
            str(video_path),
            classified_segments,
            model_path=str(flow_gate_model),
            static_threshold=flow_gate_static_threshold,
            dynamic_threshold=flow_gate_dynamic_threshold,
            demote_confidence=flow_gate_demote_confidence,
        )
        effective_model_version = f"{model_version}+{flow_gate_version_suffix}"

    video_duration_ms = get_video_duration_ms(video_path)
    if video_duration_ms is None and pose_frames:
        video_duration_ms = int(pose_frames[-1].timestamp_ms)

    summary = _build_summary(
        video_path,
        raw_segments=len(raw_segments),
        classified_count=len(classified_segments),
        final_segments=final_segments,
        flow_prob_dynamic=flow_prob_dynamic,
    )
    sanity = sanity_check_segments(
        final_segments,
        raw_segment_count=len(raw_segments),
        video_duration_ms=video_duration_ms,
    )
    payload = _dump_payload(
        video_path=video_path,
        video_duration_ms=video_duration_ms,
        model_version=effective_model_version,
        summary=summary,
        sanity=sanity,
        segments=final_segments,
        flow_gate_model=flow_gate_model,
        flow_gate_static_threshold=flow_gate_static_threshold,
        flow_gate_dynamic_threshold=flow_gate_dynamic_threshold,
        flow_gate_demote_confidence=flow_gate_demote_confidence,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.segments.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VideoDumpResult(
        video_path=video_path,
        output_path=output_path,
        summary=summary,
        sanity=sanity,
    )


def _print_result(result: VideoDumpResult) -> None:
    status = "ok" if result.sanity.ok else "failed"
    flow = ""
    if result.summary.flow_prob_dynamic is not None:
        flow = f" flow_prob_dynamic={result.summary.flow_prob_dynamic:.4f}"
    print(
        f"[{status}] {result.video_path.name}: "
        f"segments={result.summary.segments} "
        f"raw={result.summary.raw_segments} "
        f"drop_rules={result.summary.dropped_by_rules} "
        f"drop_gate={result.summary.dropped_by_gate} "
        f"techniques={result.summary.technique_counts}{flow} "
        f"-> {result.output_path}"
    )
    for err in result.sanity.errors:
        print(f"  sanity: {err}", file=sys.stderr)


def _print_batch_summary(results: Sequence[VideoDumpResult], failures: int) -> None:
    counts: Counter[str] = Counter()
    total_segments = 0
    total_rule_drops = 0
    total_gate_drops = 0
    empty_outputs = 0
    for result in results:
        counts.update(result.summary.technique_counts)
        total_segments += result.summary.segments
        total_rule_drops += result.summary.dropped_by_rules
        total_gate_drops += result.summary.dropped_by_gate
        if result.summary.segments == 0:
            empty_outputs += 1
    print(
        "\n[summary] "
        f"videos={len(results)} failures={failures} "
        f"segments={total_segments} empty_outputs={empty_outputs} "
        f"drop_rules={total_rule_drops} drop_gate={total_gate_drops} "
        f"techniques={dict(sorted(counts.items()))}"
    )


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Video files or directories containing .mp4/.mov/.avi/.mkv files.",
    )
    parser.add_argument("--file-list", type=Path, help="Newline-delimited video path list.")
    parser.add_argument("--labels-csv", type=Path, help="CSV with filename,label columns.")
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=0,
        help="When --labels-csv is set, select N dynamic and N static videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/technique_segments"),
        help="Directory for per-video *.segments.json files.",
    )
    parser.add_argument("--target-fps", type=int, default=settings.frame_target_fps)
    parser.add_argument("--model-complexity", type=int, default=settings.mp_model_complexity)
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=settings.mp_min_detection_confidence,
    )
    parser.add_argument("--mp-task-model-path", default=settings.mp_task_model_path)
    parser.add_argument("--model-version", default=settings.model_version)
    parser.add_argument("--flow-gate-model", type=Path)
    parser.add_argument(
        "--flow-gate-static-threshold",
        type=float,
        default=settings.flow_gate_static_threshold,
    )
    parser.add_argument(
        "--flow-gate-dynamic-threshold",
        type=float,
        default=settings.flow_gate_dynamic_threshold,
    )
    parser.add_argument(
        "--flow-gate-demote-confidence",
        type=float,
        default=settings.flow_gate_demote_confidence,
    )
    parser.add_argument(
        "--flow-gate-version-suffix",
        default=settings.flow_gate_version_suffix,
    )
    parser.add_argument("--limit", type=int, default=0, help="Max videos to process. 0 = all.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    videos = find_videos(args.inputs, file_list=args.file_list)
    if args.labels_csv is not None and args.sample_per_label > 0:
        videos = select_balanced_videos(
            videos,
            labels_csv=args.labels_csv,
            sample_per_label=args.sample_per_label,
        )
    if args.limit > 0:
        videos = videos[: args.limit]
    if not videos:
        print("[error] no video inputs found", file=sys.stderr)
        return 2

    results: list[VideoDumpResult] = []
    failures = 0
    for video_path in videos:
        try:
            result = run_video(
                video_path,
                output_dir=args.output_dir,
                target_fps=args.target_fps,
                model_complexity=args.model_complexity,
                min_detection_confidence=args.min_detection_confidence,
                task_model_path=args.mp_task_model_path,
                model_version=args.model_version,
                flow_gate_model=args.flow_gate_model,
                flow_gate_static_threshold=args.flow_gate_static_threshold,
                flow_gate_dynamic_threshold=args.flow_gate_dynamic_threshold,
                flow_gate_demote_confidence=args.flow_gate_demote_confidence,
                flow_gate_version_suffix=args.flow_gate_version_suffix,
            )
        except Exception as exc:
            failures += 1
            print(f"[error] {video_path}: {exc!r}", file=sys.stderr)
            continue
        _print_result(result)
        results.append(result)
        if not result.sanity.ok:
            failures += 1

    _print_batch_summary(results, failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

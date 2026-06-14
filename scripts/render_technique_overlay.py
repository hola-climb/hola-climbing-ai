"""Render an MP4 with technique segment labels burned into each frame."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.models.callback import AnalysisSegmentPayload
from scripts.compare_analysis_segments import load_segments_from_json


def load_segments(path: Path) -> list[AnalysisSegmentPayload]:
    return load_segments_from_json(path).segments


def find_active_segment(
    segments: Sequence[AnalysisSegmentPayload],
    timestamp_ms: int,
) -> AnalysisSegmentPayload | None:
    """Return segment active at timestamp using [start, end) boundaries."""
    for segment in segments:
        if segment.start_time_ms is None or segment.end_time_ms is None:
            continue
        if segment.start_time_ms <= timestamp_ms < segment.end_time_ms:
            return segment
    return None


def format_overlay_label(segment: AnalysisSegmentPayload) -> str:
    confidence = segment.confidence if segment.confidence is not None else 0.0
    motion = "dynamic" if segment.is_dynamic else "static"
    return f"#{segment.sequence_index} {segment.technique} {confidence:.2f} {motion}"


def _draw_overlay(
    frame: NDArray[np.uint8],
    segment: AnalysisSegmentPayload | None,
    *,
    timestamp_ms: int,
    font_scale: float,
    thickness: int,
) -> NDArray[np.uint8]:
    out = frame.copy()
    height, width = out.shape[:2]
    band_h = max(54, int(height * 0.10))
    overlay = out.copy()
    color = (
        (32, 32, 32)
        if segment is None
        else ((34, 84, 230) if segment.is_dynamic else (36, 148, 92))
    )
    cv2.rectangle(overlay, (0, 0), (width, band_h), color, -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    if segment is None:
        label = f"{timestamp_ms / 1000:.2f}s no segment"
    else:
        label = f"{timestamp_ms / 1000:.2f}s  {format_overlay_label(segment)}"
    cv2.putText(
        out,
        label,
        (18, max(34, band_h - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def render_overlay(
    *,
    video_path: Path,
    segments_path: Path,
    output_path: Path,
    font_scale: float = 0.8,
    thickness: int = 2,
) -> int:
    """Render overlay video and return number of frames written."""
    segments = load_segments(segments_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or fps > 240:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"cannot read video dimensions: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"cannot open output video: {output_path}")
        try:
            frame_idx = 0
            written = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = round(frame_idx * 1000.0 / fps)
                segment = find_active_segment(segments, timestamp_ms)
                frame_u8 = cast(NDArray[np.uint8], frame)
                rendered = _draw_overlay(
                    frame_u8,
                    segment,
                    timestamp_ms=timestamp_ms,
                    font_scale=font_scale,
                    thickness=thickness,
                )
                writer.write(rendered)
                frame_idx += 1
                written += 1
            return written
        finally:
            writer.release()
    finally:
        cap.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--segments-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font-scale", type=float, default=0.8)
    parser.add_argument("--thickness", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    if output is None:
        output = args.segments_json.with_name(f"{args.video.stem}.technique_overlay.mp4")
    frames = render_overlay(
        video_path=args.video,
        segments_path=args.segments_json,
        output_path=output,
        font_scale=args.font_scale,
        thickness=args.thickness,
    )
    print(f"[ok] wrote {frames} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

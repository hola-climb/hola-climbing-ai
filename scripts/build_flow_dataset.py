"""Build optical-flow feature datasets from labeled videos.

Usage:
    uv run python scripts/build_flow_dataset.py \
        --labels data/review/labels_완료_qa_v2.csv \
        --videos-dir /Users/minjoun/Movies/Original \
        --out data/flow_dataset/qa_flow_v3
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.vision.flow_features import (
    FLOW_FEATURE_VERSION,
    extract_flow_series,
    extract_flow_stats,
    remove_fall_end,
)
from app.services.vision.pose_dataset import match_labeled_videos


@dataclass(frozen=True)
class BuildFlowDatasetResult:
    written: int
    missing: list[str]
    failed: list[tuple[str, str]]
    manifest_path: Path


def build_flow_dataset(
    *,
    labels_csv: Path,
    videos_dir: Path,
    out_dir: Path,
    resize: tuple[int, int] = (320, 240),
    target_fps: int = 30,
) -> BuildFlowDatasetResult:
    """Build one compressed `.npz` optical-flow feature file per labeled video."""
    matched, missing = match_labeled_videos(labels_csv, videos_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.with_name(f"{out_dir.name}_manifest.csv")

    written = 0
    failed: list[tuple[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    for item in matched:
        try:
            flow_series, src_fps, duration_sec = extract_flow_series(
                item.video_path,
                resize=resize,
                target_fps=target_fps,
            )
            trimmed = remove_fall_end(flow_series)
            features = extract_flow_stats(trimmed, target_fps=target_fps)
            out_path = out_dir / f"{item.stem}.npz"
            np.savez_compressed(
                out_path,
                x=features,
                label=np.asarray(item.label, dtype=np.int64),
                stem=np.asarray(item.stem),
                source_path=np.asarray(str(item.video_path)),
                variant=np.asarray("flow"),
                feature_version=np.asarray(FLOW_FEATURE_VERSION),
                raw_flow_frames=np.asarray(len(flow_series), dtype=np.int64),
                flow_frames=np.asarray(len(trimmed), dtype=np.int64),
                src_fps=np.asarray(src_fps, dtype=np.float32),
                duration_sec=np.asarray(duration_sec, dtype=np.float32),
            )
            written += 1
            manifest_rows.append(
                {
                    "stem": item.stem,
                    "label": item.label,
                    "variant": "flow",
                    "feature_version": FLOW_FEATURE_VERSION,
                    "feature_dim": features.shape[0],
                    "source_path": str(item.video_path),
                    "out_path": str(out_path),
                    "raw_flow_frames": len(flow_series),
                    "flow_frames": len(trimmed),
                    "src_fps": round(src_fps, 4),
                    "duration_sec": round(duration_sec, 4),
                }
            )
        except Exception as exc:
            failed.append((item.stem, repr(exc)))

    _write_manifest(manifest_path, manifest_rows)
    return BuildFlowDatasetResult(
        written=written,
        missing=missing,
        failed=failed,
        manifest_path=manifest_path,
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "stem",
        "label",
        "variant",
        "feature_version",
        "feature_dim",
        "source_path",
        "out_path",
        "raw_flow_frames",
        "flow_frames",
        "src_fps",
        "duration_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=240)
    parser.add_argument("--target-fps", type=int, default=30)
    args = parser.parse_args()

    result = build_flow_dataset(
        labels_csv=args.labels,
        videos_dir=args.videos_dir,
        out_dir=args.out,
        resize=(args.resize_width, args.resize_height),
        target_fps=args.target_fps,
    )
    print(
        "[summary] "
        f"written={result.written} missing={len(result.missing)} "
        f"failed={len(result.failed)} manifest={result.manifest_path}"
    )
    if result.missing:
        print("[missing sample]", ", ".join(result.missing[:10]))
    if result.failed:
        print("[failed sample]", result.failed[:5])
    return 0 if result.written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build cached pose sequence dataset for dynamic/static model training.

Usage:
    uv run python scripts/build_pose_dataset.py \
        --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
        --videos /Users/minjoun/Movies/Original \
        --out data/pose_dataset \
        --target-frames 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app.services.pipeline.frames import iter_frames
from app.services.vision.pose import extract_pose_landmarks
from app.services.vision.pose_dataset import build_model_input, match_labeled_videos


def _build_one(
    video_path: Path,
    *,
    target_fps: int,
    target_frames: int,
    task_model_path: str | None,
) -> tuple[np.ndarray, int]:
    frames = iter_frames(str(video_path), target_fps=target_fps)
    pose_frames = extract_pose_landmarks(frames, task_model_path=task_model_path)
    return build_model_input(pose_frames, target_frames=target_frames), len(pose_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/pose_dataset"))
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--target-frames", type=int, default=128)
    parser.add_argument(
        "--task-model",
        type=Path,
        default=Path("models/mediapipe/pose_landmarker_lite.task"),
        help="MediaPipe tasks pose landmarker model path",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"[error] labels not found: {args.labels}", file=sys.stderr)
        return 2
    if not args.videos.exists():
        print(f"[error] videos dir not found: {args.videos}", file=sys.stderr)
        return 2

    matched, missing = match_labeled_videos(args.labels, args.videos)
    work = matched[: args.limit] if args.limit > 0 else matched
    args.out.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped_existing = 0
    failed: list[tuple[str, str]] = []
    for item in work:
        out_path = args.out / f"{item.stem}.npz"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        try:
            x, raw_pose_frames = _build_one(
                item.video_path,
                target_fps=args.target_fps,
                target_frames=args.target_frames,
                task_model_path=str(args.task_model) if args.task_model else None,
            )
            np.savez_compressed(
                out_path,
                x=x,
                label=np.asarray(item.label, dtype=np.int64),
                stem=np.asarray(item.stem),
                source_path=np.asarray(str(item.video_path)),
                raw_pose_frames=np.asarray(raw_pose_frames, dtype=np.int64),
            )
            ok += 1
            print(f"[ok] {item.stem} label={item.label} frames={raw_pose_frames} -> {out_path}")
        except Exception as exc:
            failed.append((item.stem, repr(exc)))
            print(f"[fail] {item.stem}: {exc!r}", file=sys.stderr)

    print(
        "\n[summary] "
        f"labeled={len(matched)} missing_video={len(missing)} selected={len(work)} "
        f"written={ok} skipped_existing={skipped_existing} failed={len(failed)}"
    )
    if missing:
        print("[missing sample]", ", ".join(missing[:10]))
    if failed:
        print("[failed sample]", failed[:5])
    return 0 if ok > 0 or skipped_existing > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

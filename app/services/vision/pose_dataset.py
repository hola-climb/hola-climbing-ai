"""Pose sequence dataset helpers for learned dynamic/static classification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from app.services.vision.pose import PoseFrame

VIDEO_EXTS: Final[tuple[str, ...]] = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV")
LABEL_ALIASES: Final[dict[str, int]] = {
    "0": 0,
    "0.0": 0,
    "static": 0,
    "s": 0,
    "1": 1,
    "1.0": 1,
    "dynamic": 1,
    "d": 1,
}


@dataclass(frozen=True)
class LabeledVideo:
    """A label row matched to a local video file."""

    stem: str
    label: int
    video_path: Path


def normalize_label(raw: str) -> int | None:
    """Normalize a CSV label value to `0=static` or `1=dynamic`."""
    return LABEL_ALIASES.get(raw.strip().lower())


def load_label_rows(csv_path: Path) -> list[tuple[str, int]]:
    """Load labeled rows as `(filename_stem, label)` and skip empty labels."""
    rows: list[tuple[str, int]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = normalize_label(row.get("label") or "")
            if not filename or label is None:
                continue
            rows.append((Path(filename).stem, label))
    return rows


def find_video(stem: str, videos_dir: Path) -> Path | None:
    """Find a video by stem and known video extensions."""
    allowed_suffixes = {ext.lower() for ext in VIDEO_EXTS}
    for child in videos_dir.iterdir():
        if child.is_file() and child.stem == stem and child.suffix.lower() in allowed_suffixes:
            return child
    for ext in VIDEO_EXTS:
        candidate = videos_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = [p for p in videos_dir.glob(f"{stem}.*") if p.suffix.lower() in allowed_suffixes]
    return matches[0] if matches else None


def match_labeled_videos(labels_csv: Path, videos_dir: Path) -> tuple[list[LabeledVideo], list[str]]:
    """Match labeled CSV rows to local video files."""
    matched: list[LabeledVideo] = []
    missing: list[str] = []
    for stem, label in load_label_rows(labels_csv):
        video_path = find_video(stem, videos_dir)
        if video_path is None:
            missing.append(stem)
            continue
        matched.append(LabeledVideo(stem=stem, label=label, video_path=video_path))
    return matched, missing


def pose_frames_to_array(pose_frames: list[PoseFrame]) -> NDArray[np.float32]:
    """Convert PoseFrame list to `(T, 33, 4)` float32 array."""
    if not pose_frames:
        raise ValueError("pose_frames must not be empty")
    return cast(NDArray[np.float32], np.stack([pf.landmarks for pf in pose_frames], axis=0))


def resample_pose_array(
    pose_array: NDArray[np.float32],
    target_frames: int,
) -> NDArray[np.float32]:
    """Linearly resample `(T, 33, 4)` pose sequence to `target_frames`."""
    if target_frames < 1:
        raise ValueError("target_frames must be >= 1")
    if pose_array.ndim != 3 or pose_array.shape[1:] != (33, 4):
        raise ValueError(f"pose_array must have shape (T, 33, 4), got {pose_array.shape}")
    source_frames = pose_array.shape[0]
    if source_frames == 0:
        raise ValueError("pose_array must contain at least one frame")
    if source_frames == 1:
        return cast(NDArray[np.float32], np.repeat(pose_array, target_frames, axis=0))

    flat = pose_array.reshape(source_frames, -1)
    source_x = np.linspace(0.0, 1.0, num=source_frames, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)
    out = np.empty((target_frames, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(target_x, source_x, flat[:, col]).astype(np.float32)
    return out.reshape(target_frames, 33, 4)


def build_model_input(
    pose_frames: list[PoseFrame],
    target_frames: int,
) -> NDArray[np.float32]:
    """Build flattened GRU input shaped `(target_frames, 132)`."""
    pose_array = pose_frames_to_array(pose_frames)
    resampled = resample_pose_array(pose_array, target_frames=target_frames)
    return resampled.reshape(target_frames, 33 * 4).astype(np.float32)

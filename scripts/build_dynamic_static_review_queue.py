"""Build a human QA review queue for dynamic/static model predictions."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.vision.pose_dataset import find_video, load_label_rows

REVIEW_COLUMNS = [
    "priority",
    "review_group",
    "stem",
    "current_label",
    "raw_prob",
    "raw_pred",
    "raw_correct",
    "motion_prob",
    "motion_pred",
    "motion_correct",
    "wrong_confidence",
    "raw_pose_frames",
    "suggested_status",
    "new_label",
    "reason",
    "notes",
    "video_path",
]

_GROUP_ORDER = {
    "pose_failure": 0,
    "low_pose_frames": 1,
    "both_models_missed": 2,
    "raw_high_confidence_miss": 3,
    "raw_correct_sample": 4,
}


@dataclass(frozen=True)
class Prediction:
    stem: str
    label: int
    prob_dynamic: float
    pred: int
    correct: bool
    raw_pose_frames: int

    @property
    def wrong_confidence(self) -> float:
        if self.correct:
            return 0.0
        return self.prob_dynamic if self.label == 0 else 1.0 - self.prob_dynamic


def load_predictions(path: Path) -> dict[str, Prediction]:
    """Load validation predictions keyed by stem."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "valid"]
    return {
        row["stem"]: Prediction(
            stem=row["stem"],
            label=int(row["label"]),
            prob_dynamic=float(row["prob_dynamic"]),
            pred=int(row["pred"]),
            correct=row["correct"] == "True",
            raw_pose_frames=int(row["raw_pose_frames"]),
        )
        for row in rows
    }


def load_labels(path: Path | None) -> dict[str, int]:
    """Load optional labels CSV as `stem -> label`."""
    if path is None:
        return {}
    return dict(load_label_rows(path))


def build_review_rows(
    *,
    raw_predictions: Path,
    motion_predictions: Path,
    data_dir: Path,
    videos_dir: Path,
    high_confidence_limit: int = 20,
    correct_sample_count: int = 20,
    known_failures: list[str] | None = None,
    label_map: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Build prioritized review rows for human QA."""
    raw = load_predictions(raw_predictions)
    motion = load_predictions(motion_predictions)
    labels = label_map or {}
    selected: dict[str, dict[str, str]] = {}

    for item in known_failures or []:
        stem, reason = _parse_known_failure(item)
        selected[stem] = _make_row(
            priority="P0",
            review_group="pose_failure",
            stem=stem,
            current_label=labels.get(stem),
            reason=reason,
            suggested_status="exclude",
            video_path=_video_path(stem, videos_dir),
        )

    for stem, label, frame_count in _low_pose_frame_samples(data_dir, max_frames=29):
        selected.setdefault(
            stem,
            _make_row(
                priority="P0",
                review_group="low_pose_frames",
                stem=stem,
                current_label=label,
                raw_pose_frames=str(frame_count),
                reason="video_too_short",
                suggested_status="exclude",
                video_path=_video_path(stem, videos_dir),
            ),
        )

    for stem in sorted(set(raw) & set(motion)):
        raw_pred = raw[stem]
        motion_pred = motion[stem]
        if not raw_pred.correct and not motion_pred.correct:
            selected.setdefault(
                stem,
                _prediction_row(
                    priority="P1",
                    review_group="both_models_missed",
                    raw=raw_pred,
                    motion=motion_pred,
                    videos_dir=videos_dir,
                    reason="",
                    suggested_status="review",
                ),
            )

    raw_misses = sorted(
        [pred for pred in raw.values() if not pred.correct],
        key=lambda pred: (-pred.wrong_confidence, pred.stem),
    )
    added = 0
    for pred in raw_misses:
        if added >= high_confidence_limit:
            break
        if pred.stem in selected:
            continue
        selected[pred.stem] = _prediction_row(
            priority="P2",
            review_group="raw_high_confidence_miss",
            raw=pred,
            motion=motion.get(pred.stem),
            videos_dir=videos_dir,
            reason="",
            suggested_status="review",
        )
        added += 1

    correct_candidates = [pred for pred in raw.values() if pred.correct and pred.stem not in selected]
    rng = random.Random(seed)
    rng.shuffle(correct_candidates)
    for pred in sorted(correct_candidates[:correct_sample_count], key=lambda item: item.stem):
        selected[pred.stem] = _prediction_row(
            priority="P4",
            review_group="raw_correct_sample",
            raw=pred,
            motion=motion.get(pred.stem),
            videos_dir=videos_dir,
            reason="keep",
            suggested_status="keep",
        )

    return sorted(
        selected.values(),
        key=lambda row: (row["priority"], _GROUP_ORDER.get(row["review_group"], 99), row["stem"]),
    )


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write review rows to a stable-column CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in REVIEW_COLUMNS})


def build_contact_sheets(
    review_csv: Path,
    *,
    out_dir: Path,
    max_items: int = 40,
    frames_per_video: int = 6,
) -> int:
    """Create simple contact sheet JPEGs for the first review rows with video paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with review_csv.open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("video_path")]

    written = 0
    for row in rows[:max_items]:
        video_path = Path(row["video_path"])
        if not video_path.exists():
            continue
        sheet = _make_contact_sheet(video_path, frames_per_video=frames_per_video)
        if sheet is None:
            continue
        label = f"{row['priority']} {row['review_group']} {row['stem']} label={row['current_label']}"
        cv2.putText(sheet, label[:110], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = out_dir / f"{row['priority']}_{row['review_group']}_{row['stem']}.jpg"
        cv2.imwrite(str(out_path), sheet)
        written += 1
    return written


def _make_contact_sheet(video_path: Path, *, frames_per_video: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return None
        indices = np.linspace(0, max(0, total - 1), num=frames_per_video, dtype=np.int64)
        thumbs: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            thumb = cv2.resize(frame, (240, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, f"f={int(idx)}", (8, 306), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            thumbs.append(thumb)
        if not thumbs:
            return None
        while len(thumbs) < frames_per_video:
            thumbs.append(np.zeros_like(thumbs[0]))
        top = np.hstack(thumbs[:3])
        bottom = np.hstack(thumbs[3:6])
        return np.vstack([top, bottom])
    finally:
        cap.release()


def _prediction_row(
    *,
    priority: str,
    review_group: str,
    raw: Prediction,
    motion: Prediction | None,
    videos_dir: Path,
    reason: str,
    suggested_status: str,
) -> dict[str, str]:
    wrong_confidence = max(raw.wrong_confidence, motion.wrong_confidence if motion else 0.0)
    return _make_row(
        priority=priority,
        review_group=review_group,
        stem=raw.stem,
        current_label=raw.label,
        raw_prob=f"{raw.prob_dynamic:.6f}",
        raw_pred=str(raw.pred),
        raw_correct=str(raw.correct),
        motion_prob=f"{motion.prob_dynamic:.6f}" if motion else "",
        motion_pred=str(motion.pred) if motion else "",
        motion_correct=str(motion.correct) if motion else "",
        wrong_confidence=f"{wrong_confidence:.6f}",
        raw_pose_frames=str(raw.raw_pose_frames),
        suggested_status=suggested_status,
        reason=reason,
        video_path=_video_path(raw.stem, videos_dir),
    )


def _make_row(
    *,
    priority: str,
    review_group: str,
    stem: str,
    current_label: int | None,
    raw_prob: str = "",
    raw_pred: str = "",
    raw_correct: str = "",
    motion_prob: str = "",
    motion_pred: str = "",
    motion_correct: str = "",
    wrong_confidence: str = "",
    raw_pose_frames: str = "",
    suggested_status: str,
    reason: str,
    video_path: str,
) -> dict[str, str]:
    return {
        "priority": priority,
        "review_group": review_group,
        "stem": stem,
        "current_label": "" if current_label is None else str(current_label),
        "raw_prob": raw_prob,
        "raw_pred": raw_pred,
        "raw_correct": raw_correct,
        "motion_prob": motion_prob,
        "motion_pred": motion_pred,
        "motion_correct": motion_correct,
        "wrong_confidence": wrong_confidence,
        "raw_pose_frames": raw_pose_frames,
        "suggested_status": suggested_status,
        "new_label": "",
        "reason": reason,
        "notes": "",
        "video_path": video_path,
    }


def _low_pose_frame_samples(data_dir: Path, *, max_frames: int) -> list[tuple[str, int, int]]:
    samples: list[tuple[str, int, int]] = []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            stem = str(data["stem"]) if "stem" in data.files else path.stem
            label = int(data["label"])
            raw_pose_frames = int(data["raw_pose_frames"]) if "raw_pose_frames" in data.files else int(data["x"].shape[0])
        if raw_pose_frames <= max_frames:
            samples.append((stem, label, raw_pose_frames))
    return samples


def _parse_known_failure(item: str) -> tuple[str, str]:
    if ":" not in item:
        return item, "pose_quality_bad"
    stem, reason = item.split(":", maxsplit=1)
    return stem, reason


def _video_path(stem: str, videos_dir: Path) -> str:
    path = find_video(stem, videos_dir)
    return str(path) if path else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--motion-predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pose_dataset"))
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/review/dynamic_static_review_queue.csv"))
    parser.add_argument("--high-confidence-limit", type=int, default=20)
    parser.add_argument("--correct-sample-count", type=int, default=20)
    parser.add_argument("--known-failure", action="append", default=[])
    parser.add_argument("--contact-sheets-dir", type=Path)
    parser.add_argument("--contact-sheet-max", type=int, default=40)
    args = parser.parse_args()

    rows = build_review_rows(
        raw_predictions=args.raw_predictions,
        motion_predictions=args.motion_predictions,
        data_dir=args.data_dir,
        videos_dir=args.videos_dir,
        high_confidence_limit=args.high_confidence_limit,
        correct_sample_count=args.correct_sample_count,
        known_failures=args.known_failure,
        label_map=load_labels(args.labels),
    )
    write_review_csv(args.out, rows)
    print(f"[done] review_rows={len(rows)} out={args.out}")
    if args.contact_sheets_dir is not None:
        written = build_contact_sheets(
            args.out,
            out_dir=args.contact_sheets_dir,
            max_items=args.contact_sheet_max,
        )
        print(f"[done] contact_sheets={written} out_dir={args.contact_sheets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

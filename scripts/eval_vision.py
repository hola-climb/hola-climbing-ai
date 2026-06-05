"""Vision classifier 정확도 평가 — 영상 단위 dynamic/static 분류.

# 평가 모델

사용자 정정 (2026-05-28):
- `labels.csv` 의 `label` 컬럼은 영상 단위 ground truth: `dynamic` 또는 `static`
  (해당 영상이 다이나믹 위주의 등반인지 스태틱 위주의 등반인지)
- 6 기술 segment-level GT는 존재하지 않음 (임계값 튜닝은 사용자가 영상을 직접 보면서 수행)
- 워커 측 평가:
    1. 영상 → vision pipeline (pose → segment → classify) → segments[]
    2. segments 중 `is_dynamic=True` segment의 **누적 시간 비율**을 계산
    3. 임계값 (기본 0.30) ≥ 이면 `dynamic`, 미만이면 `static`
    4. ground truth `labels.csv` 와 비교 → accuracy / confusion matrix

# 사용

    uv run python scripts/eval_vision.py \\
        --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels.csv \\
        --videos /path/to/videos \\
        --dynamic-threshold 0.30

라벨 CSV 포맷 (둘 다 지원):
    filename,label              filename,label
    IMG_0028.json,dynamic       IMG_0028.json,1
    IMG_0031.json,static        IMG_0031.json,0

    1 = dynamic, 0 = static (labels_완료.csv 컨벤션)

영상 매칭:
- labels.csv의 `filename` stem (확장자 제외) 으로 `--videos` 디렉토리에서
  같은 stem의 `.mp4`/`.mov`/`.avi`/`.mkv` 를 검색.
- 못 찾으면 해당 행은 skip (스킵 카운트 출력).

라벨이 미채움이면 자동으로 dry-run 모드 (예측만 출력, 채점 없음).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# stdlib 외 의존: app/* 만 사용. sklearn은 선택 (없으면 fallback).
try:
    from sklearn.metrics import classification_report, confusion_matrix

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV")
VALID_LABELS = {"dynamic", "static"}

# 라벨 정규화: "1" / "dynamic" → "dynamic", "0" / "static" → "static"
# 사용자 정정 (2026-05-28): labels_완료.csv는 1=dynamic, 0=static binary
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


def _normalize_label(raw: str) -> str:
    """raw label → 'dynamic' | 'static' | '' (미채움/모름)."""
    key = raw.strip().lower()
    return LABEL_ALIASES.get(key, "")


def load_labels(csv_path: Path) -> list[tuple[str, str]]:
    """labels.csv → [(stem, label), ...]. label은 정규화된 'dynamic'/'static'/''."""
    rows: list[tuple[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            filename = (r.get("filename") or "").strip()
            raw_label = (r.get("label") or "").strip()
            if not filename:
                continue
            stem = Path(filename).stem
            rows.append((stem, _normalize_label(raw_label)))
    return rows


def find_video(stem: str, videos_dir: Path) -> Path | None:
    """stem과 일치하는 영상 파일 검색 (확장자만 다름)."""
    for ext in VIDEO_EXTS:
        candidate = videos_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # fallback: glob (filename에 공백/특수문자 있는 경우)
    matches = list(videos_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def predict_video(video_path: Path, target_fps: int, dynamic_threshold: float) -> tuple[str, dict]:
    """영상 단위 dynamic/static 예측.

    Returns:
        (label, debug_info) — label은 'dynamic' | 'static' | 'unknown'
    """
    # lazy import: app 의존성은 호출 시점에만
    from app.services.pipeline.frames import iter_frames
    from app.services.vision.classifier import classify_segments
    from app.services.vision.pose import extract_pose_landmarks
    from app.services.vision.segmenter import split_segments

    try:
        frames = iter_frames(str(video_path), target_fps=target_fps)
        poses = extract_pose_landmarks(frames)
        segs = split_segments(poses)
        payloads = classify_segments(poses, segs)
    except Exception as exc:
        return "unknown", {"error": repr(exc)}

    if not payloads:
        return "unknown", {"segments": 0}

    total_ms = sum(p.end_time_ms - p.start_time_ms for p in payloads)
    dynamic_ms = sum(
        p.end_time_ms - p.start_time_ms for p in payloads if p.is_dynamic
    )
    ratio = dynamic_ms / total_ms if total_ms > 0 else 0.0
    label = "dynamic" if ratio >= dynamic_threshold else "static"
    counts = Counter(p.technique for p in payloads)
    return label, {
        "segments": len(payloads),
        "dynamic_ratio": round(ratio, 3),
        "technique_counts": dict(counts),
    }


def score(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    acc = correct / n if n else 0.0
    out: dict = {"n": n, "accuracy": round(acc, 4)}
    if _HAS_SKLEARN and n > 0:
        out["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=["dynamic", "static"]
        ).tolist()
        out["report"] = classification_report(
            y_true, y_pred, labels=["dynamic", "static"], zero_division=0
        )
    else:
        # fallback: 수동 confusion matrix
        cm = {("dynamic", "dynamic"): 0, ("dynamic", "static"): 0,
              ("static", "dynamic"): 0, ("static", "static"): 0}
        for t, p in zip(y_true, y_pred, strict=True):
            if (t, p) in cm:
                cm[(t, p)] += 1
        out["confusion_matrix"] = cm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, required=True, help="labels.csv 경로")
    ap.add_argument("--videos", type=Path, help="영상 디렉토리 (지정 없으면 dry-run)")
    ap.add_argument("--target-fps", type=int, default=15)
    ap.add_argument("--dynamic-threshold", type=float, default=0.30,
                    help="dynamic segment 누적 시간 비율 임계값 (기본 0.30)")
    ap.add_argument("--limit", type=int, default=0, help="최대 영상 수 (0=전체)")
    args = ap.parse_args()

    if not args.labels.exists():
        print(f"[error] labels file not found: {args.labels}", file=sys.stderr)
        return 2

    rows = load_labels(args.labels)
    labeled = [r for r in rows if r[1] in VALID_LABELS]
    unlabeled = [r for r in rows if r[1] not in VALID_LABELS]

    print(f"[labels] total={len(rows)} labeled={len(labeled)} unlabeled={len(unlabeled)}")
    if not labeled and not args.videos:
        print("\n라벨이 채워진 행이 없습니다. labels.csv 'label' 컬럼에 "
              "'dynamic' 또는 'static' 을 채워주세요.")
        print("--videos 가 지정되면 예측만 출력하는 dry-run 가능합니다.")
        return 1

    if not args.videos:
        print("\n[dry-run] --videos 미지정. 라벨 분포만 출력:")
        print(f"  dynamic: {sum(1 for _, label in labeled if label == 'dynamic')}")
        print(f"  static : {sum(1 for _, label in labeled if label == 'static')}")
        return 0

    if not args.videos.exists():
        print(f"[error] videos dir not found: {args.videos}", file=sys.stderr)
        return 2

    work = labeled if labeled else rows
    if args.limit > 0:
        work = work[: args.limit]

    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0
    unknown = 0
    debug_lines: list[str] = []

    for stem, label in work:
        video_path = find_video(stem, args.videos)
        if not video_path:
            skipped += 1
            continue
        pred, debug = predict_video(video_path, args.target_fps, args.dynamic_threshold)
        debug_lines.append(f"  {stem}: pred={pred} gt={label or '-'} {debug}")
        if pred == "unknown":
            unknown += 1
            continue
        if label in VALID_LABELS:
            y_true.append(label)
            y_pred.append(pred)

    print(f"\n[run] processed={len(y_pred)} skipped(missing video)={skipped} "
          f"unknown(pose fail)={unknown}")
    for line in debug_lines[:20]:
        print(line)
    if len(debug_lines) > 20:
        print(f"  ... (+{len(debug_lines) - 20} more)")

    if not y_true:
        print("\n채점할 라벨이 없습니다. labels.csv 에 dynamic/static 을 채워주세요.")
        return 0

    result = score(y_true, y_pred)
    print(f"\n[score] n={result['n']} accuracy={result['accuracy']}")
    print("[confusion_matrix]", result["confusion_matrix"])
    if "report" in result:
        print("[classification_report]")
        print(result["report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

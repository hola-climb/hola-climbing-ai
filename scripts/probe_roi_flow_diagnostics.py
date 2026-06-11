"""Probe whether pose ROI flow separates hard dynamic misses from easy dynamics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.services.vision.pose import PoseFrame, extract_pose_landmarks

_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "roi_mag_mean",
    "roi_mag_p95",
    "roi_mag_max",
    "roi_vy_mean",
    "roi_vy_std",
    "roi_vy_min",
    "roi_vy_max",
    "roi_vy_p10",
    "roi_vy_p90",
    "roi_upward_ratio",
    "roi_downward_ratio",
    "roi_max_upward_window_mean",
    "roi_max_downward_window_mean",
    "adj_mag_mean",
    "adj_mag_p95",
    "adj_mag_max",
    "adj_vy_mean",
    "adj_vy_std",
    "adj_vy_min",
    "adj_vy_max",
    "adj_vy_p10",
    "adj_vy_p90",
    "adj_upward_ratio",
    "adj_downward_ratio",
    "adj_max_upward_window_mean",
    "adj_max_downward_window_mean",
)
_EPS: Final[float] = 1e-6


@dataclass(frozen=True)
class ProbeSample:
    stem: str
    cohort: str
    label: int
    pred: int
    prob_dynamic: float
    video_path: Path


@dataclass(frozen=True)
class RoiProbeResult:
    sample: ProbeSample
    features: dict[str, float]
    sampled_frames: int
    flow_frames: int
    pose_frames: int
    roi_flow_frames: int
    fallback_flow_frames: int


@dataclass(frozen=True)
class ProbeComparison:
    name: str
    left_cohort: str
    right_cohort: str
    left_count: int
    right_count: int
    max_abs_effect_feature: str
    max_abs_effect_size: float
    top_effects: list[dict[str, float | str | bool]]


@dataclass(frozen=True)
class ProbeSummary:
    high_conf_fn_count: int
    correct_dynamic_count: int
    correct_static_count: int
    high_conf_fp_static_count: int
    pose_coverage_gte_70_count: int
    completed_count: int
    failed_count: int
    max_abs_effect_feature: str
    max_abs_effect_size: float
    top_effects: list[dict[str, float | str | bool]]
    comparisons: list[ProbeComparison]
    static_gate: dict[str, float | str | bool]
    failures: list[dict[str, str]]


def select_probe_samples(
    predictions_csv: Path,
    *,
    model: str = "rf",
    split: str = "group-kfold",
    high_confidence_threshold: float = 0.85,
    control_limit: int = 30,
    static_control_limit: int = 30,
    include_static_cohorts: bool = False,
    include_dynamic_cohorts: bool = True,
) -> list[ProbeSample]:
    """Select ROI probe cohorts from model predictions."""
    rows: list[dict[str, str]] = []
    with predictions_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") == model and row.get("split") == split:
                rows.append(row)

    high_conf_fn = [
        row
        for row in rows
        if row["label"] == "1"
        and row["pred"] == "0"
        and (1.0 - float(row["prob_dynamic"])) >= high_confidence_threshold
    ]
    correct_dynamic = [
        row for row in rows if row["label"] == "1" and row["pred"] == "1"
    ]
    correct_dynamic.sort(key=lambda row: (-float(row["prob_dynamic"]), row["stem"]))
    correct_static = [
        row for row in rows if row["label"] == "0" and row["pred"] == "0"
    ]
    correct_static.sort(key=lambda row: (float(row["prob_dynamic"]), row["stem"]))
    high_conf_fp_static = [
        row
        for row in rows
        if row["label"] == "0"
        and row["pred"] == "1"
        and float(row["prob_dynamic"]) >= high_confidence_threshold
    ]
    high_conf_fp_static.sort(key=lambda row: (-float(row["prob_dynamic"]), row["stem"]))

    samples: list[ProbeSample] = []
    if include_dynamic_cohorts:
        for row in high_conf_fn:
            samples.append(_sample_from_row(row, cohort="high_conf_fn"))
        for row in correct_dynamic[:control_limit]:
            samples.append(_sample_from_row(row, cohort="correct_dynamic"))
    if include_static_cohorts:
        for row in correct_static[:static_control_limit]:
            samples.append(_sample_from_row(row, cohort="correct_static"))
        for row in high_conf_fp_static:
            samples.append(_sample_from_row(row, cohort="high_conf_fp_static"))
    return samples


def probe_video_roi_flow(
    sample: ProbeSample,
    *,
    target_fps: int = 10,
    resize: tuple[int, int] = (320, 240),
    bbox_margin: float = 0.25,
    min_visibility: float = 0.5,
    task_model_path: str | None = "models/mediapipe/pose_landmarker_lite.task",
) -> RoiProbeResult:
    """Compute ROI and camera-subtracted flow features for one video."""
    frames, _src_fps = _sample_video_frames(sample.video_path, target_fps=target_fps, resize=resize)
    pose_frames = extract_pose_landmarks(
        frames,
        task_model_path=task_model_path,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    bbox_by_frame = {
        pose.frame_idx: _bbox_from_pose(
            pose,
            width=resize[0],
            height=resize[1],
            margin=bbox_margin,
            min_visibility=min_visibility,
        )
        for pose in pose_frames
    }

    roi_rows: list[tuple[float, float]] = []
    adj_rows: list[tuple[float, float]] = []
    roi_flow_frames = 0
    fallback_flow_frames = 0
    last_bbox: tuple[int, int, int, int] | None = None

    gray_frames = [
        (frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for frame_idx, _timestamp_ms, frame in frames
    ]
    for i in range(1, len(gray_frames)):
        frame_idx, curr = gray_frames[i]
        _prev_idx, prev = gray_frames[i - 1]
        bbox = bbox_by_frame.get(frame_idx)
        if bbox is not None:
            last_bbox = bbox
        else:
            bbox = last_bbox

        flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            np.zeros((curr.shape[0], curr.shape[1], 2), dtype=np.float32),
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_arr = cast(NDArray[np.float32], flow)
        if bbox is None:
            roi = flow_arr.reshape(-1, 2)
            bg_median = np.zeros(2, dtype=np.float32)
            fallback_flow_frames += 1
        else:
            x0, y0, x1, y1 = bbox
            roi = flow_arr[y0:y1, x0:x1].reshape(-1, 2)
            bg = _outside_bbox_flow(flow_arr, bbox)
            bg_median = np.median(bg, axis=0).astype(np.float32) if len(bg) else np.zeros(2, dtype=np.float32)
            roi_flow_frames += 1
        if len(roi) == 0:
            continue
        roi_rows.append(_flow_row(roi))
        adjusted = roi - bg_median.reshape(1, 2)
        adj_rows.append(_flow_row(adjusted))

    roi_series = np.asarray(roi_rows, dtype=np.float32)
    adj_series = np.asarray(adj_rows, dtype=np.float32)
    features = _combined_features(roi_series, adj_series, fps=target_fps)
    return RoiProbeResult(
        sample=sample,
        features=features,
        sampled_frames=len(frames),
        flow_frames=max(0, len(frames) - 1),
        pose_frames=len(pose_frames),
        roi_flow_frames=roi_flow_frames,
        fallback_flow_frames=fallback_flow_frames,
    )


def summarize_probe(
    results: list[RoiProbeResult],
    failures: list[dict[str, str]],
) -> ProbeSummary:
    """Summarize cohort effect sizes for each ROI feature."""
    high = [result for result in results if result.sample.cohort == "high_conf_fn"]
    controls = [result for result in results if result.sample.cohort == "correct_dynamic"]
    correct_static = [result for result in results if result.sample.cohort == "correct_static"]
    high_conf_fp_static = [result for result in results if result.sample.cohort == "high_conf_fp_static"]
    static_pool = [*correct_static, *high_conf_fp_static]
    coverage_results = [result for result in results if _pose_coverage(result) >= 0.70]
    coverage_high = [result for result in coverage_results if result.sample.cohort == "high_conf_fn"]
    coverage_dynamic = [result for result in coverage_results if result.sample.cohort == "correct_dynamic"]
    coverage_static = [result for result in coverage_results if result.sample.cohort == "correct_static"]
    coverage_static_pool = [
        result
        for result in coverage_results
        if result.sample.cohort in {"correct_static", "high_conf_fp_static"}
    ]

    comparisons = [
        _comparison_summary(
            name="high_conf_fn_vs_correct_dynamic",
            left=high,
            right=controls,
            left_cohort="high_conf_fn",
            right_cohort="correct_dynamic",
        ),
        _comparison_summary(
            name="high_conf_fn_vs_correct_static",
            left=high,
            right=correct_static,
            left_cohort="high_conf_fn",
            right_cohort="correct_static",
            dynamic_reference=controls,
        ),
        _comparison_summary(
            name="high_conf_fn_vs_static_pool",
            left=high,
            right=static_pool,
            left_cohort="high_conf_fn",
            right_cohort="static_pool",
            dynamic_reference=controls,
        ),
        _comparison_summary(
            name="high_conf_fn_vs_high_conf_fp_static",
            left=high,
            right=high_conf_fp_static,
            left_cohort="high_conf_fn",
            right_cohort="high_conf_fp_static",
            dynamic_reference=controls,
        ),
        _comparison_summary(
            name="high_conf_fn_vs_correct_static_pose_coverage_gte_0_70",
            left=coverage_high,
            right=coverage_static,
            left_cohort="high_conf_fn",
            right_cohort="correct_static",
            dynamic_reference=coverage_dynamic,
        ),
        _comparison_summary(
            name="high_conf_fn_vs_static_pool_pose_coverage_gte_0_70",
            left=coverage_high,
            right=coverage_static_pool,
            left_cohort="high_conf_fn",
            right_cohort="static_pool",
            dynamic_reference=coverage_dynamic,
        ),
    ]
    primary = comparisons[0]
    static_gate = _static_gate(comparisons, effect_threshold=1.0)
    return ProbeSummary(
        high_conf_fn_count=len(high),
        correct_dynamic_count=len(controls),
        correct_static_count=len(correct_static),
        high_conf_fp_static_count=len(high_conf_fp_static),
        pose_coverage_gte_70_count=len(coverage_results),
        completed_count=len(results),
        failed_count=len(failures),
        max_abs_effect_feature=primary.max_abs_effect_feature,
        max_abs_effect_size=primary.max_abs_effect_size,
        top_effects=primary.top_effects,
        comparisons=comparisons,
        static_gate=static_gate,
        failures=failures,
    )


def _comparison_summary(
    *,
    name: str,
    left: list[RoiProbeResult],
    right: list[RoiProbeResult],
    left_cohort: str,
    right_cohort: str,
    dynamic_reference: list[RoiProbeResult] | None = None,
) -> ProbeComparison:
    effects: list[dict[str, float | str | bool]] = []
    for feature in _FEATURE_NAMES:
        left_values = np.asarray([result.features[feature] for result in left], dtype=np.float32)
        right_values = np.asarray([result.features[feature] for result in right], dtype=np.float32)
        effect = _cohens_d(left_values, right_values)
        item: dict[str, float | str | bool] = {
            "feature": feature,
            "effect_size": round(effect, 4),
            "abs_effect_size": round(abs(effect), 4),
            f"{left_cohort}_mean": round(float(left_values.mean()), 6) if len(left_values) else 0.0,
            f"{right_cohort}_mean": round(float(right_values.mean()), 6) if len(right_values) else 0.0,
        }
        if dynamic_reference is not None:
            dynamic_values = np.asarray([result.features[feature] for result in dynamic_reference], dtype=np.float32)
            dynamic_mean = float(dynamic_values.mean()) if len(dynamic_values) else 0.0
            left_mean = float(left_values.mean()) if len(left_values) else 0.0
            right_mean = float(right_values.mean()) if len(right_values) else 0.0
            item["correct_dynamic_mean"] = round(dynamic_mean, 6)
            item["left_is_dynamic_side"] = _same_direction(
                candidate_delta=left_mean - right_mean,
                reference_delta=dynamic_mean - right_mean,
            )
        effects.append(item)
    effects.sort(key=lambda item: float(item["abs_effect_size"]), reverse=True)
    max_effect_feature = str(effects[0]["feature"]) if effects else ""
    max_effect_size = float(effects[0]["abs_effect_size"]) if effects else 0.0
    return ProbeComparison(
        name=name,
        left_cohort=left_cohort,
        right_cohort=right_cohort,
        left_count=len(left),
        right_count=len(right),
        max_abs_effect_feature=max_effect_feature,
        max_abs_effect_size=max_effect_size,
        top_effects=effects[:10],
    )


def _static_gate(
    comparisons: list[ProbeComparison],
    *,
    effect_threshold: float,
) -> dict[str, float | str | bool]:
    candidate_names = (
        "high_conf_fn_vs_correct_static",
        "high_conf_fn_vs_static_pool",
    )
    for comparison in comparisons:
        if comparison.name not in candidate_names:
            continue
        for effect in comparison.top_effects:
            if (
                float(effect["abs_effect_size"]) >= effect_threshold
                and effect.get("left_is_dynamic_side") is True
            ):
                return {
                    "passed": True,
                    "comparison": comparison.name,
                    "feature": str(effect["feature"]),
                    "effect_size": float(effect["effect_size"]),
                    "abs_effect_size": float(effect["abs_effect_size"]),
                    "threshold": effect_threshold,
                    "reason": "FN differs from static and points toward correct_dynamic.",
                }
    return {
        "passed": False,
        "comparison": "",
        "feature": "",
        "effect_size": 0.0,
        "abs_effect_size": 0.0,
        "threshold": effect_threshold,
        "reason": "No FN-vs-static feature exceeded threshold in the correct_dynamic direction.",
    }


def _same_direction(*, candidate_delta: float, reference_delta: float) -> bool:
    if abs(candidate_delta) <= _EPS or abs(reference_delta) <= _EPS:
        return False
    return candidate_delta * reference_delta > 0


def _pose_coverage(result: RoiProbeResult) -> float:
    if result.sampled_frames <= 0:
        return 0.0
    return result.pose_frames / result.sampled_frames


def write_results_csv(path: Path, results: list[RoiProbeResult]) -> None:
    """Write per-video ROI probe results."""
    fieldnames = [
        "stem",
        "cohort",
        "label",
        "pred",
        "prob_dynamic",
        "sampled_frames",
        "flow_frames",
        "pose_frames",
        "roi_flow_frames",
        "fallback_flow_frames",
        "video_path",
        *_FEATURE_NAMES,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row: dict[str, object] = {
                "stem": result.sample.stem,
                "cohort": result.sample.cohort,
                "label": result.sample.label,
                "pred": result.sample.pred,
                "prob_dynamic": f"{result.sample.prob_dynamic:.6f}",
                "sampled_frames": result.sampled_frames,
                "flow_frames": result.flow_frames,
                "pose_frames": result.pose_frames,
                "roi_flow_frames": result.roi_flow_frames,
                "fallback_flow_frames": result.fallback_flow_frames,
                "video_path": str(result.sample.video_path),
            }
            row.update({key: f"{value:.8f}" for key, value in result.features.items()})
            writer.writerow(row)


def read_results_csv(path: Path) -> list[RoiProbeResult]:
    """Read per-video ROI probe results previously written by this script."""
    results: list[RoiProbeResult] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            features = {name: float(row[name]) for name in _FEATURE_NAMES}
            results.append(
                RoiProbeResult(
                    sample=ProbeSample(
                        stem=row["stem"],
                        cohort=row["cohort"],
                        label=int(row["label"]),
                        pred=int(row["pred"]),
                        prob_dynamic=float(row["prob_dynamic"]),
                        video_path=Path(row["video_path"]),
                    ),
                    features=features,
                    sampled_frames=int(row["sampled_frames"]),
                    flow_frames=int(row["flow_frames"]),
                    pose_frames=int(row["pose_frames"]),
                    roi_flow_frames=int(row["roi_flow_frames"]),
                    fallback_flow_frames=int(row["fallback_flow_frames"]),
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--model", default="rf")
    parser.add_argument("--split", default="group-kfold")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--control-limit", type=int, default=30)
    parser.add_argument("--static-control-limit", type=int, default=30)
    parser.add_argument("--include-static-cohorts", action="store_true")
    parser.add_argument("--only-static-cohorts", action="store_true")
    parser.add_argument("--existing-results-csv", type=Path)
    parser.add_argument("--existing-summary", type=Path)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--resize-height", type=int, default=240)
    parser.add_argument("--task-model-path", default="models/mediapipe/pose_landmarker_lite.task")
    args = parser.parse_args()

    samples = select_probe_samples(
        args.predictions,
        model=args.model,
        split=args.split,
        high_confidence_threshold=args.high_confidence_threshold,
        control_limit=args.control_limit,
        static_control_limit=args.static_control_limit,
        include_static_cohorts=args.include_static_cohorts or args.only_static_cohorts,
        include_dynamic_cohorts=not args.only_static_cohorts,
    )
    results: list[RoiProbeResult] = read_results_csv(args.existing_results_csv) if args.existing_results_csv else []
    failures = _read_failures(args.existing_summary) if args.existing_summary else []
    completed_keys = {(result.sample.stem, result.sample.cohort) for result in results}
    samples_to_run = [
        sample for sample in samples if (sample.stem, sample.cohort) not in completed_keys
    ]
    for index, sample in enumerate(samples_to_run, start=1):
        print(f"[probe] {index}/{len(samples_to_run)} {sample.cohort} {sample.stem}", flush=True)
        try:
            results.append(
                probe_video_roi_flow(
                    sample,
                    target_fps=args.target_fps,
                    resize=(args.resize_width, args.resize_height),
                    task_model_path=args.task_model_path,
                )
            )
        except Exception as exc:
            failures.append({"stem": sample.stem, "cohort": sample.cohort, "error": repr(exc)})
            print(f"[warning] {sample.stem}: {exc!r}", flush=True)

    results = _sort_results(results)
    failures = _dedupe_failures(failures)
    write_results_csv(args.out_csv, results)
    summary = summarize_probe(results, failures)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0 if results else 1


def _read_failures(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = data.get("failures", [])
    if not isinstance(failures, list):
        return []
    out: list[dict[str, str]] = []
    for item in failures:
        if not isinstance(item, dict):
            continue
        stem = item.get("stem")
        cohort = item.get("cohort")
        error = item.get("error")
        if isinstance(stem, str) and isinstance(cohort, str) and isinstance(error, str):
            out.append({"stem": stem, "cohort": cohort, "error": error})
    return out


def _sort_results(results: list[RoiProbeResult]) -> list[RoiProbeResult]:
    cohort_order = {
        "high_conf_fn": 0,
        "correct_dynamic": 1,
        "correct_static": 2,
        "high_conf_fp_static": 3,
    }
    return sorted(
        results,
        key=lambda result: (
            cohort_order.get(result.sample.cohort, 99),
            result.sample.stem,
        ),
    )


def _dedupe_failures(failures: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for failure in failures:
        key = (failure["stem"], failure["cohort"])
        if key in seen:
            continue
        seen.add(key)
        out.append(failure)
    return out


def _sample_from_row(row: dict[str, str], *, cohort: str) -> ProbeSample:
    return ProbeSample(
        stem=row["stem"],
        cohort=cohort,
        label=int(row["label"]),
        pred=int(row["pred"]),
        prob_dynamic=float(row["prob_dynamic"]),
        video_path=Path(row["source_path"]),
    )


def _sample_video_frames(
    video_path: Path,
    *,
    target_fps: int,
    resize: tuple[int, int],
) -> tuple[list[tuple[int, int, NDArray[np.uint8]]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    frames: list[tuple[int, int, NDArray[np.uint8]]] = []
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0.0 or src_fps > 240.0:
            step = 1
            ms_per_frame = 0.0
        else:
            step = max(1, round(src_fps / target_fps))
            ms_per_frame = 1000.0 / src_fps
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                resized = cv2.resize(frame, resize)
                timestamp_ms = int(idx * ms_per_frame) if ms_per_frame > 0 else 0
                frames.append((idx, timestamp_ms, cast(NDArray[np.uint8], resized)))
            idx += 1
    finally:
        cap.release()
    if len(frames) < 5:
        raise ValueError(f"not enough sampled frames: {len(frames)}")
    return frames, src_fps


def _bbox_from_pose(
    pose: PoseFrame,
    *,
    width: int,
    height: int,
    margin: float,
    min_visibility: float,
) -> tuple[int, int, int, int] | None:
    landmarks = pose.landmarks
    visible = landmarks[:, 3] >= min_visibility
    points = landmarks[visible, :2]
    points = points[(points[:, 0] >= 0.0) & (points[:, 0] <= 1.0) & (points[:, 1] >= 0.0) & (points[:, 1] <= 1.0)]
    if len(points) < 5:
        return None
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, np.asarray([0.05, 0.05], dtype=np.float32))
    min_xy = np.maximum(min_xy - span * margin, 0.0)
    max_xy = np.minimum(max_xy + span * margin, 1.0)
    x0 = int(np.floor(float(min_xy[0]) * width))
    y0 = int(np.floor(float(min_xy[1]) * height))
    x1 = int(np.ceil(float(max_xy[0]) * width))
    y1 = int(np.ceil(float(max_xy[1]) * height))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _outside_bbox_flow(flow: NDArray[np.float32], bbox: tuple[int, int, int, int]) -> NDArray[np.float32]:
    x0, y0, x1, y1 = bbox
    mask = np.ones(flow.shape[:2], dtype=np.bool_)
    mask[y0:y1, x0:x1] = False
    return flow[mask].reshape(-1, 2)


def _flow_row(flow_vectors: NDArray[np.float32]) -> tuple[float, float]:
    magnitude = np.sqrt(flow_vectors[:, 0] ** 2 + flow_vectors[:, 1] ** 2)
    return float(magnitude.mean()), float(flow_vectors[:, 1].mean())


def _combined_features(
    roi_series: NDArray[np.float32],
    adj_series: NDArray[np.float32],
    *,
    fps: int,
) -> dict[str, float]:
    roi_features = _series_features(roi_series, prefix="roi", fps=fps)
    adj_features = _series_features(adj_series, prefix="adj", fps=fps)
    return {**roi_features, **adj_features}


def _series_features(series: NDArray[np.float32], *, prefix: str, fps: int) -> dict[str, float]:
    if series.ndim != 2 or series.shape[1] != 2 or len(series) < 2:
        return {name: 0.0 for name in _FEATURE_NAMES if name.startswith(prefix)}
    mag = series[:, 0]
    vy = series[:, 1]
    threshold = max(float(np.median(np.abs(vy))) * 2.0, _EPS)
    upward = vy < -threshold
    downward = vy > threshold
    win_means = _window_means(vy, window_sec=2.0, fps=fps)
    max_upward = float(np.max(-win_means)) if len(win_means) else 0.0
    max_downward = float(np.max(win_means)) if len(win_means) else 0.0
    return {
        f"{prefix}_mag_mean": float(np.mean(mag)),
        f"{prefix}_mag_p95": float(np.percentile(mag, 95)),
        f"{prefix}_mag_max": float(np.max(mag)),
        f"{prefix}_vy_mean": float(np.mean(vy)),
        f"{prefix}_vy_std": float(np.std(vy)),
        f"{prefix}_vy_min": float(np.min(vy)),
        f"{prefix}_vy_max": float(np.max(vy)),
        f"{prefix}_vy_p10": float(np.percentile(vy, 10)),
        f"{prefix}_vy_p90": float(np.percentile(vy, 90)),
        f"{prefix}_upward_ratio": float(upward.mean()),
        f"{prefix}_downward_ratio": float(downward.mean()),
        f"{prefix}_max_upward_window_mean": max_upward,
        f"{prefix}_max_downward_window_mean": max_downward,
    }


def _window_means(signal: NDArray[np.float32], *, window_sec: float, fps: int) -> NDArray[np.float32]:
    window_size = min(max(1, int(window_sec * fps)), max(1, len(signal) - 1))
    count = len(signal) - window_size + 1
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray([signal[i : i + window_size].mean() for i in range(count)], dtype=np.float32)


def _cohens_d(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled <= _EPS:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


if __name__ == "__main__":
    raise SystemExit(main())

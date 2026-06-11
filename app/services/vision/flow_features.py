"""Optical-flow feature extraction for dynamic/static baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

_DEFAULT_RESIZE: Final[tuple[int, int]] = (320, 240)
_DEFAULT_TARGET_FPS: Final[int] = 30
FLOW_FEATURE_VERSION: Final[str] = "flow_v4"
FLOW_FEATURE_DIM: Final[int] = 58
V3_FLOW_FEATURE_DIM: Final[int] = 46
LEGACY_FLOW_FEATURE_DIM: Final[int] = 42
_EPS: Final[float] = 1e-6


def extract_flow_series(
    video_path: Path,
    *,
    resize: tuple[int, int] = _DEFAULT_RESIZE,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> tuple[NDArray[np.float32], float, float]:
    """Extract mean Farneback optical-flow magnitude and vertical velocity.

    Returns an `(T, 2)` series where channel 0 is magnitude and channel 1 is
    `vy`. OpenCV image coordinates use +y downward, so positive `vy` means
    downward/fall-like motion and negative `vy` means upward/dyno-like motion.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if src_fps <= 0.0 or total_frames <= 1:
            raise ValueError(f"invalid video metadata: fps={src_fps}, frames={total_frames}")

        duration_sec = total_frames / src_fps
        sample_count = max(1, int(duration_sec * target_fps))
        sample_indices = set(np.linspace(0, total_frames - 1, sample_count).astype(int).tolist())

        frames: list[NDArray[np.uint8]] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in sample_indices:
                gray = cv2.cvtColor(cv2.resize(frame, resize), cv2.COLOR_BGR2GRAY)
                frames.append(cast(NDArray[np.uint8], gray))
            idx += 1
    finally:
        cap.release()

    if len(frames) < 5:
        raise ValueError(f"not enough valid frames: {len(frames)}")

    flow_rows: list[tuple[float, float]] = []
    for i in range(1, len(frames)):
        initial_flow = np.zeros((frames[i].shape[0], frames[i].shape[1], 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            frames[i - 1],
            frames[i],
            initial_flow,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flow_arr = cast(NDArray[np.float32], flow)
        magnitude = np.sqrt(flow_arr[..., 0] ** 2 + flow_arr[..., 1] ** 2).mean()
        vy = flow_arr[..., 1].mean()
        flow_rows.append((float(magnitude), float(vy)))

    return np.asarray(flow_rows, dtype=np.float32), src_fps, duration_sec


def extract_flow_magnitude(
    video_path: Path,
    *,
    resize: tuple[int, int] = _DEFAULT_RESIZE,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> tuple[NDArray[np.float32], float, float]:
    """Extract mean Farneback optical-flow magnitude at normalized FPS."""
    flow_series, src_fps, duration_sec = extract_flow_series(
        video_path,
        resize=resize,
        target_fps=target_fps,
    )
    return flow_series[:, 0], src_fps, duration_sec


def trim_fall_segment(
    flow_mag: NDArray[np.float32],
    *,
    spike_multiplier: float = 3.5,
    max_trim_ratio: float = 0.25,
    min_tail_frames: int = 2,
) -> NDArray[np.float32]:
    """Trim a contiguous terminal flow burst that likely comes from fall/slip motion."""
    signal = np.asarray(flow_mag, dtype=np.float32)
    if len(signal) < 5:
        return signal

    magnitude = signal[:, 0] if signal.ndim == 2 else signal
    baseline = float(np.median(magnitude))
    if baseline <= _EPS:
        baseline = float(np.percentile(magnitude, 50))
    threshold = max(baseline * spike_multiplier, baseline + _EPS)
    max_trim_frames = max(1, int(len(signal) * max_trim_ratio))

    trim_len = 0
    idx = len(signal) - 1
    while idx >= 0 and trim_len < max_trim_frames and float(magnitude[idx]) > threshold:
        trim_len += 1
        idx -= 1

    if trim_len < min_tail_frames:
        return signal

    trimmed = signal[: len(signal) - trim_len]
    if len(trimmed) < 5:
        return signal
    return trimmed


def remove_fall_end(flow_mag: NDArray[np.float32], tail_ratio: float = 0.25) -> NDArray[np.float32]:
    """Backward-compatible wrapper around dynamic terminal fall trimming."""
    return trim_fall_segment(flow_mag, max_trim_ratio=tail_ratio)


def extract_flow_stats(
    flow_series: NDArray[np.float32],
    *,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> NDArray[np.float32]:
    """Convert flow magnitude/vy series into the v4 58-dim feature vector."""
    series = np.asarray(flow_series, dtype=np.float32)
    magnitude, vy = _split_flow_series(series)
    feature_blocks = [
        _extract_magnitude_flow_stats(
            magnitude,
            target_fps=target_fps,
            include_burst_features=True,
        ),
        _extract_vy_stats(vy, target_fps=target_fps),
    ]
    features = np.concatenate(feature_blocks)
    return cast(NDArray[np.float32], features.astype(np.float32))


def extract_flow_stats_v3(
    flow_mag: NDArray[np.float32],
    *,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> NDArray[np.float32]:
    """Convert a 1D flow magnitude signal into the v3 46-dim feature vector."""
    return _extract_magnitude_flow_stats(flow_mag, target_fps=target_fps, include_burst_features=True)


def extract_flow_stats_legacy(
    flow_mag: NDArray[np.float32],
    *,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> NDArray[np.float32]:
    """Convert a 1D flow magnitude signal into the legacy v2 42-dim feature vector."""
    return _extract_magnitude_flow_stats(flow_mag, target_fps=target_fps, include_burst_features=False)


def _split_flow_series(flow_series: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if flow_series.ndim == 1:
        signal = flow_series.astype(np.float32)
        return signal, np.zeros_like(signal, dtype=np.float32)
    if flow_series.ndim == 2 and flow_series.shape[1] >= 2:
        return flow_series[:, 0].astype(np.float32), flow_series[:, 1].astype(np.float32)
    raise ValueError(f"flow series must have shape (T,) or (T, >=2), got {flow_series.shape}")


def _extract_magnitude_flow_stats(
    flow_mag: NDArray[np.float32],
    *,
    target_fps: int,
    include_burst_features: bool,
) -> NDArray[np.float32]:
    signal = np.asarray(flow_mag, dtype=np.float32)
    if len(signal) < 5:
        raise ValueError("flow signal must contain at least 5 values")

    smoothed = _smooth(signal)
    mean_flow = float(np.mean(smoothed))

    global_stats = np.asarray(
        [
            mean_flow,
            np.std(smoothed),
            np.max(smoothed),
            np.percentile(smoothed, 99),
            np.percentile(smoothed, 95),
            np.percentile(smoothed, 90),
        ],
        dtype=np.float32,
    )
    peak_ratios = np.asarray(
        [
            np.max(smoothed) / (mean_flow + _EPS),
            np.percentile(smoothed, 99) / (mean_flow + _EPS),
            np.percentile(smoothed, 95) / (mean_flow + _EPS),
            np.std(smoothed) / (mean_flow + _EPS),
        ],
        dtype=np.float32,
    )
    win_05 = _sliding_window_stats(smoothed, window_sec=0.5, fps=target_fps)
    win_10 = _sliding_window_stats(smoothed, window_sec=1.0, fps=target_fps)
    win_20 = _sliding_window_stats(smoothed, window_sec=2.0, fps=target_fps)
    dist_stats = np.asarray(
        [
            _skewness(smoothed),
            _kurtosis(smoothed),
            np.percentile(smoothed, 95) / (np.percentile(smoothed, 5) + _EPS),
        ],
        dtype=np.float32,
    )

    acc = np.abs(np.diff(smoothed))
    acc_feats = np.asarray(
        [
            np.max(acc),
            np.percentile(acc, 95),
            np.mean(acc),
            np.max(acc) / (np.mean(acc) + _EPS),
        ],
        dtype=np.float32,
    )
    threshold = np.percentile(smoothed, 70)
    is_active = smoothed > threshold
    max_streak = 0
    current_streak = 0
    for value in is_active:
        current_streak = current_streak + 1 if bool(value) else 0
        max_streak = max(max_streak, current_streak)

    hist, _ = np.histogram(smoothed, bins=20, density=False)
    hist_sum = float(hist.sum())
    if hist_sum <= _EPS:
        entropy = 0.0
    else:
        probs = hist.astype(np.float32) / hist_sum
        probs = probs[probs > 0.0]
        entropy = float(-np.sum(probs * np.log(probs)))
    motion_struct = np.asarray(
        [
            float(is_active.mean()),
            max_streak / len(smoothed),
            entropy,
        ],
        dtype=np.float32,
    )

    segment_size = max(1, len(smoothed) // 4)
    phase_means = np.asarray(
        [smoothed[i * segment_size : (i + 1) * segment_size].mean() for i in range(4)],
        dtype=np.float32,
    )
    feature_blocks = [
        global_stats,
        peak_ratios,
        win_05,
        win_10,
        win_20,
        dist_stats,
        acc_feats,
        motion_struct,
        phase_means,
    ]
    if include_burst_features:
        feature_blocks.append(_burst_window_stats(smoothed, window_sec=2.0, fps=target_fps))

    features = np.concatenate(feature_blocks)
    return cast(NDArray[np.float32], features.astype(np.float32))


def _extract_vy_stats(
    vy: NDArray[np.float32],
    *,
    target_fps: int,
) -> NDArray[np.float32]:
    signal = np.asarray(vy, dtype=np.float32)
    if len(signal) < 5:
        raise ValueError("flow signal must contain at least 5 values")

    smoothed = _smooth(signal)
    abs_baseline = float(np.median(np.abs(smoothed)))
    threshold = max(abs_baseline * 2.0, abs_baseline + _EPS)
    upward = smoothed < -threshold
    downward = smoothed > threshold
    win_means = _window_means(smoothed, window_sec=2.0, fps=target_fps)
    if len(win_means) == 0:
        max_upward_window_mean = 0.0
        max_downward_window_mean = 0.0
    else:
        max_upward_window_mean = float(np.max(-win_means))
        max_downward_window_mean = float(np.max(win_means))

    return np.asarray(
        [
            float(np.mean(smoothed)),
            float(np.std(smoothed)),
            float(np.min(smoothed)),
            float(np.max(smoothed)),
            float(np.percentile(smoothed, 10)),
            float(np.percentile(smoothed, 90)),
            float(_count_true_runs(upward)),
            float(upward.mean()),
            float(_count_true_runs(downward)),
            float(downward.mean()),
            max_upward_window_mean,
            max_downward_window_mean,
        ],
        dtype=np.float32,
    )


def _smooth(signal: NDArray[np.float32], window: int = 15) -> NDArray[np.float32]:
    window_length = min(window, len(signal) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, 3)
    return cast(
        NDArray[np.float32],
        savgol_filter(signal, window_length=window_length, polyorder=2).astype(np.float32),
    )


def _sliding_window_stats(
    signal: NDArray[np.float32],
    *,
    window_sec: float,
    fps: int = _DEFAULT_TARGET_FPS,
) -> NDArray[np.float32]:
    win_means = _window_means(signal, window_sec=window_sec, fps=fps)
    if len(win_means) == 0:
        return np.zeros(6, dtype=np.float32)

    window_size = _window_size(signal, window_sec=window_sec, fps=fps)
    count = len(win_means)
    win_maxes = np.asarray(
        [signal[i : i + window_size].max() for i in range(count)],
        dtype=np.float32,
    )
    global_mean = float(np.mean(signal))
    return np.asarray(
        [
            np.max(win_means),
            np.percentile(win_means, 95),
            np.max(win_maxes),
            np.std(win_means),
            np.max(win_means) / (global_mean + _EPS),
            np.max(win_maxes) / (global_mean + _EPS),
        ],
        dtype=np.float32,
    )


def _burst_window_stats(
    signal: NDArray[np.float32],
    *,
    window_sec: float,
    fps: int = _DEFAULT_TARGET_FPS,
    burst_multiplier: float = 2.0,
) -> NDArray[np.float32]:
    win_means = _window_means(signal, window_sec=window_sec, fps=fps)
    if len(win_means) == 0:
        return np.zeros(4, dtype=np.float32)

    top_count = min(3, len(win_means))
    top3_window_mean = float(np.mean(np.sort(win_means)[-top_count:]))
    baseline = float(np.median(signal))
    threshold = max(baseline * burst_multiplier, baseline + _EPS)
    is_burst = win_means > threshold
    return np.asarray(
        [
            float(np.max(win_means)),
            top3_window_mean,
            float(_count_true_runs(is_burst)),
            float(is_burst.mean()),
        ],
        dtype=np.float32,
    )


def _window_means(
    signal: NDArray[np.float32],
    *,
    window_sec: float,
    fps: int = _DEFAULT_TARGET_FPS,
) -> NDArray[np.float32]:
    window_size = _window_size(signal, window_sec=window_sec, fps=fps)
    count = len(signal) - window_size + 1
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(
        [signal[i : i + window_size].mean() for i in range(count)],
        dtype=np.float32,
    )


def _window_size(
    signal: NDArray[np.float32],
    *,
    window_sec: float,
    fps: int = _DEFAULT_TARGET_FPS,
) -> int:
    window_size = max(1, int(window_sec * fps))
    return min(window_size, len(signal) - 1)


def _count_true_runs(values: NDArray[np.bool_]) -> int:
    runs = 0
    in_run = False
    for value in values:
        if bool(value):
            if not in_run:
                runs += 1
                in_run = True
        else:
            in_run = False
    return runs


def _skewness(x: NDArray[np.float32]) -> float:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return float(np.mean(((x - mean) / (std + 1e-9)) ** 3))


def _kurtosis(x: NDArray[np.float32]) -> float:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return float(np.mean(((x - mean) / (std + 1e-9)) ** 4) - 3)

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
_EPS: Final[float] = 1e-6


def extract_flow_magnitude(
    video_path: Path,
    *,
    resize: tuple[int, int] = _DEFAULT_RESIZE,
    target_fps: int = _DEFAULT_TARGET_FPS,
) -> tuple[NDArray[np.float32], float, float]:
    """Extract mean Farneback optical-flow magnitude at normalized FPS."""
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

    flow_mags: list[float] = []
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
        flow_mags.append(float(magnitude))

    return np.asarray(flow_mags, dtype=np.float32), src_fps, duration_sec


def remove_fall_end(flow_mag: NDArray[np.float32], tail_ratio: float = 0.05) -> NDArray[np.float32]:
    """Trim a final tail spike that looks like camera/body fall motion."""
    signal = np.asarray(flow_mag, dtype=np.float32)
    tail_len = max(1, int(len(signal) * tail_ratio))
    tail = signal[-tail_len:]
    body = signal[:-tail_len]
    if len(body) == 0:
        return signal
    if float(tail.max()) > float(np.percentile(body, 99)) * 2.0:
        return body
    return signal


def extract_flow_stats(flow_mag: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert a 1D flow magnitude signal into the original 42-dim feature vector."""
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
    win_05 = _sliding_window_stats(smoothed, window_sec=0.5)
    win_10 = _sliding_window_stats(smoothed, window_sec=1.0)
    win_20 = _sliding_window_stats(smoothed, window_sec=2.0)
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

    hist, _ = np.histogram(smoothed, bins=20, density=True)
    hist = hist + 1e-9
    entropy = float(-np.sum(hist * np.log(hist)))
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
    features = np.concatenate(
        [
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
    )
    return cast(NDArray[np.float32], features.astype(np.float32))


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
    window_size = max(1, int(window_sec * fps))
    window_size = min(window_size, len(signal) - 1)
    count = len(signal) - window_size + 1
    if count <= 0:
        return np.zeros(6, dtype=np.float32)

    win_means = np.asarray(
        [signal[i : i + window_size].mean() for i in range(count)],
        dtype=np.float32,
    )
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


def _skewness(x: NDArray[np.float32]) -> float:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return float(np.mean(((x - mean) / (std + 1e-9)) ** 3))


def _kurtosis(x: NDArray[np.float32]) -> float:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return float(np.mean(((x - mean) / (std + 1e-9)) ** 4) - 3)

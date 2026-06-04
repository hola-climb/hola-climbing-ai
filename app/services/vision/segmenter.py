"""Movement-based segmentation — 연속 pose에서 동작 단위 구간을 검출.

vision-engineer 구현 영역.

알고리즘 (휴리스틱):
  1. 손/발 4개 landmark의 평균 속도가 'quiet' 임계 미만으로 연속 N 프레임 이상 지속되면
     해당 영역을 정지 구간으로 간주, 인접 동작과의 경계로 사용한다.
  2. 골반 y의 정점/저점(prominence ≥ 임계)을 추가 경계 후보로 잡는다 (다이노/락오프 신호).
  3. 위 경계들로 분할 후 너무 짧은 구간(< MIN_SEGMENT_DURATION_MS)은 인접 구간에 흡수.
  4. 너무 긴 구간(> MAX_SEGMENT_DURATION_MS)은 DEFAULT_WINDOW_MS 단위 sliding window로 재분할.

출력은 (start_ms, end_ms) 튜플의 list, 시간순 정렬·비중첩.
"""

from __future__ import annotations

import numpy as np

from app.services.vision._landmarks import (
    FOOT_IDX,
    HAND_IDX,
    pelvis_y,
    stack_landmarks,
)
from app.services.vision._thresholds import (
    DEFAULT_WINDOW_MS,
    MAX_SEGMENT_DURATION_MS,
    MIN_SEGMENT_DURATION_MS,
    MOTION_QUIET_VELOCITY,
    PELVIS_PEAK_PROMINENCE,
    QUIET_FRAMES_FOR_BOUNDARY,
)
from app.services.vision.pose import PoseFrame


def _detect_quiet_boundaries(arr: np.ndarray) -> list[int]:
    """손+발 평균 속도 기준 정지 구간 중앙 프레임 인덱스 반환."""
    limb_idx = HAND_IDX + FOOT_IDX
    # mean velocity time series (T-1,)
    from app.services.vision._landmarks import mean_velocity

    vel = mean_velocity(arr, limb_idx)
    quiet_mask = vel < MOTION_QUIET_VELOCITY  # (T-1,)
    boundaries: list[int] = []
    run_start: int | None = None
    for i, q in enumerate(quiet_mask):
        if q:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= QUIET_FRAMES_FOR_BOUNDARY:
                boundaries.append((run_start + i) // 2)
            run_start = None
    if run_start is not None and (len(quiet_mask) - run_start) >= QUIET_FRAMES_FOR_BOUNDARY:
        boundaries.append((run_start + len(quiet_mask)) // 2)
    return boundaries


def _detect_pelvis_extrema(arr: np.ndarray) -> list[int]:
    """골반 y의 prominence 정점/저점 프레임 인덱스 반환 (다이노/락오프 후보 경계).

    SciPy 의존을 피하기 위해 단순 1차 미분 부호 변화 + prominence 체크로 처리.
    """
    y = pelvis_y(arr)
    if len(y) < 3:
        return []
    extrema: list[int] = []
    for i in range(1, len(y) - 1):
        prev_d = y[i] - y[i - 1]
        next_d = y[i + 1] - y[i]
        if prev_d == 0 or next_d == 0:
            continue
        if (prev_d > 0 and next_d < 0) or (prev_d < 0 and next_d > 0):
            # prominence: 양쪽으로 가장 가까운 반대 극단점까지의 진폭
            left_window = y[max(0, i - 10) : i]
            right_window = y[i + 1 : min(len(y), i + 11)]
            if left_window.size == 0 or right_window.size == 0:
                continue
            local_amp = max(
                abs(float(y[i] - np.min(left_window))),
                abs(float(y[i] - np.max(left_window))),
                abs(float(y[i] - np.min(right_window))),
                abs(float(y[i] - np.max(right_window))),
            )
            if local_amp >= PELVIS_PEAK_PROMINENCE:
                extrema.append(i)
    return extrema


def _merge_and_clean(
    boundaries: list[int], total_frames: int, timestamps_ms: np.ndarray
) -> list[tuple[int, int]]:
    """경계 인덱스 → (start_ms, end_ms) 구간으로 변환, 짧은 구간 흡수 + 긴 구간 분할."""
    # boundary 정렬·중복 제거, 양 끝 포함
    bset = sorted({0, *boundaries, total_frames - 1})
    raw_segments: list[tuple[int, int]] = []
    for a, b in zip(bset[:-1], bset[1:]):
        if b <= a:
            continue
        raw_segments.append((int(timestamps_ms[a]), int(timestamps_ms[b])))

    # 짧은 구간 흡수
    merged: list[tuple[int, int]] = []
    for seg in raw_segments:
        duration = seg[1] - seg[0]
        if duration < MIN_SEGMENT_DURATION_MS and merged:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)
    # 첫 구간이 너무 짧으면 두 번째와 병합
    if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < MIN_SEGMENT_DURATION_MS:
        first = merged.pop(0)
        nxt = merged[0]
        merged[0] = (first[0], nxt[1])

    # 긴 구간 분할
    final: list[tuple[int, int]] = []
    for start_ms, end_ms in merged:
        dur = end_ms - start_ms
        if dur <= MAX_SEGMENT_DURATION_MS:
            final.append((start_ms, end_ms))
            continue
        cursor = start_ms
        while cursor + DEFAULT_WINDOW_MS < end_ms:
            final.append((cursor, cursor + DEFAULT_WINDOW_MS))
            cursor += DEFAULT_WINDOW_MS
        # 잔여
        if end_ms - cursor >= MIN_SEGMENT_DURATION_MS:
            final.append((cursor, end_ms))
        elif final:
            # 잔여가 너무 짧으면 마지막 구간에 흡수
            last = final[-1]
            final[-1] = (last[0], end_ms)
        else:
            final.append((cursor, end_ms))

    return final


def split_segments(pose_frames: list[PoseFrame]) -> list[tuple[int, int]]:
    """Pose 시퀀스를 동작 구간으로 분할한다.

    Args:
        pose_frames: 시간순 PoseFrame 리스트 (extract_pose_landmarks 결과).

    Returns:
        (start_time_ms, end_time_ms) 튜플 리스트, 시간순·비중첩.
        pose_frames가 비어있거나 1프레임뿐이면 빈 리스트.
    """
    if len(pose_frames) < 2:
        return []

    arr = stack_landmarks(pose_frames)
    timestamps_ms = np.asarray([pf.timestamp_ms for pf in pose_frames], dtype=np.int64)

    quiet_idx = _detect_quiet_boundaries(arr)
    pelvis_idx = _detect_pelvis_extrema(arr)

    # 경계 후보 합집합
    boundaries = sorted(set(quiet_idx) | set(pelvis_idx))

    # 경계가 하나도 없으면 전체를 단일/sliding window로
    if not boundaries:
        total_ms = int(timestamps_ms[-1] - timestamps_ms[0])
        if total_ms < MIN_SEGMENT_DURATION_MS:
            return []
        return _merge_and_clean([], len(pose_frames), timestamps_ms)

    return _merge_and_clean(boundaries, len(pose_frames), timestamps_ms)

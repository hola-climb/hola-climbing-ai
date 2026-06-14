"""Rule-based technique classifier — 6 클라이밍 기술 라벨링.

vision-engineer 구현 영역.

라벨 (콜백 body `technique` 필드 값으로 그대로 사용):
  - high_step
  - flagging
  - toe_hook
  - heel_hook
  - lock_off
  - dyno
  - coordination

각 segment에 대해 6+1 기술 score를 계산한 뒤,
  1) TECHNIQUE_PRIORITY 순서로 임계값을 만족하는 첫 기술을 채택
  2) 또는 score가 가장 높은 기술 (priority는 tiebreaker)
임계값을 어느 기술도 만족하지 못하면 해당 segment는 결과에서 drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.models.callback import AnalysisSegmentPayload
from app.services.vision._landmarks import (
    HAND_IDX,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT_INDEX,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    joint_angle_deg,
    mean_velocity,
    midpoint,
    pelvis_y,
    stack_landmarks,
    velocity_xy,
)
from app.services.vision._thresholds import (
    COORDINATION_LIMB_MOVE_VEL,
    COORDINATION_MIN_ACTIVE_FRAME_RATIO,
    COORDINATION_MIN_MOVING_FEET,
    COORDINATION_MIN_MOVING_HANDS,
    COORDINATION_MIN_MOVING_LIMBS,
    COORDINATION_MIN_SIMULTANEOUS_FRAMES,
    COORDINATION_WINDOW_MS,
    DYNAMIC_TECHNIQUES,
    DYNO_BOTH_HANDS_OFF_VEL,
    DYNO_PELVIS_VERTICAL_VEL,
    FLAGGING_CROSSING_THRESHOLD,
    FLAGGING_SAME_SIDE_RATIO,
    HEEL_HOOK_HEEL_ABOVE_FOOT_INDEX,
    HIGH_STEP_ANKLE_ABOVE_HIP_RATIO,
    HIGH_STEP_HAND_STATIC_VEL,
    HOOK_ANKLE_ABOVE_SUPPORT_DELTA,
    HOOK_LIFTED_FOOT_RATIO,
    LOCK_OFF_ELBOW_FLEX_STD,
    LOCK_OFF_HAND_STATIC_VEL,
    LOCK_OFF_MIN_ELBOW_FLEX_DEG,
    LOCK_OFF_PELVIS_Y_STD,
    MIN_CONFIDENCE_TO_EMIT,
    TECHNIQUE_PRIORITY,
    TOE_HOOK_FOOT_INDEX_ABOVE_HEEL,
)
from app.services.vision.pose import PoseFrame

TECHNIQUE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "high_step",
        "flagging",
        "toe_hook",
        "heel_hook",
        "lock_off",
        "dyno",
        "coordination",
    }
)


@dataclass(frozen=True)
class _Score:
    """단일 기술 점수. matched=True면 임계 통과."""

    technique: str
    matched: bool
    confidence: float  # 0.0~1.0


# --------------------------------------------------------------------
# 개별 기술 score 함수
# --------------------------------------------------------------------


def _slice_arr_by_time(
    pose_frames: list[PoseFrame], start_ms: int, end_ms: int
) -> NDArray[np.float32] | None:
    """[start_ms, end_ms] 범위의 landmark array (T, 33, 4). 부족하면 None."""
    selected = [pf for pf in pose_frames if start_ms <= pf.timestamp_ms <= end_ms]
    if len(selected) < 3:
        return None
    return stack_landmarks(selected)


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(min(1.0, max(0.0, numer / denom)))


def _score_high_step(arr: NDArray[np.float32]) -> _Score:
    """발목 y가 골반 y보다 위(=y 더 작음)인 프레임 비율 + 손 정적성."""
    hip_y = midpoint(arr, LEFT_HIP, RIGHT_HIP)[:, 1]
    l_ankle_y = arr[:, LEFT_ANKLE, 1]
    r_ankle_y = arr[:, RIGHT_ANKLE, 1]
    above = ((l_ankle_y < hip_y) | (r_ankle_y < hip_y)).astype(np.float32)
    ratio = float(above.mean())

    hand_vel = mean_velocity(arr, HAND_IDX) if arr.shape[0] >= 2 else np.zeros(1)
    hand_static = float(hand_vel.mean()) < HIGH_STEP_HAND_STATIC_VEL

    matched = ratio >= HIGH_STEP_ANKLE_ABOVE_HIP_RATIO and hand_static
    # confidence: ratio 기반 + static bonus
    conf = _ratio(ratio, 1.0) * (1.0 if hand_static else 0.6)
    return _Score("high_step", matched, conf)


def _score_flagging(arr: NDArray[np.float32]) -> _Score:
    """양발이 골반 중심선 기준 같은 쪽으로 몰린 cross-body 자세."""
    hip_center_x = midpoint(arr, LEFT_HIP, RIGHT_HIP)[:, 0]
    left_side = arr[:, LEFT_ANKLE, 0] - hip_center_x
    right_side = arr[:, RIGHT_ANKLE, 0] - hip_center_x

    same_left = (
        (left_side < -FLAGGING_CROSSING_THRESHOLD)
        & (right_side < -FLAGGING_CROSSING_THRESHOLD)
    )
    same_right = (
        (left_side > FLAGGING_CROSSING_THRESHOLD)
        & (right_side > FLAGGING_CROSSING_THRESHOLD)
    )
    same_side_ratio = float((same_left | same_right).mean())
    lateral_extent = float(np.mean(np.maximum(np.abs(left_side), np.abs(right_side))))

    matched = same_side_ratio >= FLAGGING_SAME_SIDE_RATIO
    conf = (
        _ratio(same_side_ratio, 1.0) * 0.75
        + _ratio(lateral_extent, FLAGGING_CROSSING_THRESHOLD * 4.0) * 0.25
    )
    return _Score("flagging", matched, conf)


def _score_hook(arr: NDArray[np.float32], *, is_toe: bool) -> _Score:
    """toe/heel hook 공통: 지지 발 대비 들린 발의 foot_index vs heel 상대 위치."""
    # 양쪽 발 중 더 위에 있는 발을 hook 후보로
    l_foot_y = arr[:, LEFT_ANKLE, 1]
    r_foot_y = arr[:, RIGHT_ANKLE, 1]
    use_left = float(l_foot_y.mean()) < float(r_foot_y.mean())
    ankle_y = l_foot_y if use_left else r_foot_y
    support_ankle_y = r_foot_y if use_left else l_foot_y
    foot_index_y = arr[:, LEFT_FOOT_INDEX if use_left else RIGHT_FOOT_INDEX, 1]
    heel_y = arr[:, LEFT_HEEL if use_left else RIGHT_HEEL, 1]

    lifted_ratio = float(
        (ankle_y < support_ankle_y - HOOK_ANKLE_ABOVE_SUPPORT_DELTA).mean()
    )

    # toe: foot_index가 heel보다 위 (foot_index_y < heel_y)
    # heel: heel이 foot_index보다 위 (heel_y < foot_index_y)
    if is_toe:
        toe_above_heel = heel_y - foot_index_y
        toe_above_ankle = ankle_y - foot_index_y
        delta = float(np.maximum(toe_above_heel, toe_above_ankle).mean())
        threshold = TOE_HOOK_FOOT_INDEX_ABOVE_HEEL
        label = "toe_hook"
    else:
        heel_above_toe = foot_index_y - heel_y
        heel_above_ankle = ankle_y - heel_y
        delta = float(np.maximum(heel_above_toe, heel_above_ankle).mean())
        threshold = HEEL_HOOK_HEEL_ABOVE_FOOT_INDEX
        label = "heel_hook"

    matched = lifted_ratio >= HOOK_LIFTED_FOOT_RATIO and delta >= threshold
    conf = (
        _ratio(lifted_ratio, 1.0) * 0.45
        + _ratio(delta, threshold * 3.0) * 0.55
    )
    return _Score(label, matched, conf)


def _score_lock_off(arr: NDArray[np.float32]) -> _Score:
    """양 팔꿈치 굽힘 변동성 작음 + 골반 y 안정 + 손 정적 + 최소 한 팔이 충분히 굽혀짐."""
    l_elbow = joint_angle_deg(arr, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    r_elbow = joint_angle_deg(arr, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    elbow_std = float(min(np.std(l_elbow), np.std(r_elbow)))
    min_elbow_mean = float(min(np.mean(l_elbow), np.mean(r_elbow)))

    py = pelvis_y(arr)
    pelvis_std = float(np.std(py))

    hand_vel = mean_velocity(arr, HAND_IDX) if arr.shape[0] >= 2 else np.zeros(1)
    hand_static = float(hand_vel.mean()) < LOCK_OFF_HAND_STATIC_VEL

    matched = (
        elbow_std < LOCK_OFF_ELBOW_FLEX_STD
        and pelvis_std < LOCK_OFF_PELVIS_Y_STD
        and hand_static
        and min_elbow_mean < LOCK_OFF_MIN_ELBOW_FLEX_DEG
    )
    # confidence: 임계 대비 여유도 (작을수록 좋음 = 1 - normalized)
    conf = (
        (1.0 - _ratio(elbow_std, LOCK_OFF_ELBOW_FLEX_STD * 2.0)) * 0.35
        + (1.0 - _ratio(pelvis_std, LOCK_OFF_PELVIS_Y_STD * 2.0)) * 0.35
        + (1.0 if hand_static else 0.0) * 0.15
        + (1.0 - _ratio(min_elbow_mean, 180.0)) * 0.15
    )
    return _Score("lock_off", matched, max(0.0, min(1.0, conf)))


def _score_dyno(arr: NDArray[np.float32], durations_ms: NDArray[np.int64]) -> _Score:
    """골반 y 속도의 정점 큰 값 + 양손이 동시에 hold 이탈한 프레임 존재."""
    if arr.shape[0] < 3:
        return _Score("dyno", False, 0.0)
    py = pelvis_y(arr)
    py_vel = np.abs(np.diff(py))
    max_py_vel = float(np.max(py_vel))

    l_wrist_v = velocity_xy(arr, LEFT_WRIST)
    r_wrist_v = velocity_xy(arr, RIGHT_WRIST)
    both_off_frames = int(
        np.sum((l_wrist_v >= DYNO_BOTH_HANDS_OFF_VEL) & (r_wrist_v >= DYNO_BOTH_HANDS_OFF_VEL))
    )

    matched = max_py_vel >= DYNO_PELVIS_VERTICAL_VEL and both_off_frames >= 1
    conf = (
        _ratio(max_py_vel, DYNO_PELVIS_VERTICAL_VEL * 3.0) * 0.6
        + _ratio(float(both_off_frames), 5.0) * 0.4
    )
    return _Score("dyno", matched, conf)


def _score_coordination(
    arr: NDArray[np.float32], timestamps_ms: NDArray[np.int64]
) -> _Score:
    """짧은 시간 window 내에 4 limb 중 3개 이상이 지속적으로 함께 이동."""
    if arr.shape[0] < 3:
        return _Score("coordination", False, 0.0)
    # 4 limb 각각의 속도 시계열 (T-1,)
    limb_indices = [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]
    vels = np.stack([velocity_xy(arr, i) for i in limb_indices], axis=0)  # (4, T-1)
    moving = vels >= COORDINATION_LIMB_MOVE_VEL  # (4, T-1)

    # window 단위 동시성: 각 시점에서 forward window 내 limb별 max
    ts_diff = np.diff(timestamps_ms)
    frame_ms = float(np.median(ts_diff)) if ts_diff.size > 0 else 33.0
    window_frames = max(1, int(COORDINATION_WINDOW_MS / max(1.0, frame_ms)))

    frame_count = moving.shape[1]
    best_conf = 0.0
    best_matched = False
    for start in range(0, max(1, frame_count - window_frames + 1)):
        end = min(frame_count, start + window_frames)
        win = moving[:, start:end]
        if win.shape[1] == 0:
            continue

        active_ratio_by_limb = win.mean(axis=1)
        active_limb_count = int(
            np.sum(active_ratio_by_limb >= COORDINATION_MIN_ACTIVE_FRAME_RATIO)
        )
        active_by_limb = active_ratio_by_limb >= COORDINATION_MIN_ACTIVE_FRAME_RATIO
        active_hand_count = int(np.sum(active_by_limb[:2]))
        active_foot_count = int(np.sum(active_by_limb[2:]))
        qualified_moving = win & active_by_limb[:, np.newaxis]
        simultaneous_frames = int(
            np.sum(qualified_moving.sum(axis=0) >= COORDINATION_MIN_MOVING_LIMBS)
        )

        limb_conf = _ratio(float(active_limb_count), 4.0)
        simult_conf = _ratio(
            float(simultaneous_frames),
            float(max(COORDINATION_MIN_SIMULTANEOUS_FRAMES, win.shape[1])),
        )
        conf = limb_conf * 0.55 + simult_conf * 0.45
        if conf > best_conf:
            best_conf = conf
        if (
            active_limb_count >= COORDINATION_MIN_MOVING_LIMBS
            and active_hand_count >= COORDINATION_MIN_MOVING_HANDS
            and active_foot_count >= COORDINATION_MIN_MOVING_FEET
            and simultaneous_frames >= COORDINATION_MIN_SIMULTANEOUS_FRAMES
        ):
            best_matched = True

    return _Score("coordination", best_matched, best_conf if best_matched else 0.0)


# --------------------------------------------------------------------
# 통합
# --------------------------------------------------------------------


def _classify_one_segment(
    pose_frames: list[PoseFrame], start_ms: int, end_ms: int
) -> tuple[str, float, bool] | None:
    """단일 segment를 (technique, confidence, is_dynamic)로 분류. 해당 없으면 None."""
    arr = _slice_arr_by_time(pose_frames, start_ms, end_ms)
    if arr is None:
        return None

    selected = [pf for pf in pose_frames if start_ms <= pf.timestamp_ms <= end_ms]
    timestamps_ms = np.asarray([pf.timestamp_ms for pf in selected], dtype=np.int64)
    durations_ms = np.diff(timestamps_ms) if len(timestamps_ms) > 1 else np.array([0])

    scores: dict[str, _Score] = {
        "high_step": _score_high_step(arr),
        "flagging": _score_flagging(arr),
        "toe_hook": _score_hook(arr, is_toe=True),
        "heel_hook": _score_hook(arr, is_toe=False),
        "lock_off": _score_lock_off(arr),
        "dyno": _score_dyno(arr, durations_ms),
        "coordination": _score_coordination(arr, timestamps_ms),
    }

    # 1) priority 순으로 matched 첫 기술 채택
    for tech in TECHNIQUE_PRIORITY:
        s = scores.get(tech)
        if s and s.matched and s.confidence >= MIN_CONFIDENCE_TO_EMIT:
            return s.technique, float(s.confidence), tech in DYNAMIC_TECHNIQUES

    return None


def classify_segments(
    pose_frames: list[PoseFrame],
    segments: list[tuple[int, int]],
) -> list[AnalysisSegmentPayload]:
    """각 구간에 대해 6+1 기술 라벨 중 하나를 부여한다.

    Args:
        pose_frames: 전체 pose 시퀀스 (시간순).
        segments: split_segments의 결과 (start_ms, end_ms) 리스트.

    Returns:
        AnalysisSegmentPayload 리스트.
        sequence_index는 결과 emit 순서대로 0부터 부여.
        매칭 기술이 없거나 confidence가 임계 미만이면 해당 segment는 결과에서 제외.
    """
    out: list[AnalysisSegmentPayload] = []
    if not pose_frames or not segments:
        return out

    seq = 0
    for start_ms, end_ms in segments:
        result = _classify_one_segment(pose_frames, start_ms, end_ms)
        if result is None:
            continue
        technique, confidence, is_dynamic = result
        if technique not in TECHNIQUE_LABELS:
            # 방어적 가드: 알 수 없는 라벨은 skip
            continue
        out.append(
            AnalysisSegmentPayload(
                sequence_index=seq,
                start_time_ms=int(start_ms),
                end_time_ms=int(end_ms),
                technique=technique,
                is_dynamic=is_dynamic,
                confidence=round(float(confidence), 3),
            )
        )
        seq += 1
    return out

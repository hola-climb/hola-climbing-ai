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
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT_INDEX,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    center_of_mass_x,
    joint_angle_deg,
    mean_velocity,
    midpoint,
    opposite_ankle,
    pelvis_y,
    stack_landmarks,
    support_foot_index,
    velocity_xy,
)
from app.services.vision._thresholds import (
    COORDINATION_LIMB_MOVE_VEL,
    COORDINATION_MIN_MOVING_LIMBS,
    COORDINATION_WINDOW_MS,
    DYNAMIC_TECHNIQUES,
    DYNO_BOTH_HANDS_OFF_VEL,
    DYNO_PELVIS_VERTICAL_VEL,
    FLAGGING_COM_LATERAL_OFFSET,
    FLAGGING_LEG_LATERAL_OFFSET,
    HEEL_HOOK_HEEL_ABOVE_FOOT_INDEX,
    HIGH_STEP_ANKLE_ABOVE_HIP_RATIO,
    HIGH_STEP_HAND_STATIC_VEL,
    HOOK_FOOT_ABOVE_KNEE_RATIO,
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
    """지지 발 기준 반대쪽 다리가 반대 방향으로 외측 확장 + 무게중심 외측 이동."""
    support_idx = support_foot_index(arr)  # ankle index
    opp_idx = opposite_ankle(support_idx)

    support_x = arr[:, support_idx, 0].mean()
    opp_x = arr[:, opp_idx, 0].mean()
    com_x = float(center_of_mass_x(arr).mean())

    # 지지 발이 left(=화면 x 작음 측)인지 right(=화면 x 큼 측)인지 일관성 가정
    # opp ankle이 support 반대 방향으로 멀리: |opp_x - support_x|
    leg_lateral = abs(float(opp_x - support_x))
    # COM이 지지 발 반대 방향으로 이동했는가
    com_lateral = abs(com_x - float(support_x))

    matched = (
        leg_lateral >= FLAGGING_LEG_LATERAL_OFFSET
        and com_lateral >= FLAGGING_COM_LATERAL_OFFSET
    )
    conf = (
        _ratio(leg_lateral, FLAGGING_LEG_LATERAL_OFFSET * 2.0) * 0.6
        + _ratio(com_lateral, FLAGGING_COM_LATERAL_OFFSET * 2.0) * 0.4
    )
    return _Score("flagging", matched, conf)


def _score_hook(arr: NDArray[np.float32], *, is_toe: bool) -> _Score:
    """toe/heel hook 공통: 발이 무릎보다 위인 프레임 비율 + foot_index vs heel 상대 위치."""
    # 양쪽 발 중 더 위에 있는 발을 hook 후보로
    l_foot_y = arr[:, LEFT_ANKLE, 1]
    r_foot_y = arr[:, RIGHT_ANKLE, 1]
    use_left = float(l_foot_y.mean()) < float(r_foot_y.mean())
    foot_y = l_foot_y if use_left else r_foot_y
    knee_y = arr[:, LEFT_KNEE if use_left else RIGHT_KNEE, 1]
    foot_index_y = arr[:, LEFT_FOOT_INDEX if use_left else RIGHT_FOOT_INDEX, 1]
    heel_y = arr[:, LEFT_HEEL if use_left else RIGHT_HEEL, 1]

    above_knee_ratio = float((foot_y < knee_y).mean())

    # toe: foot_index가 heel보다 위 (foot_index_y < heel_y)
    # heel: heel이 foot_index보다 위 (heel_y < foot_index_y)
    if is_toe:
        delta = float((heel_y - foot_index_y).mean())  # 양수면 toe
        threshold = TOE_HOOK_FOOT_INDEX_ABOVE_HEEL
        label = "toe_hook"
    else:
        delta = float((foot_index_y - heel_y).mean())  # 양수면 heel
        threshold = HEEL_HOOK_HEEL_ABOVE_FOOT_INDEX
        label = "heel_hook"

    matched = (
        above_knee_ratio >= HOOK_FOOT_ABOVE_KNEE_RATIO and delta >= threshold
    )
    conf = (
        _ratio(above_knee_ratio, 1.0) * 0.5
        + _ratio(delta, threshold * 2.5) * 0.5
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
    """짧은 시간 window 내에 4 limb 중 3개 이상이 동시에 이동."""
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
    max_simul = 0
    for start in range(0, max(1, frame_count - window_frames + 1)):
        end = min(frame_count, start + window_frames)
        win = moving[:, start:end]
        # 이 윈도우 안에서 limb별로 한 번이라도 움직였는가
        any_moved = win.any(axis=1)  # (4,)
        simul = int(any_moved.sum())
        if simul > max_simul:
            max_simul = simul

    matched = max_simul >= COORDINATION_MIN_MOVING_LIMBS
    conf = _ratio(float(max_simul), 4.0)
    return _Score("coordination", matched, conf)


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

    # 2) matched가 하나도 없으면 score 최대값 후보 (priority는 tiebreaker)
    best: _Score | None = None
    best_priority = len(TECHNIQUE_PRIORITY)
    for tech in TECHNIQUE_PRIORITY:
        s = scores[tech]
        pri = TECHNIQUE_PRIORITY.index(tech)
        if best is None or s.confidence > best.confidence or (
            s.confidence == best.confidence and pri < best_priority
        ):
            best = s
            best_priority = pri
    if best is None or best.confidence < MIN_CONFIDENCE_TO_EMIT:
        return None
    return best.technique, float(best.confidence), best.technique in DYNAMIC_TECHNIQUES


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

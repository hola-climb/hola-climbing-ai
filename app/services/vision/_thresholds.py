"""Centralized threshold constants for rule-based technique classification.

본 파일은 vision/_workspace/02_vision_technique_rules.md 와 1:1 동기화 대상.
값 변경 시 두 파일을 함께 갱신할 것.

좌표계 가정:
  - MediaPipe Pose의 normalized landmark (x,y in [0,1], y는 아래로 갈수록 큼).
  - z는 카메라 깊이(상대값), visibility는 0~1.
"""

from __future__ import annotations

from typing import Final

# --- 공통 ---
MIN_LANDMARK_VISIBILITY: Final[float] = 0.5
"""신뢰할 수 있는 landmark visibility 하한."""

MIN_SEGMENT_DURATION_MS: Final[int] = 400
"""너무 짧은 segment는 노이즈로 간주하고 drop."""

MAX_SEGMENT_DURATION_MS: Final[int] = 4000
"""너무 긴 segment는 sliding window로 강제 분할."""

DEFAULT_WINDOW_MS: Final[int] = 1500
"""sliding window 기본 길이 (정점/저점 검출 후 잔여 구간용)."""

# --- segmenter ---
MOTION_QUIET_VELOCITY: Final[float] = 0.008
"""정규화 좌표 기준 프레임당 손/발 속도가 이 값 미만 → 정지 프레임으로 간주."""

QUIET_FRAMES_FOR_BOUNDARY: Final[int] = 6
"""연속 N 프레임 정지하면 segment 경계 후보 (≈ 200ms @ 30fps)."""

PELVIS_PEAK_PROMINENCE: Final[float] = 0.04
"""골반 y의 정점/저점 검출 prominence (정규화 좌표). 다이노/락오프 후보 구간 추출."""

# --- high_step ---
HIGH_STEP_ANKLE_ABOVE_HIP_RATIO: Final[float] = 0.5
"""segment 프레임 중 발목 y가 골반 y보다 위(=y값이 더 작음)인 비율 임계."""

HIGH_STEP_HAND_STATIC_VEL: Final[float] = 0.012
"""hand 평균 속도가 이 값 미만 → '손은 정적' 조건."""

# --- flagging ---
FLAGGING_LEG_LATERAL_OFFSET: Final[float] = 0.08
"""반대쪽 다리의 x 오프셋 (지지 발 기준 반대 방향으로 외측 이동). 정규화 좌표."""

FLAGGING_COM_LATERAL_OFFSET: Final[float] = 0.04
"""무게중심(어깨+골반 중점) x가 지지 발 기준 외측으로 얼마나 이동했는지."""

# --- toe_hook / heel_hook ---
HOOK_FOOT_ABOVE_KNEE_RATIO: Final[float] = 0.4
"""segment 중 발(foot_index or heel) y가 무릎 y보다 위인 비율."""

TOE_HOOK_FOOT_INDEX_ABOVE_HEEL: Final[float] = 0.02
"""toe_hook: foot_index y < heel y - 임계 (발끝이 뒤꿈치보다 위)."""

HEEL_HOOK_HEEL_ABOVE_FOOT_INDEX: Final[float] = 0.02
"""heel_hook: heel y < foot_index y - 임계 (뒤꿈치가 발끝보다 위)."""

# --- lock_off ---
LOCK_OFF_ELBOW_FLEX_STD: Final[float] = 8.0  # degrees
"""양 팔꿈치 각도의 표준편차가 이 값 미만 → 굽힘 변동성 작음."""

LOCK_OFF_PELVIS_Y_STD: Final[float] = 0.012
"""골반 y 표준편차 임계 (정적 고정)."""

LOCK_OFF_HAND_STATIC_VEL: Final[float] = 0.010
"""상위 손이 거의 정지."""

LOCK_OFF_MIN_ELBOW_FLEX_DEG: Final[float] = 70.0
"""최소 한쪽 팔의 평균 굽힘 각이 이 값 이하 (= 충분히 굽혀져 있음). 180=완전 폄."""

# --- dyno ---
DYNO_PELVIS_VERTICAL_VEL: Final[float] = 0.020
"""골반 y의 프레임 속도 (정규화/frame)가 이 값 이상이 정점에 존재."""

DYNO_BOTH_HANDS_OFF_VEL: Final[float] = 0.025
"""양손 속도가 동시에 이 값 이상인 프레임 1개 이상 → '양손 hold 이탈' 추정."""

# --- coordination ---
COORDINATION_LIMB_MOVE_VEL: Final[float] = 0.015
"""사지(손2+발2) 중 한 limb의 평균 속도가 이 값 이상이면 '움직이는 limb'."""

COORDINATION_MIN_MOVING_LIMBS: Final[int] = 3
"""짧은 시간 window 내에 동시에 움직인 limb 수 임계."""

COORDINATION_WINDOW_MS: Final[int] = 600
"""coordination 동조성 판정 시간 window."""

# --- 분류 priority (동시 매칭 시 우선순위 결정) ---
TECHNIQUE_PRIORITY: Final[tuple[str, ...]] = (
    "dyno",          # 가장 dynamic, 우선
    "coordination",
    "toe_hook",
    "heel_hook",
    "high_step",
    "flagging",
    "lock_off",      # 정적, 마지막
)

DYNAMIC_TECHNIQUES: Final[frozenset[str]] = frozenset({"dyno", "coordination"})

# --- confidence 정규화 ---
MIN_CONFIDENCE_TO_EMIT: Final[float] = 0.35
"""이 미만이면 segment 자체를 drop."""

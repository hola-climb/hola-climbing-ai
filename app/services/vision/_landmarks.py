"""MediaPipe Pose landmark index 상수 + 기하/통계 헬퍼.

좌표계:
  - x, y: image-normalized [0, 1]. y는 위에서 아래로 증가.
  - 즉 "위쪽에 있다" = y가 작다.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# MediaPipe Pose 33-keypoint index (공식 문서)
NOSE: Final[int] = 0
LEFT_SHOULDER: Final[int] = 11
RIGHT_SHOULDER: Final[int] = 12
LEFT_ELBOW: Final[int] = 13
RIGHT_ELBOW: Final[int] = 14
LEFT_WRIST: Final[int] = 15
RIGHT_WRIST: Final[int] = 16
LEFT_HIP: Final[int] = 23
RIGHT_HIP: Final[int] = 24
LEFT_KNEE: Final[int] = 25
RIGHT_KNEE: Final[int] = 26
LEFT_ANKLE: Final[int] = 27
RIGHT_ANKLE: Final[int] = 28
LEFT_HEEL: Final[int] = 29
RIGHT_HEEL: Final[int] = 30
LEFT_FOOT_INDEX: Final[int] = 31
RIGHT_FOOT_INDEX: Final[int] = 32

HAND_IDX: Final[tuple[int, int]] = (LEFT_WRIST, RIGHT_WRIST)
FOOT_IDX: Final[tuple[int, int]] = (LEFT_ANKLE, RIGHT_ANKLE)
HIP_IDX: Final[tuple[int, int]] = (LEFT_HIP, RIGHT_HIP)
SHOULDER_IDX: Final[tuple[int, int]] = (LEFT_SHOULDER, RIGHT_SHOULDER)
KNEE_IDX: Final[tuple[int, int]] = (LEFT_KNEE, RIGHT_KNEE)


def stack_landmarks(pose_frames) -> np.ndarray:
    """PoseFrame 리스트의 landmarks를 (T, 33, 4) array로 쌓는다."""
    return np.stack([pf.landmarks for pf in pose_frames], axis=0)


def midpoint(arr: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    """(T, 33, 4)에서 두 landmark의 중점 (T, 4)를 반환."""
    return (arr[:, idx_a, :] + arr[:, idx_b, :]) * 0.5


def velocity_xy(arr: np.ndarray, idx: int) -> np.ndarray:
    """단일 landmark의 프레임당 xy 속도 (T-1,) 스칼라 (L2)."""
    pts = arr[:, idx, :2]
    diff = np.diff(pts, axis=0)
    return np.linalg.norm(diff, axis=1)


def mean_velocity(arr: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    """여러 landmark의 평균 속도 시계열 (T-1,)."""
    vs = [velocity_xy(arr, i) for i in indices]
    return np.mean(np.stack(vs, axis=0), axis=0)


def joint_angle_deg(
    arr: np.ndarray, a_idx: int, vertex_idx: int, c_idx: int
) -> np.ndarray:
    """각 vertex에서 a-vertex-c 의 각도 (도). shape (T,).

    팔꿈치 굽힘각: shoulder-elbow-wrist.
    180도 ≈ 완전 펴짐, 0도 ≈ 완전 굽힘.
    """
    a = arr[:, a_idx, :2]
    v = arr[:, vertex_idx, :2]
    c = arr[:, c_idx, :2]
    va = a - v
    vc = c - v
    # 분모 보호
    na = np.linalg.norm(va, axis=1) + 1e-8
    nc = np.linalg.norm(vc, axis=1) + 1e-8
    cos = np.einsum("ij,ij->i", va, vc) / (na * nc)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def pelvis_y(arr: np.ndarray) -> np.ndarray:
    """골반 중점 y 시계열 (T,)."""
    return midpoint(arr, LEFT_HIP, RIGHT_HIP)[:, 1]


def center_of_mass_x(arr: np.ndarray) -> np.ndarray:
    """간이 무게중심 x = (어깨 중점 x + 골반 중점 x) / 2."""
    sh = midpoint(arr, LEFT_SHOULDER, RIGHT_SHOULDER)[:, 0]
    hp = midpoint(arr, LEFT_HIP, RIGHT_HIP)[:, 0]
    return (sh + hp) * 0.5


def support_foot_index(arr: np.ndarray) -> int:
    """전체 segment에서 더 아래쪽(y가 큰)에 평균적으로 위치한 발 = 지지 발 추정."""
    ly = float(np.mean(arr[:, LEFT_ANKLE, 1]))
    ry = float(np.mean(arr[:, RIGHT_ANKLE, 1]))
    return LEFT_ANKLE if ly > ry else RIGHT_ANKLE


def opposite_ankle(support_ankle_idx: int) -> int:
    return RIGHT_ANKLE if support_ankle_idx == LEFT_ANKLE else LEFT_ANKLE

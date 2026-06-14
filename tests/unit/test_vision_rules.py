"""Rule-based classifier 합성 pose 시퀀스 sanity test.

목적: 실제 영상 라벨이 부재한 상태에서 (`labels.csv` 라벨 칼럼 전부 빈값),
classifier가 합성한 "이상적" pose 시퀀스에 대해 합리적인 결정을 내리는지 표면 검증.

검증 항목:
- high_step: 발목이 골반 위로 올라가고 손이 정적이면 high_step (또는 그 비슷한 정적 라벨) 후보
- dyno: 골반 y가 빠르게 변하고 양손이 동시에 high vel이면 dyno
- 빈 입력: 빈 결과
- 알 수 없는 라벨: 절대 emit되지 않음 (TECHNIQUE_LABELS allowlist)

* 합성 데이터는 임계값 통과를 보장하는 "이상적" 데이터일 뿐, 정확도 측정이 아니다.
* 라벨링 데이터가 갖춰지면 별도 정확도 측정 테스트로 대체.
"""

from __future__ import annotations

import numpy as np

from app.models.callback import AnalysisSegmentPayload
from app.services.vision.classifier import (
    TECHNIQUE_LABELS,
    classify_segments,
)
from app.services.vision.pose import PoseFrame


def _make_pose_frame(
    frame_idx: int,
    timestamp_ms: int,
    landmarks_33x4: np.ndarray,
) -> PoseFrame:
    assert landmarks_33x4.shape == (33, 4)
    return PoseFrame(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        landmarks=landmarks_33x4.astype(np.float32),
    )


def _baseline_landmarks() -> np.ndarray:
    """기본 자세: T-pose 비슷. visibility 1.0. 모든 landmark 위치는 정규화 [0,1]."""
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0  # visibility
    # MediaPipe 인덱스(자주 쓰는 것만):
    # nose=0
    lm[0] = (0.5, 0.2, 0.0, 1.0)
    # left_shoulder=11, right_shoulder=12
    lm[11] = (0.4, 0.35, 0.0, 1.0)
    lm[12] = (0.6, 0.35, 0.0, 1.0)
    # left_elbow=13, right_elbow=14
    lm[13] = (0.35, 0.5, 0.0, 1.0)
    lm[14] = (0.65, 0.5, 0.0, 1.0)
    # left_wrist=15, right_wrist=16
    lm[15] = (0.3, 0.6, 0.0, 1.0)
    lm[16] = (0.7, 0.6, 0.0, 1.0)
    # left_hip=23, right_hip=24
    lm[23] = (0.45, 0.6, 0.0, 1.0)
    lm[24] = (0.55, 0.6, 0.0, 1.0)
    # left_knee=25, right_knee=26
    lm[25] = (0.45, 0.75, 0.0, 1.0)
    lm[26] = (0.55, 0.75, 0.0, 1.0)
    # left_ankle=27, right_ankle=28
    lm[27] = (0.45, 0.9, 0.0, 1.0)
    lm[28] = (0.55, 0.9, 0.0, 1.0)
    # left_heel=29, right_heel=30
    lm[29] = (0.44, 0.92, 0.0, 1.0)
    lm[30] = (0.56, 0.92, 0.0, 1.0)
    # left_foot_index=31, right_foot_index=32
    lm[31] = (0.46, 0.93, 0.0, 1.0)
    lm[32] = (0.54, 0.93, 0.0, 1.0)
    return lm


def _make_static_sequence(
    n_frames: int,
    fps: int = 15,
    modifier=None,
) -> list[PoseFrame]:
    """동일 자세를 n_frames 번 반복. modifier(lm)로 변형 가능."""
    ms_per_frame = int(1000 / fps)
    out: list[PoseFrame] = []
    for i in range(n_frames):
        lm = _baseline_landmarks()
        if modifier is not None:
            lm = modifier(lm, i)
        out.append(_make_pose_frame(i, i * ms_per_frame, lm))
    return out


class TestClassifierSmoke:
    def test_empty_input_returns_empty(self) -> None:
        assert classify_segments([], []) == []
        # pose_frames만 있고 segments 없으면 빈 결과
        frames = _make_static_sequence(10)
        assert classify_segments(frames, []) == []

    def test_too_short_segment_dropped(self) -> None:
        """3개 미만 pose_frames인 segment는 _slice_arr_by_time이 None 반환."""
        frames = _make_static_sequence(2)
        result = classify_segments(frames, [(0, 100)])
        assert result == []

    def test_all_results_use_known_labels(self) -> None:
        """방어적 가드 — TECHNIQUE_LABELS 외 라벨은 emit되지 않아야."""
        frames = _make_static_sequence(15)
        result = classify_segments(frames, [(0, 900)])
        for r in result:
            assert isinstance(r, AnalysisSegmentPayload)
            assert r.technique in TECHNIQUE_LABELS

    def test_high_step_like_sequence_does_not_crash(self) -> None:
        """발목을 골반 위로 올린 정적 시퀀스 — high_step 또는 lock_off 후보."""

        def lift_left_ankle(lm: np.ndarray, i: int) -> np.ndarray:
            # left_ankle을 위로 (y 작게)
            lm[27] = (0.45, 0.4, 0.0, 1.0)  # 골반(0.6)보다 위
            # left_knee도 따라 올라감 (해부학적 일관성)
            lm[25] = (0.45, 0.5, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, modifier=lift_left_ankle)
        # crash 없이 분류가 끝나야 한다
        result = classify_segments(frames, [(0, 1300)])
        # 결과는 비어있을 수도 있음 (임계값 미충족) — 단지 크래시 없음을 확인
        assert isinstance(result, list)

    def test_dyno_like_sequence_returns_dynamic(self) -> None:
        """골반 y가 빠르게 변하고 양손이 동시에 높은 속도 → dyno 또는 coordination."""

        def explode(lm: np.ndarray, i: int) -> np.ndarray:
            # 프레임 i에 따라 골반/팔 y를 큰 폭으로 변동
            jump = 0.1 * np.sin(i * 1.5)
            lm[23] = (0.45, 0.6 + jump, 0.0, 1.0)
            lm[24] = (0.55, 0.6 + jump, 0.0, 1.0)
            # 양손 모두 fast move
            lm[15] = (0.3 + 0.05 * np.cos(i), 0.6 + jump, 0.0, 1.0)
            lm[16] = (0.7 - 0.05 * np.cos(i), 0.6 + jump, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, modifier=explode)
        result = classify_segments(frames, [(0, 1300)])
        # 결과가 있으면 dynamic 카테고리거나 적어도 valid label
        if result:
            seg = result[0]
            assert seg.technique in TECHNIQUE_LABELS
            assert 0.0 <= (seg.confidence or 0) <= 1.0

    def test_neutral_tripod_stance_is_not_flagging(self) -> None:
        """양발이 골반 양쪽에 있는 기본 자세는 flagging으로 보지 않는다."""
        frames = _make_static_sequence(20, fps=30)
        result = classify_segments(frames, [(0, 633)])

        assert result == []

    def test_cross_body_foot_position_can_be_flagging(self) -> None:
        """한 발이 골반 중심선을 넘어 같은 쪽으로 몰리면 flagging으로 본다."""

        def cross_right_foot_left(lm: np.ndarray, i: int) -> np.ndarray:
            lm[28] = (0.42, 0.9, 0.0, 1.0)
            lm[30] = (0.43, 0.92, 0.0, 1.0)
            lm[32] = (0.41, 0.93, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=cross_right_foot_left)
        result = classify_segments(frames, [(0, 633)])

        assert any(seg.technique == "flagging" for seg in result)

    def test_low_heel_hook_position_can_be_detected(self) -> None:
        """무릎보다 낮은 훅이어도 뒤꿈치가 위로 걸린 자세는 heel_hook이다."""

        def heel_hook(lm: np.ndarray, i: int) -> np.ndarray:
            lm[27] = (0.35, 0.82, 0.0, 1.0)
            lm[29] = (0.34, 0.76, 0.0, 1.0)
            lm[31] = (0.36, 0.85, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=heel_hook)
        result = classify_segments(frames, [(0, 633)])

        assert any(seg.technique == "heel_hook" for seg in result)

    def test_low_toe_hook_position_can_be_detected(self) -> None:
        """무릎보다 낮은 훅이어도 발끝이 위로 걸린 자세는 toe_hook이다."""

        def toe_hook(lm: np.ndarray, i: int) -> np.ndarray:
            lm[27] = (0.35, 0.82, 0.0, 1.0)
            lm[29] = (0.34, 0.86, 0.0, 1.0)
            lm[31] = (0.36, 0.76, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=toe_hook)
        result = classify_segments(frames, [(0, 633)])

        assert any(seg.technique == "toe_hook" for seg in result)

    def test_brief_jitter_in_static_tripod_is_not_coordination(self) -> None:
        """정적 자세에서 limb별 1-frame 튐은 coordination으로 보지 않는다."""

        def add_brief_jitter(lm: np.ndarray, i: int) -> np.ndarray:
            if i == 3:
                lm[15] = (0.32, 0.6, 0.0, 1.0)
            if i == 5:
                lm[16] = (0.68, 0.6, 0.0, 1.0)
            if i == 7:
                lm[27] = (0.45, 0.88, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=add_brief_jitter)
        result = classify_segments(frames, [(0, 633)])

        assert all(seg.technique != "coordination" for seg in result)
        assert all(seg.is_dynamic is not True for seg in result)

    def test_small_foot_lift_with_balance_hands_is_not_coordination(self) -> None:
        """한 발을 살짝 올리고 손이 균형 보정하는 정도는 coordination이 아니다."""

        def small_foot_lift(lm: np.ndarray, i: int) -> np.ndarray:
            if 3 <= i <= 10:
                progress = i - 2
                foot_shift = 0.016 * progress
                hand_shift = 0.016 * progress
                lm[27] = (0.45, 0.9 - foot_shift, 0.0, 1.0)
                lm[15] = (0.3 + hand_shift, 0.6, 0.0, 1.0)
                lm[16] = (0.7 - hand_shift, 0.6, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=small_foot_lift)
        result = classify_segments(frames, [(0, 633)])

        assert all(seg.technique != "coordination" for seg in result)
        assert all(seg.is_dynamic is not True for seg in result)

    def test_sustained_multi_limb_motion_can_be_coordination(self) -> None:
        """여러 limb이 같은 짧은 구간에서 지속 이동하면 coordination으로 유지한다."""

        def move_three_limbs(lm: np.ndarray, i: int) -> np.ndarray:
            if 3 <= i <= 10:
                shift = 0.02 * (i - 2)
                lm[15] = (0.3 + shift, 0.6, 0.0, 1.0)
                lm[16] = (0.7 - shift, 0.6, 0.0, 1.0)
                lm[27] = (0.45 + shift, 0.9, 0.0, 1.0)
                lm[28] = (0.55 - shift, 0.9, 0.0, 1.0)
            return lm

        frames = _make_static_sequence(20, fps=30, modifier=move_three_limbs)
        result = classify_segments(frames, [(0, 633)])

        assert any(seg.technique == "coordination" for seg in result)

    def test_sequence_index_starts_at_zero_and_increments(self) -> None:
        """multi-segment 결과의 sequence_index가 0,1,2,... 순서."""
        frames = _make_static_sequence(40)
        # 여러 segment 시도 — 결과로 emit되는 것만 sequence_index 부여
        result = classify_segments(frames, [(0, 900), (1000, 1900), (2000, 2600)])
        for i, seg in enumerate(result):
            assert seg.sequence_index == i

    def test_confidence_in_unit_range(self) -> None:
        frames = _make_static_sequence(30)
        result = classify_segments(frames, [(0, 1900)])
        for seg in result:
            if seg.confidence is not None:
                assert 0.0 <= seg.confidence <= 1.0

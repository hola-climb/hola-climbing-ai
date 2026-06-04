"""MediaPipe Pose landmark extraction.

vision-engineer 구현 영역.

설계 메모:
- MediaPipe import는 lazy. 모듈 import 시 모델 다운로드를 유발하지 않는다.
- CPU 추론. Apple Silicon에서도 동작 (model_complexity=1 권장).
- 입력 프레임은 BGR np.ndarray (OpenCV 기본). 내부에서 RGB로 변환.
- 미검출 프레임은 결과에서 제외. 전체가 미검출이면 AnalysisException 발생.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np

from app.core.errors import AnalysisException, AnalysisFailureReason

POSE_LANDMARK_COUNT: Final[int] = 33


@dataclass(frozen=True)
class PoseFrame:
    """단일 프레임의 pose landmark 결과.

    landmarks: shape (33, 4) — (x, y, z, visibility) per MediaPipe Pose.
      - x, y: image-normalized 좌표 [0, 1]. y는 위→아래.
      - z: 카메라 깊이 (상대값, 단위 없음).
      - visibility: [0, 1].
    """

    frame_idx: int
    timestamp_ms: int
    landmarks: np.ndarray  # shape (33, 4), float32


def _build_pose_estimator(
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
):
    """MediaPipe Pose 인스턴스를 lazy import 후 생성한다.

    분리 이유: 테스트에서 mock 주입을 쉽게 하기 위해.
    """
    import mediapipe as mp  # noqa: PLC0415 — lazy import 의도

    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_pose_landmarks(
    frames: Iterable[tuple[int, int, np.ndarray]],
    *,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[PoseFrame]:
    """프레임 iterator로부터 MediaPipe Pose landmark를 추출한다.

    Args:
        frames: (frame_idx, timestamp_ms, BGR np.ndarray) iterator.
            pipeline-engineer의 ``services.pipeline.frames.iter_frames`` 출력 형식.
        model_complexity: MediaPipe 모델 복잡도 (0=light, 1=full, 2=heavy).
        min_detection_confidence: 첫 검출 신뢰도 임계.
        min_tracking_confidence: 트래킹 신뢰도 임계.

    Returns:
        프레임별 PoseFrame 리스트 (시간순). landmark 미검출 프레임은 제외.

    Raises:
        AnalysisException(POSE_NOT_DETECTED): 모든 프레임에서 pose 미검출.
    """
    pose = _build_pose_estimator(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    results: list[PoseFrame] = []
    total_frames = 0
    try:
        for frame_idx, timestamp_ms, bgr in frames:
            total_frames += 1
            if bgr is None or bgr.size == 0:
                continue
            # BGR → RGB
            rgb = bgr[..., ::-1]
            # MediaPipe는 contiguous array 요구
            rgb = np.ascontiguousarray(rgb)
            res = pose.process(rgb)
            if res.pose_landmarks is None:
                continue
            arr = np.empty((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
            for i, lm in enumerate(res.pose_landmarks.landmark):
                arr[i, 0] = lm.x
                arr[i, 1] = lm.y
                arr[i, 2] = lm.z
                arr[i, 3] = lm.visibility
            results.append(
                PoseFrame(
                    frame_idx=int(frame_idx),
                    timestamp_ms=int(timestamp_ms),
                    landmarks=arr,
                )
            )
    finally:
        # MediaPipe Pose는 context manager지만 명시적 close도 안전
        try:
            pose.close()
        except Exception:  # noqa: BLE001 — close 실패는 무시 (best-effort)
            pass

    if not results:
        raise AnalysisException(
            reason=AnalysisFailureReason.POSE_NOT_DETECTED,
            message=(
                f"전체 {total_frames} 프레임에서 pose가 한 번도 검출되지 않았습니다."
            ),
        )

    return results

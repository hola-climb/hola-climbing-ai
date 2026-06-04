"""OpenCV frame extraction — video file → frame iterator.

pipeline-engineer 구현 영역.

샘플링 전략:
  원본 fps에서 target_fps로 다운샘플. step = max(1, round(src_fps / target_fps)).
  예: src=30fps, target=15 → step=2 (매 2번째 프레임).
  src_fps 미지 시(`cap.get(CAP_PROP_FPS) == 0` 또는 너무 큰 값) step=1로 전부 yield.

timestamp_ms:
  cv2.CAP_PROP_POS_MSEC가 부정확할 수 있으므로 `frame_idx * 1000 / src_fps`로 계산.
  src_fps 미지 시 frame_idx만 사용 (timestamp_ms = -1로 표시 가능하나, 본 구현은 0 fallback).
"""

from __future__ import annotations

import logging
from typing import Iterator

import cv2
import numpy as np

from app.core.errors import AnalysisException, AnalysisFailureReason

logger = logging.getLogger(__name__)


def iter_frames(video_path: str, target_fps: int = 15) -> Iterator[tuple[int, int, np.ndarray]]:
    """비디오 파일에서 프레임을 (frame_idx, timestamp_ms, BGR ndarray)로 yield한다.

    Args:
        video_path: 로컬 파일 경로 (GCS 다운로드 후).
        target_fps: 다운샘플링 fps. 원본이 30fps고 target=15면 매 2번째 프레임만.

    Yields:
        (frame_idx, timestamp_ms, np.ndarray of shape (H, W, 3) BGR uint8).
        frame_idx는 **원본 영상 기준 절대 인덱스** (다운샘플 후에도 0,2,4,...).

    Raises:
        AnalysisException(VIDEO_DECODE): cv2.VideoCapture가 열리지 않거나 첫 프레임 읽기 실패.
    """
    if target_fps < 1:
        raise ValueError(f"target_fps must be >= 1, got {target_fps}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise AnalysisException(
            AnalysisFailureReason.VIDEO_DECODE,
            f"cv2.VideoCapture cannot open {video_path}",
        )
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if src_fps <= 0 or src_fps > 240:
            # 알 수 없거나 비현실적인 fps — step=1, timestamp는 frame_idx 기반 fallback
            step = 1
            ms_per_frame = 0.0
        else:
            step = max(1, round(src_fps / target_fps))
            ms_per_frame = 1000.0 / src_fps

        logger.info(
            "iter_frames open",
            extra={
                "video_path": video_path,
                "src_fps": src_fps,
                "target_fps": target_fps,
                "step": step,
            },
        )

        idx = 0
        yielded = 0
        first_read_ok: bool | None = None
        while True:
            ok, frame = cap.read()
            if not ok:
                # 첫 프레임부터 실패면 디코드 에러
                if first_read_ok is None:
                    raise AnalysisException(
                        AnalysisFailureReason.VIDEO_DECODE,
                        f"failed to read first frame from {video_path}",
                    )
                break
            first_read_ok = True
            if idx % step == 0:
                ts_ms = int(idx * ms_per_frame) if ms_per_frame > 0 else 0
                yield idx, ts_ms, frame
                yielded += 1
            idx += 1

        logger.info(
            "iter_frames done",
            extra={
                "video_path": video_path,
                "total_frames_read": idx,
                "yielded": yielded,
            },
        )
    finally:
        cap.release()

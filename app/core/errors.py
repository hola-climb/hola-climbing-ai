"""Error types — worker-internal failure reasons + Spring ErrorCode constants.

Spring 서버(hola-climbing-server)의 `ErrorCode` enum과 호환되는 상수만 정의한다.
워커는 자체 ErrorCode를 콜백 body에 절대 넣지 않는다 — 실패는 `status="failed"` 한 방식만.
본 enum은 워커 내부 로깅/메트릭/dead-letter 분류 용도.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class AnalysisFailureReason(str, Enum):
    """Worker-internal failure reason. 콜백 body에 들어가지 않는다."""

    VIDEO_DOWNLOAD = "video_download"
    VIDEO_DECODE = "video_decode"
    POSE_NOT_DETECTED = "pose_not_detected"
    CALLBACK_FAILED = "callback_failed"
    INTERNAL = "internal"


class AnalysisException(Exception):
    """워커 내부 분석 실패. orchestrator가 캐치 후 status='failed' 콜백 발송."""

    def __init__(self, reason: AnalysisFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


# --- Spring ErrorCode 상수 (워커가 수신/인식할 수 있는 코드) ---
# 출처: hola-climbing-server `common/exception/error/ErrorCode.java`
SPRING_INVALID_INPUT: Final[str] = "C001"
SPRING_VIDEO_NOT_FOUND: Final[str] = "V001"
SPRING_ANALYSIS_FAILED: Final[str] = "V005"
SPRING_GCS_UPLOAD_FAILED: Final[str] = "S001"
SPRING_AI_SERVER_UNAVAILABLE: Final[str] = "S002"

# 콜백 응답에서 받았을 때 재시도 무의미한 코드 (즉시 dead-letter)
NON_RETRYABLE_SPRING_CODES: Final[frozenset[str]] = frozenset(
    {SPRING_INVALID_INPUT, SPRING_VIDEO_NOT_FOUND}
)

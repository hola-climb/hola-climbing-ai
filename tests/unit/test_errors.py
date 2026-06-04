"""Spring ErrorCode 상수가 Java enum과 정확히 일치하는지 검증."""

from __future__ import annotations

from app.core.errors import (
    NON_RETRYABLE_SPRING_CODES,
    SPRING_AI_SERVER_UNAVAILABLE,
    SPRING_ANALYSIS_FAILED,
    SPRING_GCS_UPLOAD_FAILED,
    SPRING_INVALID_INPUT,
    SPRING_VIDEO_NOT_FOUND,
    AnalysisException,
    AnalysisFailureReason,
)


class TestSpringErrorCodeConstants:
    """Spring `common/exception/error/ErrorCode.java` 값과 정확히 일치해야."""

    def test_invalid_input_c001(self) -> None:
        assert SPRING_INVALID_INPUT == "C001"

    def test_video_not_found_v001(self) -> None:
        assert SPRING_VIDEO_NOT_FOUND == "V001"

    def test_analysis_failed_v005(self) -> None:
        assert SPRING_ANALYSIS_FAILED == "V005"

    def test_gcs_upload_failed_s001(self) -> None:
        assert SPRING_GCS_UPLOAD_FAILED == "S001"

    def test_ai_server_unavailable_s002(self) -> None:
        assert SPRING_AI_SERVER_UNAVAILABLE == "S002"

    def test_non_retryable_codes(self) -> None:
        """C001(계약 위반), V001(없는 video) — 재시도 무의미."""
        assert SPRING_INVALID_INPUT in NON_RETRYABLE_SPRING_CODES
        assert SPRING_VIDEO_NOT_FOUND in NON_RETRYABLE_SPRING_CODES
        # 재시도 가능한 코드는 포함되지 않아야 함
        assert SPRING_ANALYSIS_FAILED not in NON_RETRYABLE_SPRING_CODES
        assert SPRING_AI_SERVER_UNAVAILABLE not in NON_RETRYABLE_SPRING_CODES


class TestAnalysisFailureReason:
    def test_all_reasons_defined(self) -> None:
        """architect 설계 문서의 5가지 reason."""
        expected = {
            "video_download",
            "video_decode",
            "pose_not_detected",
            "callback_failed",
            "internal",
        }
        assert {r.value for r in AnalysisFailureReason} == expected


class TestAnalysisException:
    def test_carries_reason_and_message(self) -> None:
        exc = AnalysisException(
            AnalysisFailureReason.VIDEO_DOWNLOAD, "gcs object not found"
        )
        assert exc.reason == AnalysisFailureReason.VIDEO_DOWNLOAD
        assert exc.message == "gcs object not found"
        assert "gcs object not found" in str(exc)

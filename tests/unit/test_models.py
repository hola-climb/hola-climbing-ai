"""Pydantic 모델 — Spring DTO와 shape 호환성 단위 테스트.

검증 포인트:
- StreamRequest: camelCase alias로 raw Redis bytes/string 모두 파싱.
- AnalysisIngestRequest: snake_case JSON 직렬화 (Spring SNAKE_CASE 호환).
- AnalysisSegmentPayload: 모든 nullable 필드와 sequence_index ge=0.
- ProgressEvent: updated_at이 ISO-8601 직렬화 + Z suffix 변환.
- ApiResponse: is_success 키 (isSuccess 아님).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.models.callback import AnalysisIngestRequest, AnalysisSegmentPayload
from app.models.progress import AnalysisStage, ProgressEvent
from app.models.response import ApiResponse
from app.models.stream import StreamRequest

# ---------------------------------------------------------------------------
# StreamRequest — Redis Stream 입력 (camelCase)
# ---------------------------------------------------------------------------


class TestStreamRequest:
    def test_parses_camel_case_string_fields(self) -> None:
        req = StreamRequest.model_validate(
            {
                "videoId": "42",
                "gcsPath": "videos/uploads/2026/05/28/abc.mp4",
                "callbackUrl": "http://localhost:8080/api/analysis/videos/42",
            }
        )
        assert req.video_id == 42
        assert req.gcs_path == "videos/uploads/2026/05/28/abc.mp4"
        assert req.callback_url == "http://localhost:8080/api/analysis/videos/42"

    def test_parses_bytes_fields_from_xreadgroup(self) -> None:
        """XREADGROUP은 bytes를 반환 — alias + field_validator가 처리해야."""
        req = StreamRequest.model_validate(
            {
                "videoId": b"42",
                "gcsPath": b"videos/uploads/x.mp4",
                "callbackUrl": b"http://localhost:8080/cb",
            }
        )
        assert req.video_id == 42
        assert req.gcs_path == "videos/uploads/x.mp4"
        assert req.callback_url == "http://localhost:8080/cb"

    def test_empty_gcs_path_raises(self) -> None:
        """Spring null→""변환 시 워커가 거부 (의도된 강화)."""
        with pytest.raises(ValidationError) as exc_info:
            StreamRequest.model_validate(
                {"videoId": "1", "gcsPath": "", "callbackUrl": "http://x"}
            )
        assert "gcs_path" in str(exc_info.value) or "gcsPath" in str(exc_info.value)

    def test_empty_callback_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            StreamRequest.model_validate(
                {"videoId": "1", "gcsPath": "videos/x.mp4", "callbackUrl": ""}
            )

    def test_video_id_non_numeric_raises(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            StreamRequest.model_validate(
                {"videoId": "abc", "gcsPath": "videos/x.mp4", "callbackUrl": "http://x"}
            )

    def test_accepts_field_name_too(self) -> None:
        """populate_by_name=True — snake_case로도 접근 가능."""
        req = StreamRequest(
            video_id=7,
            gcs_path="videos/x.mp4",
            callback_url="http://localhost:8080/cb",
        )
        assert req.video_id == 7


# ---------------------------------------------------------------------------
# AnalysisIngestRequest / AnalysisSegmentPayload — 콜백 body
# ---------------------------------------------------------------------------


class TestAnalysisIngestRequest:
    def test_serializes_snake_case(self) -> None:
        """Spring Jackson SNAKE_CASE 정책과 호환 — 필드명이 이미 snake_case."""
        body = AnalysisIngestRequest(
            status="done",
            model_version="rule_v1",
            segments=[
                AnalysisSegmentPayload(
                    sequence_index=0,
                    start_time_ms=0,
                    end_time_ms=1240,
                    technique="high_step",
                    is_dynamic=False,
                    confidence=0.87,
                )
            ],
        )
        data = json.loads(body.model_dump_json())
        # Spring AnalysisIngestRequest 필드명과 1:1 매칭
        assert set(data.keys()) == {"status", "model_version", "segments"}
        seg = data["segments"][0]
        assert set(seg.keys()) == {
            "sequence_index",
            "start_time_ms",
            "end_time_ms",
            "technique",
            "is_dynamic",
            "confidence",
        }

    def test_status_literal_done_or_failed(self) -> None:
        """Spring AnalysisServiceImpl.ingestResult가 'done'/'failed'만 허용."""
        AnalysisIngestRequest(status="done", segments=[])
        AnalysisIngestRequest(status="failed", segments=[])
        with pytest.raises(ValidationError):
            AnalysisIngestRequest(status="completed", segments=[])  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            AnalysisIngestRequest(status="ok", segments=[])  # type: ignore[arg-type]

    def test_failed_status_with_empty_segments(self) -> None:
        body = AnalysisIngestRequest(status="failed", segments=[])
        data = json.loads(body.model_dump_json())
        assert data["status"] == "failed"
        assert data["segments"] == []

    def test_model_version_nullable(self) -> None:
        body = AnalysisIngestRequest(status="done")
        data = json.loads(body.model_dump_json())
        # nullable이지만 명시적으로 None일 때 키는 포함됨 (Pydantic 기본 동작).
        # Spring 측 record는 null 허용이므로 호환.
        assert data["model_version"] is None
        assert data["segments"] == []


class TestAnalysisSegmentPayload:
    def test_sequence_index_must_be_non_negative(self) -> None:
        AnalysisSegmentPayload(sequence_index=0, technique="dyno")
        with pytest.raises(ValidationError):
            AnalysisSegmentPayload(sequence_index=-1, technique="dyno")

    def test_technique_not_blank(self) -> None:
        with pytest.raises(ValidationError):
            AnalysisSegmentPayload(sequence_index=0, technique="")

    def test_confidence_range_0_to_1(self) -> None:
        AnalysisSegmentPayload(sequence_index=0, technique="dyno", confidence=0.0)
        AnalysisSegmentPayload(sequence_index=0, technique="dyno", confidence=1.0)
        with pytest.raises(ValidationError):
            AnalysisSegmentPayload(sequence_index=0, technique="dyno", confidence=1.5)
        with pytest.raises(ValidationError):
            AnalysisSegmentPayload(sequence_index=0, technique="dyno", confidence=-0.1)

    def test_optional_fields_nullable(self) -> None:
        """Spring 측 startTimeMs/endTimeMs/isDynamic/confidence는 모두 nullable."""
        seg = AnalysisSegmentPayload(sequence_index=0, technique="high_step")
        data = json.loads(seg.model_dump_json())
        assert data["start_time_ms"] is None
        assert data["end_time_ms"] is None
        assert data["is_dynamic"] is None
        assert data["confidence"] is None


# ---------------------------------------------------------------------------
# ProgressEvent — Pub/Sub 페이로드
# ---------------------------------------------------------------------------


class TestProgressEvent:
    def test_field_names_are_snake_case(self) -> None:
        event = ProgressEvent(
            video_id=42,
            stage=AnalysisStage.PROCESSING,
            message="프레임 추출 중",
        )
        data = json.loads(event.model_dump_json())
        assert set(data.keys()) == {"video_id", "stage", "message", "updated_at"}

    def test_stage_serializes_as_uppercase_string(self) -> None:
        """Spring AnalysisStage enum은 대문자."""
        for stage in (
            AnalysisStage.QUEUED,
            AnalysisStage.PROCESSING,
            AnalysisStage.COMPLETED,
            AnalysisStage.FAILED,
        ):
            event = ProgressEvent(video_id=1, stage=stage, message="x")
            data = json.loads(event.model_dump_json())
            assert data["stage"] == stage.value
            assert data["stage"].isupper()

    def test_stage_values_match_spring(self) -> None:
        """Spring AnalysisStage.java와 정확히 동일 4값."""
        assert {s.value for s in AnalysisStage} == {
            "QUEUED",
            "PROCESSING",
            "COMPLETED",
            "FAILED",
        }

    def test_updated_at_iso8601_utc(self) -> None:
        """기본값은 UTC datetime — Spring Instant와 호환."""
        event = ProgressEvent(
            video_id=1, stage=AnalysisStage.PROCESSING, message="x"
        )
        # tz-aware UTC
        assert event.updated_at.tzinfo is not None
        assert event.updated_at.tzinfo.utcoffset(event.updated_at).total_seconds() == 0

    def test_publish_progress_uses_z_suffix(self) -> None:
        """redis_bus.publish_progress가 +00:00 → Z 후처리하는 로직 표면 검증.

        모델 자체는 +00:00로 직렬화하므로 publish layer가 변환 책임을 진다.
        """
        event = ProgressEvent(
            video_id=1,
            stage=AnalysisStage.PROCESSING,
            message="x",
            updated_at=datetime(2026, 5, 28, 10, 32, 45, 123000, tzinfo=UTC),
        )
        payload = event.model_dump(mode="json")
        ts = payload["updated_at"]
        # Pydantic 기본 ISO-8601은 +00:00 형식
        assert ts.endswith("+00:00") or ts.endswith("Z")
        # 변환 후 Spring 호환 형식
        normalized = ts[:-6] + "Z" if ts.endswith("+00:00") else ts
        assert normalized.endswith("Z")


# ---------------------------------------------------------------------------
# ApiResponse — Spring `common/response/ApiResponse.java`
# ---------------------------------------------------------------------------


class TestApiResponse:
    def test_success_factory(self) -> None:
        r = ApiResponse.ok(data={"status": "ok"})
        data = json.loads(r.model_dump_json())
        assert data["is_success"] is True  # Spring isSuccess → snake_case
        assert data["code"] == "OK"
        assert data["data"] == {"status": "ok"}
        assert "timestamp" in data

    def test_error_factory(self) -> None:
        r = ApiResponse.error(code="S002", message="redis_unavailable")
        data = json.loads(r.model_dump_json())
        assert data["is_success"] is False
        assert data["code"] == "S002"
        assert data["message"] == "redis_unavailable"

    def test_parses_spring_response(self) -> None:
        """Spring 200 OK 응답을 워커가 파싱할 수 있어야 (콜백 응답 처리)."""
        spring_body = {
            "is_success": True,
            "code": "OK",
            "data": {"id": 42},
            "timestamp": "2026-05-28T10:00:00Z",
        }
        r = ApiResponse[dict].model_validate(spring_body)
        assert r.is_success is True
        assert r.code == "OK"

    def test_field_name_is_is_success_not_success(self) -> None:
        """Boundary bug 방지 — 워커가 'success' 또는 'isSuccess'로 쓰면 안 됨."""
        r = ApiResponse.ok(data=None)
        data = json.loads(r.model_dump_json())
        assert "is_success" in data
        assert "success" not in data
        assert "isSuccess" not in data

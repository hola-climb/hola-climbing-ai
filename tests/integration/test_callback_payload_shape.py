"""콜백 body가 실제 Spring AnalysisIngestRequest와 호환되는지 검증.

respx로 실제 HTTP POST를 가로채 wire-level body를 캡처한 뒤,
Spring DTO의 필수/허용 필드 set과 일치하는지 강제.
"""

from __future__ import annotations

import json

import pytest

respx = pytest.importorskip("respx")
import httpx  # noqa: E402

from app.models.callback import AnalysisIngestRequest, AnalysisSegmentPayload  # noqa: E402
from app.services.callback.client import post_callback  # noqa: E402


URL = "http://test-spring/api/analysis/videos/42"

# Spring AnalysisIngestRequest record 필드 (Jackson SNAKE_CASE 변환 후)
SPRING_INGEST_FIELDS = {"status", "model_version", "segments"}
SPRING_SEGMENT_FIELDS = {
    "sequence_index",
    "start_time_ms",
    "end_time_ms",
    "technique",
    "is_dynamic",
    "confidence",
}


@respx.mock
async def test_done_callback_matches_spring_shape() -> None:
    """status=done, segments=[1개]가 Spring 측 record 필드와 1:1."""
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200, json={"is_success": True, "code": "OK", "data": {"id": 42}}
        )

    respx.post(URL).mock(side_effect=_capture)

    body = AnalysisIngestRequest(
        status="done",
        model_version="rule_v1",
        segments=[
            AnalysisSegmentPayload(
                sequence_index=0,
                start_time_ms=0,
                end_time_ms=1240,
                technique="dyno",
                is_dynamic=True,
                confidence=0.92,
            ),
            AnalysisSegmentPayload(
                sequence_index=1,
                start_time_ms=1240,
                end_time_ms=2500,
                technique="lock_off",
                is_dynamic=False,
                confidence=0.6,
            ),
        ],
    )
    await post_callback(URL, body)

    sent = captured["body"]
    # 1) 최상위 필드 set
    assert set(sent.keys()) == SPRING_INGEST_FIELDS

    # 2) status는 "done"|"failed"만
    assert sent["status"] in {"done", "failed"}

    # 3) segments[]의 각 필드 set
    for seg in sent["segments"]:
        assert set(seg.keys()) == SPRING_SEGMENT_FIELDS
        # snake_case 키 명시 검증
        assert "sequence_index" in seg
        assert "start_time_ms" in seg
        assert "end_time_ms" in seg
        # NOT camelCase
        assert "sequenceIndex" not in seg
        assert "startTimeMs" not in seg
        assert "endTimeMs" not in seg
        assert "isDynamic" not in seg


@respx.mock
async def test_failed_callback_shape() -> None:
    """status=failed면 segments는 빈 리스트, model_version 보존."""
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"is_success": True})

    respx.post(URL).mock(side_effect=_capture)

    body = AnalysisIngestRequest(
        status="failed", model_version="rule_v1", segments=[]
    )
    await post_callback(URL, body)

    sent = captured["body"]
    assert sent["status"] == "failed"
    assert sent["segments"] == []
    assert sent["model_version"] == "rule_v1"


@respx.mock
async def test_technique_values_in_known_set() -> None:
    """워커가 정의한 6+1 라벨만 사용 (Spring은 자유 문자열이라 강제 안 함, 워커가 보수)."""
    from app.services.vision.classifier import TECHNIQUE_LABELS

    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"is_success": True})

    respx.post(URL).mock(side_effect=_capture)

    body = AnalysisIngestRequest(
        status="done",
        model_version="rule_v1",
        segments=[
            AnalysisSegmentPayload(
                sequence_index=0, technique="high_step", confidence=0.5
            )
        ],
    )
    await post_callback(URL, body)
    sent = captured["body"]
    assert sent["segments"][0]["technique"] in TECHNIQUE_LABELS


@respx.mock
async def test_content_type_header_is_json() -> None:
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"is_success": True})

    respx.post(URL).mock(side_effect=_capture)
    await post_callback(URL, AnalysisIngestRequest(status="failed", segments=[]))
    assert captured["headers"].get("content-type") == "application/json"

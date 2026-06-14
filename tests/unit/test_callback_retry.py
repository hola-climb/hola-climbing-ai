"""콜백 클라이언트 재시도 정책 단위 테스트.

검증:
- 200: 즉시 성공
- 200 + is_success=false: 200으로 처리 (재시도 안 함), warning 로그
- 5xx: tenacity 지수 백오프 재시도, max=CALLBACK_MAX_RETRIES
- 429: 5xx와 동일 재시도
- 4xx: 즉시 raise (AnalysisException CALLBACK_FAILED)
- network error: 5xx와 동일 재시도
"""

from __future__ import annotations

import pytest

# respx가 없으면 스킵
respx = pytest.importorskip("respx")
import httpx  # noqa: E402

from app.core.errors import AnalysisException, AnalysisFailureReason  # noqa: E402
from app.models.callback import AnalysisIngestRequest, AnalysisSegmentPayload  # noqa: E402
from app.services.callback.client import post_callback  # noqa: E402


@pytest.fixture
def callback_body() -> AnalysisIngestRequest:
    return AnalysisIngestRequest(
        status="done",
        model_version="rule_v1",
        segments=[
            AnalysisSegmentPayload(
                sequence_index=0,
                start_time_ms=0,
                end_time_ms=1000,
                technique="high_step",
                is_dynamic=False,
                confidence=0.8,
            )
        ],
    )


URL = "http://test-spring/api/analysis/videos/42"


@respx.mock
async def test_200_immediate_success(callback_body: AnalysisIngestRequest) -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={"is_success": True, "code": "OK", "data": None, "timestamp": "2026-05-28T10:00:00Z"},
        )
    )
    await post_callback(URL, callback_body)
    assert route.call_count == 1


@respx.mock
async def test_200_with_is_success_false_does_not_retry(
    callback_body: AnalysisIngestRequest,
) -> None:
    """is_success=false여도 HTTP 200이면 재시도 안 함 (계약 §3.1: status code만 판정)."""
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "is_success": False,
                "code": "V001",
                "message": "video not found",
                "timestamp": "2026-05-28T10:00:00Z",
            },
        )
    )
    await post_callback(URL, callback_body)
    assert route.call_count == 1


@respx.mock
async def test_500_retries_then_raises(callback_body: AnalysisIngestRequest) -> None:
    """5xx 응답을 max_retries 만큼 시도 후 AnalysisException(CALLBACK_FAILED)."""
    route = respx.post(URL).mock(
        return_value=httpx.Response(500, text="internal server error")
    )
    with pytest.raises(AnalysisException) as exc_info:
        await post_callback(URL, callback_body)
    assert exc_info.value.reason == AnalysisFailureReason.CALLBACK_FAILED
    # config: CALLBACK_MAX_RETRIES=3
    assert route.call_count == 3


@respx.mock
async def test_500_then_200_succeeds(callback_body: AnalysisIngestRequest) -> None:
    """첫 시도 500, 두번째 200 → 성공."""
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(200, json={"is_success": True, "code": "OK"}),
        ]
    )
    await post_callback(URL, callback_body)
    assert route.call_count == 2


@respx.mock
async def test_429_is_treated_as_transient(callback_body: AnalysisIngestRequest) -> None:
    """Rate-limit 429도 재시도 대상."""
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, text="rate limit"),
            httpx.Response(200, json={"is_success": True}),
        ]
    )
    await post_callback(URL, callback_body)
    assert route.call_count == 2


@respx.mock
async def test_404_immediate_raise(callback_body: AnalysisIngestRequest) -> None:
    """V001(VIDEO_NOT_FOUND)이면 Spring이 404 응답 — 재시도 무의미."""
    route = respx.post(URL).mock(return_value=httpx.Response(404, text="not found"))
    with pytest.raises(AnalysisException) as exc_info:
        await post_callback(URL, callback_body)
    assert exc_info.value.reason == AnalysisFailureReason.CALLBACK_FAILED
    assert route.call_count == 1  # no retry


@respx.mock
async def test_400_immediate_raise(callback_body: AnalysisIngestRequest) -> None:
    """C001(INVALID_INPUT)이면 Spring이 400 — 계약 위반, 재시도 무의미."""
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad request"))
    with pytest.raises(AnalysisException):
        await post_callback(URL, callback_body)
    assert route.call_count == 1


@respx.mock
async def test_network_error_retries(callback_body: AnalysisIngestRequest) -> None:
    """ConnectError를 5xx와 동일하게 재시도."""
    route = respx.post(URL).mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(AnalysisException) as exc_info:
        await post_callback(URL, callback_body)
    assert exc_info.value.reason == AnalysisFailureReason.CALLBACK_FAILED
    assert route.call_count == 3


@respx.mock
async def test_callback_body_serialization_is_snake_case(
    callback_body: AnalysisIngestRequest,
) -> None:
    """실제로 보낸 body가 Spring AnalysisIngestRequest와 호환 (snake_case)."""
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured["body"] = _json.loads(request.content.decode())
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"is_success": True})

    respx.post(URL).mock(side_effect=_capture)
    await post_callback(URL, callback_body)

    body = captured["body"]
    assert body["status"] == "done"
    assert body["model_version"] == "rule_v1"
    assert "segments" in body
    assert body["techniques"] == ["high_step"]
    assert body["is_dynamic"] is None
    assert body["dynamic_probability"] is None
    seg = body["segments"][0]
    assert seg["sequence_index"] == 0
    assert seg["start_time_ms"] == 0
    assert seg["end_time_ms"] == 1000
    assert seg["technique"] == "high_step"
    assert seg["is_dynamic"] is False
    assert seg["confidence"] == 0.8

    assert captured["headers"].get("content-type") == "application/json"
    assert "x-ai-callback-secret" not in captured["headers"]


@respx.mock
async def test_callback_secret_header_is_sent_when_configured(
    callback_body: AnalysisIngestRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spring AiCallbackSecretFilter와 맞추기 위해 공유 시크릿 헤더를 보낸다."""
    from app.core.config import get_settings

    monkeypatch.setenv("AI_CALLBACK_SECRET", "secret-for-test")
    get_settings.cache_clear()
    captured: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"is_success": True})

    respx.post(URL).mock(side_effect=_capture)
    await post_callback(URL, callback_body)

    assert captured["headers"].get("x-ai-callback-secret") == "secret-for-test"
    get_settings.cache_clear()

"""Spring 콜백 HTTP 클라이언트 — tenacity 지수 백오프.

pipeline-engineer 구현 영역.

재시도 정책 (architect 계약, `_workspace/01_architect_contract.md` §3.1):
  - 5xx: 지수 백오프, max CALLBACK_MAX_RETRIES회
  - 4xx: 즉시 raise (계약 위반 — dead-letter)
  - 네트워크 에러 (timeout, connection refused): 5xx와 동일 재시도
  - 응답 body는 무시. HTTP 상태 코드만 확인.

추가:
  - 응답 body가 ApiResponse 형태이면 파싱해서 `code` 로깅 (재시도 판단은 status code 우선).
  - non-retryable Spring code(V001, C001)는 200이 아니라 4xx로 회신될 것이므로 자연스럽게 raise됨.
"""

from __future__ import annotations

import json
import logging

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.errors import AnalysisException, AnalysisFailureReason
from app.models.callback import AnalysisIngestRequest

logger = logging.getLogger(__name__)


class _TransientCallbackError(Exception):
    """5xx 또는 네트워크 에러. tenacity가 재시도."""


class _PermanentCallbackError(Exception):
    """4xx. 즉시 dead-letter."""


async def _post_once(
    callback_url: str,
    body_json: str,
    timeout: float,
    callback_secret: str = "",
) -> None:
    """단일 POST. transient/permanent 예외로 변환."""
    headers = {"Content-Type": "application/json"}
    if callback_secret:
        headers["X-AI-Callback-Secret"] = callback_secret
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                callback_url,
                content=body_json,
                headers=headers,
            )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise _TransientCallbackError(f"network: {exc!r}") from exc

    status = resp.status_code
    if 200 <= status < 300:
        # 응답 body가 ApiResponse 형태이면 is_success 로깅 (재시도 판단 아님)
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("is_success") is False:
                logger.warning(
                    "callback 2xx but is_success=false",
                    extra={"code": data.get("code"), "api_message": data.get("message")},
                )
        except (ValueError, json.JSONDecodeError):
            pass
        return
    if status >= 500 or status == 429:
        raise _TransientCallbackError(f"http {status}: {resp.text[:200]}")
    # 4xx
    raise _PermanentCallbackError(f"http {status}: {resp.text[:200]}")


async def post_callback(callback_url: str, body: AnalysisIngestRequest) -> None:
    """Spring 콜백 엔드포인트 호출 (멱등).

    Args:
        callback_url: Stream 메시지의 callbackUrl 그대로. 워커가 path 조립 금지.
        body: AnalysisIngestRequest (snake_case JSON으로 직렬화).

    Raises:
        AnalysisException(CALLBACK_FAILED): 4xx 또는 최대 재시도 소진.
    """
    s = get_settings()
    body_json = body.model_dump_json()
    logger.info(
        "callback.post start",
        extra={"url": callback_url, "status": body.status, "segments": len(body.segments)},
    )

    retrying = AsyncRetrying(
        stop=stop_after_attempt(s.callback_max_retries),
        wait=wait_exponential(
            multiplier=s.callback_retry_initial_seconds,
            min=s.callback_retry_initial_seconds,
            max=30.0,
        ),
        retry=retry_if_exception_type(_TransientCallbackError),
        reraise=True,
    )

    try:
        async for attempt in retrying:
            with attempt:
                await _post_once(
                    callback_url,
                    body_json,
                    s.callback_timeout_seconds,
                    s.ai_callback_secret,
                )
    except _PermanentCallbackError as exc:
        logger.error(
            "callback.post permanent failure",
            extra={"url": callback_url, "error": str(exc)},
        )
        raise AnalysisException(
            AnalysisFailureReason.CALLBACK_FAILED,
            f"callback 4xx: {exc}",
        ) from exc
    except _TransientCallbackError as exc:
        logger.error(
            "callback.post retries exhausted",
            extra={"url": callback_url, "error": str(exc)},
        )
        raise AnalysisException(
            AnalysisFailureReason.CALLBACK_FAILED,
            f"callback retries exhausted: {exc}",
        ) from exc
    except RetryError as exc:
        raise AnalysisException(
            AnalysisFailureReason.CALLBACK_FAILED,
            f"callback retry error: {exc}",
        ) from exc

    logger.info("callback.post done", extra={"url": callback_url})

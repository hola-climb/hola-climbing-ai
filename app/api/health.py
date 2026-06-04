"""Health endpoints — liveness and readiness.

`/health`     : 프로세스가 살아있는지 (200 무조건).
`/health/ready`: Redis/GCS 연결 가능 여부 (200 또는 503).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.core.errors import SPRING_AI_SERVER_UNAVAILABLE
from app.infra.gcs import can_access_bucket
from app.infra.redis_bus import ping as redis_ping
from app.models.response import ApiResponse

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


@router.get("/health", response_model=ApiResponse[dict[str, object]])
async def health() -> ApiResponse[dict[str, object]]:
    """Liveness probe — 항상 200. uvicorn이 살아 응답할 수만 있으면 OK."""
    uptime = int(time.monotonic() - _START_TIME)
    return ApiResponse.ok({"status": "ok", "uptime_seconds": uptime})


@router.get("/health/ready", response_model=ApiResponse[dict[str, str]])
async def health_ready(response: Response) -> ApiResponse[dict[str, str]]:
    """Readiness probe — Redis/GCS 의존 시스템 연결 확인.

    Redis와 GCS bucket 접근이 모두 가능해야 ready로 본다.
    """
    settings = get_settings()
    redis_ok = await redis_ping()
    gcs_ok = await can_access_bucket(settings.gcs_bucket)
    data = {
        "redis": "ok" if redis_ok else "unavailable",
        "gcs": "ok" if gcs_ok else "unavailable",
    }
    if redis_ok and gcs_ok:
        return ApiResponse.ok(data)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ApiResponse(
        is_success=False,
        code=SPRING_AI_SERVER_UNAVAILABLE,
        message="readiness check failed",
        data=data,
    )

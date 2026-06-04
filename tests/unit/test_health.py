"""Health endpoint behavior."""

from __future__ import annotations

from fastapi import Response

from app.api import health as health_module


async def test_health_ready_returns_ok_when_dependencies_are_available(
    monkeypatch,
) -> None:
    """Redis와 GCS가 모두 가능하면 ready OK."""

    async def _redis_ok() -> bool:
        return True

    async def _gcs_ok(_bucket: str) -> bool:
        return True

    monkeypatch.setattr(health_module, "redis_ping", _redis_ok)
    monkeypatch.setattr(health_module, "can_access_bucket", _gcs_ok)

    response = Response()
    body = await health_module.health_ready(response)

    assert response.status_code == 200
    assert body.is_success is True
    assert body.data == {"redis": "ok", "gcs": "ok"}


async def test_health_ready_returns_503_when_dependency_is_unavailable(
    monkeypatch,
) -> None:
    """하나라도 실패하면 503 + 실패 상태를 반환."""

    async def _redis_fail() -> bool:
        return False

    async def _gcs_ok(_bucket: str) -> bool:
        return True

    monkeypatch.setattr(health_module, "redis_ping", _redis_fail)
    monkeypatch.setattr(health_module, "can_access_bucket", _gcs_ok)

    response = Response()
    body = await health_module.health_ready(response)

    assert response.status_code == 503
    assert body.is_success is False
    assert body.code == "S002"
    assert body.data == {"redis": "unavailable", "gcs": "ok"}

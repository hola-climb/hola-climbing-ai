"""Redis bus helper behavior."""

from __future__ import annotations

from typing import Any

import pytest


class _FakeRedis:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def xautoclaim(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


@pytest.mark.asyncio
async def test_xautoclaim_pending_normalizes_pending_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """XAUTOCLAIM 응답도 xreadgroup과 같은 msg_id + string-key fields로 정규화한다."""
    from app.infra import redis_bus
    from app.infra.redis_bus import xautoclaim_pending

    fake = _FakeRedis(
        (
            b"0-0",
            [
                (
                    b"1748400000000-0",
                    {
                        b"videoId": b"42",
                        b"gcsPath": b"videos/uploads/test.mp4",
                        b"callbackUrl": b"http://localhost:8080/api/analysis/videos/42",
                    },
                )
            ],
        )
    )
    monkeypatch.setattr(redis_bus, "_client", fake)

    messages = await xautoclaim_pending(
        stream_key="analysis:requests",
        group="hola-ai-worker",
        consumer="worker-a",
        min_idle_ms=60_000,
        count=10,
    )

    assert messages == [
        (
            "1748400000000-0",
            {
                "videoId": b"42",
                "gcsPath": b"videos/uploads/test.mp4",
                "callbackUrl": b"http://localhost:8080/api/analysis/videos/42",
            },
        )
    ]
    assert fake.calls == [
        {
            "name": "analysis:requests",
            "groupname": "hola-ai-worker",
            "consumername": "worker-a",
            "min_idle_time": 60_000,
            "start_id": "0-0",
            "count": 10,
        }
    ]

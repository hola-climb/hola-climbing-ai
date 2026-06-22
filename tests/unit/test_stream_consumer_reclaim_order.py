"""Redis Streams consumer polling order."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_consumer_reads_new_messages_before_reclaiming_stale_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """새 메시지가 있으면 stale pending 회수보다 신규 소비를 우선한다."""
    import app.workers.stream_consumer as consumer_module
    from app.core.config import Settings

    calls: list[str] = []

    async def _ensure_consumer_group(_stream_key: str, _group: str) -> None:
        calls.append("ensure")

    async def _xreadgroup(**_kwargs: Any) -> list[tuple[str, dict[str, object]]]:
        calls.append("read")
        return [
            (
                "new-message-0",
                {
                    "videoId": b"26",
                    "gcsPath": b"videos/uploads/new.mp4",
                    "callbackUrl": b"https://hola-climb.app/api/analysis/videos/26",
                },
            )
        ]

    async def _xautoclaim_pending(**_kwargs: Any) -> list[tuple[str, dict[str, object]]]:
        calls.append("claim")
        return [
            (
                "old-message-0",
                {
                    "videoId": b"25",
                    "gcsPath": b"videos/uploads/old.mp4",
                    "callbackUrl": b"https://hola-climb.app/api/analysis/videos/25",
                },
            )
        ]

    async def _handle_one(
        _settings: Settings,
        _consumer: str,
        msg_id: str,
        _request_or_error: object,
    ) -> None:
        calls.append(f"handle:{msg_id}")
        raise asyncio.CancelledError

    monkeypatch.setattr(consumer_module, "ensure_consumer_group", _ensure_consumer_group)
    monkeypatch.setattr(consumer_module, "xreadgroup", _xreadgroup)
    monkeypatch.setattr(consumer_module, "xautoclaim_pending", _xautoclaim_pending)
    monkeypatch.setattr(consumer_module, "_handle_one", _handle_one)

    settings = Settings(redis_consumer_name="test-consumer", redis_block_ms=1)

    with pytest.raises(asyncio.CancelledError):
        await consumer_module.run_consumer(settings)

    assert calls == ["ensure", "read", "handle:new-message-0"]

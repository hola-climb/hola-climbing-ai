"""FastAPI lifespan worker task orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_lifespan_starts_configured_number_of_consumer_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WORKER_CONCURRENCY만큼 Redis consumer task를 띄운다."""
    import app.main as main_module

    monkeypatch.setenv("WORKER_CONCURRENCY", "3")

    started_slots: list[int | None] = []
    keep_running = asyncio.Event()

    async def _fake_run_consumer(_settings: Any, consumer_slot: int | None = None) -> None:
        started_slots.append(consumer_slot)
        await keep_running.wait()

    async def _fake_close_redis() -> None:
        return None

    monkeypatch.setattr(main_module, "run_consumer", _fake_run_consumer)
    monkeypatch.setattr(main_module, "close_redis", _fake_close_redis)

    async with main_module.lifespan(main_module.app):
        for _ in range(10):
            if len(started_slots) >= 3:
                break
            await asyncio.sleep(0)

        assert started_slots == [1, 2, 3]

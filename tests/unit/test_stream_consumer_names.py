"""Redis consumer naming behavior."""

from __future__ import annotations

import os
import socket


def test_resolve_consumer_name_adds_slot_for_parallel_default_name() -> None:
    """기본 consumer 이름은 프로세스/slot 조합으로 고유화한다."""
    import app.workers.stream_consumer as consumer_module
    from app.core.config import Settings

    settings = Settings(redis_consumer_name="worker-1")

    assert consumer_module._resolve_consumer_name(settings, consumer_slot=2) == (
        f"worker-{socket.gethostname()}-{os.getpid()}-2"
    )


def test_resolve_consumer_name_adds_slot_for_parallel_custom_name() -> None:
    """커스텀 consumer 이름도 병렬 실행 시 slot suffix로 충돌을 피한다."""
    import app.workers.stream_consumer as consumer_module
    from app.core.config import Settings

    settings = Settings(redis_consumer_name="analysis-worker")

    assert consumer_module._resolve_consumer_name(settings, consumer_slot=2) == (
        "analysis-worker-2"
    )

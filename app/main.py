"""FastAPI application entrypoint.

장기 실행 Redis Streams 컨슈머는 lifespan에서 background task로 spawn된다.
HTTP 엔드포인트(`/health`, `/health/ready`)는 운영 보조 용도.

실행:
    uvicorn app.main:app --host $WORKER_HOST --port $WORKER_PORT

또는 컨테이너:
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infra.redis_bus import close_redis
from app.workers.stream_consumer import run_consumer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Bootstrap on startup, cleanup on shutdown.

    - 시작: structlog 설정 → consumer task spawn.
    - 종료: consumer task cancel → await → Redis 클라이언트 close.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "starting hola-climbing-ai worker",
        extra={
            "consumer_group": settings.redis_consumer_group,
            "consumer_name": settings.redis_consumer_name,
            "worker_concurrency": settings.worker_concurrency,
            "stream_key": settings.redis_stream_key,
        },
    )

    consumer_tasks = [
        asyncio.create_task(
            run_consumer(
                settings,
                consumer_slot=slot if settings.worker_concurrency > 1 else None,
            ),
            name=f"stream-consumer-{slot}",
        )
        for slot in range(1, settings.worker_concurrency + 1)
    ]

    try:
        yield
    finally:
        logger.info("shutting down hola-climbing-ai worker")
        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            with suppress(asyncio.CancelledError):
                try:
                    await task
                except Exception:
                    logger.exception("consumer task raised during shutdown")
        logger.info("consumer tasks cancelled cleanly", extra={"count": len(consumer_tasks)})
        await close_redis()


app = FastAPI(
    title="Hola Climbing AI Worker",
    version="0.1.0",
    description="MediaPipe pose + rule-based technique classifier over Redis Streams.",
    lifespan=lifespan,
)
app.include_router(health_router)

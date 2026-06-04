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
from contextlib import asynccontextmanager
from typing import AsyncIterator

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
            "stream_key": settings.redis_stream_key,
        },
    )

    consumer_task = asyncio.create_task(run_consumer(settings), name="stream-consumer")

    try:
        yield
    finally:
        logger.info("shutting down hola-climbing-ai worker")
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            logger.info("consumer task cancelled cleanly")
        except Exception:  # noqa: BLE001
            logger.exception("consumer task raised during shutdown")
        await close_redis()


app = FastAPI(
    title="Hola Climbing AI Worker",
    version="0.1.0",
    description="MediaPipe pose + rule-based technique classifier over Redis Streams.",
    lifespan=lifespan,
)
app.include_router(health_router)

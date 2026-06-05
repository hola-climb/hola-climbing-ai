"""Redis Streams consumer — XREADGROUP 장기 실행 루프.

pipeline-engineer 구현 영역.

흐름:
  1. ensure_consumer_group(stream_key, group)  # XGROUP CREATE ... MKSTREAM (BUSYGROUP 무시)
  2. while not shutdown:
        msgs = XREADGROUP group consumer COUNT 1 BLOCK block_ms STREAMS stream_key >
        for msg in msgs:
            try:
                process_job(req)
                xack(stream_key, group, msg.id)
            except ValidationError / non-retryable:
                xadd_dead_letter(...)
                xack(...)
            except CALLBACK_FAILED:
                xadd_dead_letter(...)
                xack(...)  # PEL 누적 방지

shutdown:
  - asyncio.CancelledError를 catch하여 graceful 종료. 진행 중인 process_job은 await로 마무리.
  - main.py lifespan에서 task.cancel() → 본 함수 종료 → close_redis().

idempotency:
  - Spring AnalysisServiceImpl.ingestResult는 deleteByVideoId 후 insert (멱등). 따라서 워커가
    같은 video_id로 재호출해도 안전 (snapshot §10 #4).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time

from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AnalysisException
from app.infra.redis_bus import (
    ensure_consumer_group,
    xack,
    xadd_dead_letter,
    xautoclaim_pending,
    xreadgroup,
)
from app.models.stream import StreamRequest
from app.services.pipeline.orchestrator import process_job

logger = logging.getLogger(__name__)


def _resolve_consumer_name(settings: Settings) -> str:
    """기본 consumer 이름이 `worker-1`이면 hostname/pid로 고유화."""
    base = settings.redis_consumer_name
    if base and base != "worker-1":
        return base
    host = socket.gethostname()
    return f"worker-{host}-{os.getpid()}"


async def _handle_one(
    settings: Settings,
    consumer: str,
    msg_id: str,
    request_or_error: object,
) -> None:
    """단일 메시지 처리. 어떤 결과든 ack 보장 (PEL 누적 방지)."""
    stream_key = settings.redis_stream_key
    group = settings.redis_consumer_group

    # 파싱 실패 (raw fields가 들어온 경우는 사전에 처리)
    if isinstance(request_or_error, ValidationError):
        logger.error(
            "stream payload validation failed",
            extra={"msg_id": msg_id, "errors": request_or_error.errors()},
        )
        await xadd_dead_letter(
            {
                "original_msg_id": msg_id,
                "reason": "validation_error",
                "detail": str(request_or_error)[:500],
                "ts": str(int(time.time() * 1000)),
            }
        )
        await xack(stream_key, group, msg_id)
        return

    request = request_or_error  # StreamRequest
    try:
        await process_job(request)  # type: ignore[arg-type]
        await xack(stream_key, group, msg_id)
    except AnalysisException as exc:
        # process_job 내부에서 failed 콜백조차 실패한 경우 — dead-letter
        logger.error(
            "process_job raised (callback failed) → dead-letter",
            extra={"msg_id": msg_id, "reason": exc.reason.value, "msg": exc.message},
        )
        await xadd_dead_letter(
            {
                "original_msg_id": msg_id,
                "video_id": str(getattr(request, "video_id", "")),
                "reason": exc.reason.value,
                "detail": exc.message[:500],
                "ts": str(int(time.time() * 1000)),
            }
        )
        await xack(stream_key, group, msg_id)
    except asyncio.CancelledError:
        # 진행 중 cancel — ack 보류. 재시작 후 idle 시간이 지나면 XAUTOCLAIM으로 회수한다.
        logger.info("process_job cancelled mid-flight", extra={"msg_id": msg_id})
        raise
    except Exception as exc:
        logger.exception("process_job unexpected exception", extra={"msg_id": msg_id})
        await xadd_dead_letter(
            {
                "original_msg_id": msg_id,
                "video_id": str(getattr(request, "video_id", "")),
                "reason": "internal",
                "detail": repr(exc)[:500],
                "ts": str(int(time.time() * 1000)),
            }
        )
        await xack(stream_key, group, msg_id)


async def run_consumer(settings: Settings) -> None:
    """장기 실행 컨슈머 진입점. lifespan에서 asyncio.create_task로 spawn.

    Args:
        settings: Settings 인스턴스.

    Cancellation:
        asyncio.CancelledError를 catch하여 graceful shutdown.
        진행 중인 job은 마무리 또는 ack 보류 후 종료.
    """
    stream_key = settings.redis_stream_key
    group = settings.redis_consumer_group
    consumer = _resolve_consumer_name(settings)

    logger.info(
        "consumer starting",
        extra={"stream": stream_key, "group": group, "consumer": consumer},
    )

    # 시작 시 group 보장. 실패하면 워커는 진행 불가 → 상위로 raise.
    await ensure_consumer_group(stream_key, group)

    try:
        while True:
            try:
                msgs = await xautoclaim_pending(
                    stream_key=stream_key,
                    group=group,
                    consumer=consumer,
                    min_idle_ms=settings.redis_pending_min_idle_ms,
                    count=1,
                )
                if msgs:
                    logger.info(
                        "claimed stale pending message",
                        extra={"stream": stream_key, "group": group, "consumer": consumer},
                    )
                else:
                    msgs = await xreadgroup(
                        stream_key=stream_key,
                        group=group,
                        consumer=consumer,
                        block_ms=settings.redis_block_ms,
                        count=1,
                    )
            except ValidationError as ve:
                # 메시지가 있었지만 파싱 실패 — msg_id를 못 얻음. 로그만 남기고 계속.
                # (XREADGROUP는 이미 메시지를 PEL에 등록했으므로 다음 루프에서는 다른 메시지가 옴)
                logger.error("stream parse failure (no msg_id)", extra={"errors": ve.errors()})
                await asyncio.sleep(1.0)
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("xreadgroup error, sleeping 2s")
                await asyncio.sleep(2.0)
                continue

            if not msgs:
                continue

            for msg_id, fields in msgs:
                try:
                    request = StreamRequest.model_validate(fields)
                except ValidationError as ve:
                    await _handle_one(settings, consumer, msg_id, ve)
                    continue
                await _handle_one(settings, consumer, msg_id, request)

    except asyncio.CancelledError:
        logger.info("consumer cancelled, shutting down")
        raise
    finally:
        logger.info("consumer stopped", extra={"consumer": consumer})

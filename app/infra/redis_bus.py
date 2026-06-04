"""Redis async client + Stream/PubSub helpers.

pipeline-engineer 구현 영역.

`redis.asyncio` 사용. Spring과 동일 인스턴스 공유.

구현 결정:
- 모듈 레벨 lazy singleton (`_client`). 첫 호출 시 `Settings`에서 connection 정보 읽음.
  `get_settings()`는 `@lru_cache`로 인스턴스 1개 보장.
- `decode_responses=False`: bytes 그대로 받음. `StreamRequest.model_validate`의
  field_validator가 bytes/str 둘 다 처리하도록 architect가 설계함.
- BUSYGROUP 예외 swallow는 `ResponseError` 메시지 매칭 (redis-py 표준 패턴).
- xreadgroup은 빈 결과 시 [] 반환. block 타임아웃 / no-new-message 둘 다 동일하게 처리.
"""

from __future__ import annotations

import json
import logging
from typing import Final

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from app.core.config import get_settings
from app.models.progress import ProgressEvent

logger = logging.getLogger(__name__)

_client: aioredis.Redis | None = None

_BUSYGROUP: Final[str] = "BUSYGROUP"


async def get_redis() -> aioredis.Redis:
    """Async Redis 클라이언트 (모듈 레벨 lazy singleton).

    `Settings`의 redis_host/port/password/db 사용. decode_responses=False 유지.
    """
    global _client
    if _client is None:
        s = get_settings()
        _client = aioredis.Redis(
            host=s.redis_host,
            port=s.redis_port,
            password=s.redis_password or None,
            db=s.redis_db,
            decode_responses=False,
            health_check_interval=30,
        )
    return _client


async def close_redis() -> None:
    """프로세스 종료 시 호출. lifespan finally에서 호출."""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            logger.warning("redis close error", exc_info=True)
        _client = None


async def ensure_consumer_group(stream_key: str, group: str) -> None:
    """`XGROUP CREATE {stream_key} {group} $ MKSTREAM`.

    BUSYGROUP 에러(이미 존재)는 무시. 그 외 에러는 raise.
    `id="$"`로 신규 메시지만 소비 (재시작 시 미처리 메시지는 PEL 회수로 처리하는 게 정공법이지만
    MVP는 단순화: 새 메시지만).
    """
    r = await get_redis()
    try:
        await r.xgroup_create(name=stream_key, groupname=group, id="$", mkstream=True)
        logger.info("xgroup created", extra={"stream": stream_key, "group": group})
    except ResponseError as exc:
        msg = str(exc)
        if _BUSYGROUP in msg:
            logger.debug("xgroup already exists", extra={"stream": stream_key, "group": group})
            return
        raise


async def publish_progress(event: ProgressEvent) -> None:
    """`PUBLISH analysis:progress <json>`. snake_case 직렬화.

    Pydantic v2: `model_dump_json()`이 `datetime` → ISO-8601 직렬화.
    Spring `AnalysisProgress` record(`Instant updatedAt`)와 호환되도록 UTC `Z` 형식 유지를
    위해 `mode="json"`으로 dump 후 직접 직렬화.
    """
    s = get_settings()
    r = await get_redis()
    # `model_dump(mode="json")`은 datetime을 ISO-8601 문자열로 변환.
    payload = event.model_dump(mode="json")
    # Python ISO-8601은 `+00:00`로 끝남 → Spring Instant 호환 위해 'Z' suffix 변환.
    updated_at = payload.get("updated_at")
    if isinstance(updated_at, str) and updated_at.endswith("+00:00"):
        payload["updated_at"] = updated_at[:-6] + "Z"
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    await r.publish(s.redis_progress_channel, body)
    logger.debug(
        "progress published",
        extra={"video_id": event.video_id, "stage": event.stage, "event_message": event.message},
    )


async def xreadgroup(
    stream_key: str,
    group: str,
    consumer: str,
    block_ms: int = 5000,
    count: int = 1,
) -> list[tuple[str, dict[str, object]]]:
    """`XREADGROUP GROUP {group} {consumer} COUNT {count} BLOCK {block_ms} STREAMS {stream_key} >`.

    Returns:
        [(message_id, raw_fields), ...]. block 타임아웃 시 빈 리스트.

    Notes:
        - `>` (only new) 사용. PEL replay는 별도 함수 (MVP 미구현).
        - 파싱은 호출자(stream_consumer)가 수행한다. 그래야 ValidationError가 나도
          message_id를 보존하여 dead-letter + ack 처리할 수 있다.
    """
    r = await get_redis()
    msgs = await r.xreadgroup(
        groupname=group,
        consumername=consumer,
        streams={stream_key: ">"},
        count=count,
        block=block_ms,
    )
    if not msgs:
        return []
    parsed: list[tuple[str, dict[str, object]]] = []
    # msgs shape: [(stream_name_bytes, [(msg_id_bytes, {field_bytes: value_bytes}), ...])]
    for _stream_name, entries in msgs:
        for msg_id, fields in entries:
            msg_id_str = msg_id.decode() if isinstance(msg_id, (bytes, bytearray)) else str(msg_id)
            # XREADGROUP 결과의 키가 bytes — StreamRequest 검증기가 bytes 처리하지만
            # alias 매칭을 위해 string key dict로 변환.
            str_fields: dict[str, object] = {}
            for k, v in fields.items():
                key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
                str_fields[key] = v
            parsed.append((msg_id_str, str_fields))
    return parsed


async def xack(stream_key: str, group: str, message_id: str) -> None:
    """`XACK {stream_key} {group} {message_id}`."""
    r = await get_redis()
    await r.xack(stream_key, group, message_id)


async def xadd_dead_letter(payload: dict[str, str]) -> None:
    """`XADD {dlq_key} *` — 콜백 4xx 또는 max retry 초과 시 dead-letter 큐로 이동.

    payload는 전부 string으로 직렬화 (Spring Stream과 동일 패턴).
    """
    s = get_settings()
    r = await get_redis()
    await r.xadd(s.redis_dlq_key, payload)
    logger.warning("dead-letter pushed", extra={"dlq": s.redis_dlq_key, **payload})


async def ping() -> bool:
    """Health check. Redis 연결 가능 여부."""
    try:
        r = await get_redis()
        return bool(await r.ping())
    except Exception:
        return False

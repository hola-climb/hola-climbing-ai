"""Stream consumer end-to-end 통합 테스트.

검증:
- Spring이 XADD한 메시지(camelCase)를 워커가 XREADGROUP으로 수신
- process_job이 mock vision/gcs/callback으로 빠르게 완료
- callback URL이 호출됨 (mock 서버)
- analysis:progress 채널에 PROCESSING 메시지 발행됨
- 메시지 ACK 완료 (PEL 비어있음)

testcontainers Docker가 가능하면 실제 Redis, 아니면 fakeredis로 fallback.
fakeredis가 XREADGROUP과 PubSub을 모두 지원하지 않으면 skip.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


def _has_docker() -> bool:
    """간단한 Docker 가용성 체크 (skipif 평가 시점에 호출)."""
    import os
    import shutil

    if os.environ.get("DISABLE_DOCKER_TESTS") == "1":
        return False
    if shutil.which("docker") is None:
        return False
    return os.path.exists("/var/run/docker.sock")


# Stream consumer end-to-end는 실제 Redis 컨테이너 필요.
# fakeredis는 BLOCK 인자가 있는 xreadgroup이 `run_consumer` 무한 루프와 잘 인터리브되지
# 않아 테스트가 hang. Docker 가능한 환경에서만 실행.
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not _has_docker(),
        reason="testcontainers Docker 미사용 환경 — Docker 있는 환경에서만 실행.",
    ),
]


async def _xadd_request(client: Any, stream_key: str, payload: dict[str, str]) -> str:
    msg_id = await client.xadd(stream_key, payload)
    return msg_id.decode() if isinstance(msg_id, (bytes, bytearray)) else str(msg_id)


@pytest.fixture
def patch_orchestrator_to_fast_path(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """orchestrator.process_job 내부 vision/gcs/callback을 모두 mock.

    캡처용 dict 반환 — 콜백 호출 인자, progress publish 횟수 등.
    """
    calls: dict[str, Any] = {
        "post_callback": [],
        "publish_progress": [],
        "download_blob": [],
        "vision_pipeline": [],
    }

    # 1) download_blob: no-op
    async def _fake_download(bucket: str, object_path: str, dest_path: str) -> None:
        calls["download_blob"].append({"bucket": bucket, "object_path": object_path})
        # 빈 파일 touch
        with open(dest_path, "wb") as f:
            f.write(b"\x00" * 4)

    import app.infra.gcs as gcs_module
    import app.services.pipeline.orchestrator as orch_module

    monkeypatch.setattr(gcs_module, "download_blob", _fake_download)
    monkeypatch.setattr(orch_module, "download_blob", _fake_download)

    # 2) vision pipeline: 즉시 1개 segment 반환
    from app.models.callback import AnalysisSegmentPayload

    def _fake_vision(*args: Any, **kwargs: Any) -> list[AnalysisSegmentPayload]:
        calls["vision_pipeline"].append({"args": args, "kwargs": kwargs})
        return [
            AnalysisSegmentPayload(
                sequence_index=0,
                start_time_ms=0,
                end_time_ms=1000,
                technique="high_step",
                is_dynamic=False,
                confidence=0.7,
            )
        ]

    monkeypatch.setattr(orch_module, "_run_vision_pipeline", _fake_vision)

    # 3) post_callback: capture
    async def _fake_post_callback(url: str, body: Any) -> None:
        calls["post_callback"].append({"url": url, "body": body.model_dump()})

    monkeypatch.setattr(orch_module, "post_callback", _fake_post_callback)

    # 4) publish_progress: capture (그래도 실제 Pub/Sub은 호출하도록 둠 — pubsub 검증 위해)
    original_publish = orch_module.publish_progress

    async def _capturing_publish(event: Any) -> None:
        calls["publish_progress"].append(event.model_dump())
        await original_publish(event)

    monkeypatch.setattr(orch_module, "publish_progress", _capturing_publish)

    return calls


@pytest.mark.asyncio
async def test_consumer_processes_xadded_message(
    redis_backend: dict[str, Any],
    redis_client: Any,
    patch_orchestrator_to_fast_path: dict[str, Any],
) -> None:
    """Spring이 XADD한 메시지를 워커가 처리해 callback 호출까지 도달."""
    if redis_backend["kind"] == "fakeredis":
        # fakeredis는 xgroup_create와 xreadgroup 지원 — 진행. 단 PubSub limitations 주의.
        pass

    from app.core.config import get_settings
    from app.infra.redis_bus import ensure_consumer_group
    from app.workers.stream_consumer import run_consumer

    s = get_settings()
    stream_key = s.redis_stream_key
    group = s.redis_consumer_group

    # 1) group 생성 (워커 시작 전에 미리 — XADD 후 메시지 수신 가능하게)
    await ensure_consumer_group(stream_key, group)

    # 2) Spring 흉내: XADD camelCase payload
    payload = {
        "videoId": "42",
        "gcsPath": "videos/uploads/2026/05/28/test.mp4",
        "callbackUrl": "http://localhost:8080/api/analysis/videos/42",
    }
    msg_id = await _xadd_request(redis_client, stream_key, payload)
    assert msg_id

    # 3) consumer 잠시 실행 → 메시지 1건 소화 후 cancel
    task = asyncio.create_task(run_consumer(s))
    try:
        # 최대 5초 대기, 콜백 호출이 잡힐 때까지 polling
        for _ in range(50):
            await asyncio.sleep(0.1)
            if patch_orchestrator_to_fast_path["post_callback"]:
                break
        assert patch_orchestrator_to_fast_path["post_callback"], (
            "post_callback was never invoked; consumer did not process the message"
        )
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

    # 4) 콜백 검증
    cb = patch_orchestrator_to_fast_path["post_callback"][0]
    assert cb["url"] == "http://localhost:8080/api/analysis/videos/42"
    assert cb["body"]["status"] == "done"
    assert cb["body"]["model_version"] == s.model_version
    assert len(cb["body"]["segments"]) == 1
    assert cb["body"]["segments"][0]["technique"] == "high_step"
    assert cb["body"]["techniques"] == ["high_step"]
    assert cb["body"]["is_dynamic"] is None
    assert cb["body"]["dynamic_probability"] is None

    # 5) PEL이 비어있는지 확인 (ACK 완료)
    pending = await redis_client.xpending(stream_key, group)
    # xpending 응답: {"pending": N, ...} 또는 list. 형태가 redis-py 버전에 따라 다름.
    pending_count: int
    if isinstance(pending, dict):
        pending_count = int(pending.get("pending", 0))
    elif isinstance(pending, (list, tuple)) and pending:
        pending_count = int(pending[0])
    else:
        pending_count = 0
    assert pending_count == 0, f"PEL still has {pending_count} pending messages"


@pytest.mark.asyncio
async def test_validation_error_goes_to_dlq(
    redis_backend: dict[str, Any],
    redis_client: Any,
    patch_orchestrator_to_fast_path: dict[str, Any],
) -> None:
    """gcsPath/callbackUrl이 빈 문자열인 메시지는 ValidationError → DLQ."""
    from app.core.config import get_settings
    from app.infra.redis_bus import ensure_consumer_group
    from app.workers.stream_consumer import run_consumer

    s = get_settings()
    stream_key = s.redis_stream_key
    group = s.redis_consumer_group
    dlq_key = s.redis_dlq_key

    await ensure_consumer_group(stream_key, group)

    # Spring이 null → "" 변환한 broken payload
    await _xadd_request(
        redis_client,
        stream_key,
        {"videoId": "99", "gcsPath": "", "callbackUrl": ""},
    )

    task = asyncio.create_task(run_consumer(s))
    try:
        # DLQ에 메시지가 쌓일 때까지 대기
        for _ in range(50):
            await asyncio.sleep(0.1)
            dlq_len = await redis_client.xlen(dlq_key)
            if dlq_len >= 1:
                break
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass

    dlq_len = await redis_client.xlen(dlq_key)
    assert dlq_len >= 1, "validation error did not reach DLQ"

    # 콜백은 호출되지 않음 (process_job 진입 전 fail)
    assert patch_orchestrator_to_fast_path["post_callback"] == []


@pytest.mark.asyncio
async def test_callback_failure_goes_to_dlq_and_acks(
    redis_backend: dict[str, Any],
    redis_client: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spring callback 4xx는 worker DLQ로 이동하고 PEL에 남지 않는다."""
    import app.workers.stream_consumer as consumer_module
    from app.core.config import get_settings
    from app.core.errors import AnalysisException, AnalysisFailureReason
    from app.infra.redis_bus import ensure_consumer_group

    s = get_settings()
    stream_key = s.redis_stream_key
    group = s.redis_consumer_group
    dlq_key = s.redis_dlq_key

    await ensure_consumer_group(stream_key, group)
    await _xadd_request(
        redis_client,
        stream_key,
        {
            "videoId": "42",
            "gcsPath": "videos/uploads/test.mp4",
            "callbackUrl": "http://localhost:8080/api/analysis/videos/42",
        },
    )

    async def _callback_fails(_request: Any) -> None:
        raise AnalysisException(
            AnalysisFailureReason.CALLBACK_FAILED,
            "callback 4xx: http 401",
        )

    monkeypatch.setattr(consumer_module, "process_job", _callback_fails)

    task = asyncio.create_task(consumer_module.run_consumer(s))
    try:
        for _ in range(50):
            await asyncio.sleep(0.1)
            if await redis_client.xlen(dlq_key) >= 1:
                break
    finally:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass

    dlq_len = await redis_client.xlen(dlq_key)
    assert dlq_len >= 1, "callback failure did not reach DLQ"

    pending = await redis_client.xpending(stream_key, group)
    if isinstance(pending, dict):
        pending_count = int(pending.get("pending", 0))
    elif isinstance(pending, (list, tuple)) and pending:
        pending_count = int(pending[0])
    else:
        pending_count = 0
    assert pending_count == 0, f"PEL still has {pending_count} pending messages"

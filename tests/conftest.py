"""pytest 공용 fixture.

- redis_url: testcontainers Redis (env가 지원하면) 또는 fakeredis fallback.
- redis_client: 위 url로 연결된 redis.asyncio.Redis.
- mock_gcs: app.infra.gcs.download_blob 모킹 (가짜 mp4 생성).
- mock_callback_server: respx로 콜백 URL stub.
- settings_override: 테스트용 env 주입.
- 비동기 모드: pyproject.toml `asyncio_mode = "auto"` 활성.

testcontainers Docker 미사용 환경에서는 fakeredis로 자동 폴백 + 경고 출력.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# 환경 변수 사전 주입 — settings 로딩 전에 적용
# ---------------------------------------------------------------------------

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_PASSWORD", "")
os.environ.setdefault("GCS_BUCKET", "test-bucket")
os.environ.setdefault("MODEL_VERSION", "rule_v1")
os.environ.setdefault("CALLBACK_MAX_RETRIES", "3")
os.environ.setdefault("CALLBACK_RETRY_INITIAL_SECONDS", "0.01")
os.environ.setdefault("CALLBACK_TIMEOUT_SECONDS", "2.0")
os.environ.setdefault("REDIS_BLOCK_MS", "200")
os.environ.setdefault("REDIS_CONSUMER_GROUP", "hola-ai-worker-test")
os.environ.setdefault("REDIS_CONSUMER_NAME", "test-worker")
os.environ.setdefault("REDIS_STREAM_KEY", "analysis:requests:test")
os.environ.setdefault("REDIS_DLQ_KEY", "analysis:requests:test:dlq")
os.environ.setdefault("REDIS_PROGRESS_CHANNEL", "analysis:progress:test")


def _docker_available() -> bool:
    """Docker 데몬에 닿을 수 있는지 빠르게 확인 (테스트 환경 분기).

    환경변수 DISABLE_DOCKER_TESTS=1로 강제 false 반환 가능 (CI/로컬 분기).
    """
    import shutil

    if os.environ.get("DISABLE_DOCKER_TESTS") == "1":
        return False
    if shutil.which("docker") is None:
        return False
    try:
        # `docker info` 대신 unix socket 직접 검사 (빠름).
        sock_paths = ["/var/run/docker.sock"]
        for p in sock_paths:
            if os.path.exists(p):
                s = socket.socket(socket.AF_UNIX)
                s.settimeout(0.5)
                try:
                    s.connect(p)
                    s.close()
                    return True
                except OSError:
                    pass
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Redis fixture — testcontainers or fakeredis fallback
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_backend() -> Iterator[dict[str, Any]]:
    """세션 단위 Redis backend. testcontainers 가능하면 사용, 아니면 fakeredis.

    yield dict:
      - kind: "testcontainers" or "fakeredis"
      - host, port: 연결 정보
      - container: 컨테이너 핸들 (testcontainers일 때만)
    """
    if _docker_available():
        try:
            from testcontainers.redis import RedisContainer  # type: ignore

            container = RedisContainer("redis:7-alpine")
            container.start()
            host = container.get_container_host_ip()
            port = int(container.get_exposed_port(6379))
            os.environ["REDIS_HOST"] = host
            os.environ["REDIS_PORT"] = str(port)
            os.environ["REDIS_PASSWORD"] = ""
            # 캐시된 settings 무효화
            from app.core.config import get_settings

            get_settings.cache_clear()
            try:
                yield {"kind": "testcontainers", "host": host, "port": port, "container": container}
            finally:
                container.stop()
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[conftest] testcontainers fallback to fakeredis: {exc!r}")

    # Fallback: fakeredis (in-process). decode_responses=False 모드로 동작.
    yield {"kind": "fakeredis", "host": "fakeredis", "port": 0}


@pytest.fixture
async def redis_client(redis_backend: dict[str, Any]) -> AsyncIterator[Any]:
    """async redis client. fakeredis면 in-process, testcontainers면 실 connection.

    각 테스트 전에 flushdb로 격리.
    """
    if redis_backend["kind"] == "testcontainers":
        import redis.asyncio as aioredis

        client = aioredis.Redis(
            host=redis_backend["host"],
            port=redis_backend["port"],
            decode_responses=False,
        )
    else:
        try:
            import fakeredis.aioredis as fakeasync  # type: ignore
        except ImportError:
            pytest.skip("fakeredis not installed and docker not available")
        client = fakeasync.FakeRedis(decode_responses=False)

    # 깨끗한 상태로 시작
    try:
        await client.flushdb()
    except Exception:
        pass

    # app 모듈의 lazy singleton에 주입
    from app.infra import redis_bus

    redis_bus._client = client  # type: ignore[attr-defined]

    try:
        yield client
    finally:
        redis_bus._client = None  # type: ignore[attr-defined]
        try:
            await client.aclose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# mock_gcs: download_blob 패치 — 가짜 mp4 생성
# ---------------------------------------------------------------------------


def _create_tiny_mp4(path: Path, *, frames: int = 30, fps: int = 30) -> None:
    """numpy + opencv-python으로 1초짜리 검정 mp4 생성. 테스트 영상 픽스처."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 240, 320
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open (codec mp4v not available?)")
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # 프레임마다 약간 다른 값 (디코더가 동일프레임 압축으로 길이 줄이는 것 방지)
        frame[:, :, 0] = (i * 7) % 255
        writer.write(frame)
    writer.release()


@pytest.fixture
def tiny_mp4(tmp_path: Path) -> Path:
    """1초짜리 320x240 검정 mp4. (frames.iter_frames 입력 픽스처)"""
    p = tmp_path / "tiny.mp4"
    _create_tiny_mp4(p)
    return p


@pytest.fixture
def mock_gcs_download(monkeypatch: pytest.MonkeyPatch, tiny_mp4: Path) -> Path:
    """app.infra.gcs.download_blob를 mock — tiny_mp4를 dest_path로 복사."""
    import shutil

    from app.infra import gcs as gcs_module

    async def _fake_download(bucket: str, object_path: str, dest_path: str) -> None:
        shutil.copyfile(tiny_mp4, dest_path)

    monkeypatch.setattr(gcs_module, "download_blob", _fake_download)
    # orchestrator는 from import 했으므로 그쪽도 패치
    import app.services.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "download_blob", _fake_download)
    return tiny_mp4


# ---------------------------------------------------------------------------
# settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def settings():
    """Test settings (env-loaded). 캐시 클리어 후 fresh 인스턴스."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    return s


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop_policy():
    """pytest-asyncio가 새 이벤트 루프를 매 테스트마다 만들도록 (default 동작 유지)."""
    return asyncio.DefaultEventLoopPolicy()

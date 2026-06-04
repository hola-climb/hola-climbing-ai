"""GCS object download — ADC (Application Default Credentials) 기반.

pipeline-engineer 구현 영역.

스냅샷 §6: Stream 메시지의 gcsPath는 객체 경로 (`videos/uploads/.../*.mp4`).
gs:// 또는 https:// prefix 없음. Signed URL 발급 불필요. ADC로 직접 다운로드.

구현 노트:
- `google.cloud.storage` SDK는 동기 API. `asyncio.to_thread`로 블로킹 호출을 워커 이벤트 루프에서 분리.
- `Client()`는 환경변수 `GOOGLE_APPLICATION_CREDENTIALS`를 자동 인식 (ADC).
- 큰 영상이므로 `download_to_filename`이 내부적으로 chunk 다운로드 (기본 100MB chunk).
- `gs://` 또는 절대 URL이 들어와도 객체 경로로 정규화 (방어적 처리).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import storage

from app.core.errors import AnalysisException, AnalysisFailureReason

logger = logging.getLogger(__name__)

_GS_PREFIX: Final[str] = "gs://"


def _normalize_object_path(object_path: str, bucket: str) -> str:
    """gs://bucket/key 또는 bucket/key 형태가 와도 객체 경로만 추출.

    Spring 계약상 raw object path가 정상이지만, 운영 중 다른 형식이 흘러들어와도
    조용히 다운로드 가능하도록 방어적으로 정규화.
    """
    if object_path.startswith(_GS_PREFIX):
        without_prefix = object_path[len(_GS_PREFIX):]
        # `gs://bucket/path/to/file` -> 'path/to/file'
        if "/" in without_prefix:
            _, _, key = without_prefix.partition("/")
            return key
        return without_prefix
    # 'bucket-name/key' 형태로 들어왔다면 버킷명 제거
    if object_path.startswith(f"{bucket}/"):
        return object_path[len(bucket) + 1:]
    return object_path


def _download_blocking(bucket: str, object_path: str, dest_path: str) -> None:
    """동기 GCS 다운로드. to_thread 안에서만 호출."""
    client = storage.Client()
    blob = client.bucket(bucket).blob(object_path)
    blob.download_to_filename(dest_path)


def _bucket_exists_blocking(bucket: str) -> bool:
    """동기 GCS bucket 접근 확인. readiness에서 to_thread로 호출."""
    client = storage.Client()
    return bool(client.bucket(bucket).exists(timeout=3.0))


async def can_access_bucket(bucket: str) -> bool:
    """GCS bucket metadata에 접근 가능한지 확인한다."""
    try:
        return await asyncio.to_thread(_bucket_exists_blocking, bucket)
    except Exception:  # GCS/ADC 설정 실패는 readiness false로만 표현
        logger.warning("gcs readiness check failed", extra={"bucket": bucket}, exc_info=True)
        return False


async def download_blob(bucket: str, object_path: str, dest_path: str) -> None:
    """GCS 객체를 로컬 파일로 다운로드.

    Args:
        bucket: GCS 버킷명 (`GCS_BUCKET` 환경변수).
        object_path: 버킷 내 객체 경로 (`videos/uploads/2026/.../abc.mp4`).
        dest_path: 로컬 저장 경로.

    Raises:
        AnalysisException(VIDEO_DOWNLOAD): NotFound, 권한 거부, 네트워크 실패 등.
    """
    normalized = _normalize_object_path(object_path, bucket)
    logger.info(
        "gcs.download_blob start",
        extra={"bucket": bucket, "object_path": normalized, "dest": dest_path},
    )
    try:
        await asyncio.to_thread(_download_blocking, bucket, normalized, dest_path)
    except NotFound as exc:
        raise AnalysisException(
            AnalysisFailureReason.VIDEO_DOWNLOAD,
            f"gcs object not found: gs://{bucket}/{normalized}",
        ) from exc
    except GoogleAPIError as exc:
        raise AnalysisException(
            AnalysisFailureReason.VIDEO_DOWNLOAD,
            f"gcs api error for gs://{bucket}/{normalized}: {exc}",
        ) from exc
    except OSError as exc:
        # 디스크 풀 / 권한 / 파일 시스템 에러
        raise AnalysisException(
            AnalysisFailureReason.VIDEO_DOWNLOAD,
            f"local fs error writing {dest_path}: {exc}",
        ) from exc
    logger.info(
        "gcs.download_blob done",
        extra={"bucket": bucket, "object_path": normalized, "dest": dest_path},
    )

"""End-to-end job orchestrator — 1 Stream message 처리 흐름.

pipeline-engineer 구현 영역.

흐름:
  1. PROCESSING("분석 시작") publish
  2. GCS download (asyncio.to_thread)
  3. PROCESSING("영상 다운로드 완료") publish
  4. OpenCV frame iterator
  5. MediaPipe pose extraction
  6. PROCESSING("포즈 추정 완료") publish
  7. Segmentation + technique classification
  8. PROCESSING("기술 분류 완료, 결과 전송 중") publish
  9. POST {callback_url} with AnalysisIngestRequest(status="done", segments=[...])
  10. 예외 발생 시: AnalysisIngestRequest(status="failed", segments=[]) 콜백

설계:
- vision 함수는 동기 / CPU-bound → `asyncio.to_thread`로 감싼다.
- 임시 파일은 `tempfile.mkdtemp(dir=settings.gcs_download_dir)`로 격리, finally에서 rmtree.
- AnalysisException은 catch하여 status='failed' 콜백 발송. 콜백 자체 4xx (CALLBACK_FAILED)는
  상위로 re-raise하여 consumer가 dead-letter로 보낸다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AnalysisException, AnalysisFailureReason
from app.infra.gcs import download_blob
from app.infra.redis_bus import publish_progress
from app.models.callback import AnalysisIngestRequest, AnalysisSegmentPayload
from app.models.progress import AnalysisStage, ProgressEvent
from app.models.stream import StreamRequest
from app.services.callback.client import post_callback
from app.services.pipeline.frames import iter_frames
from app.services.vision.classifier import classify_segments
from app.services.vision.pose import extract_pose_landmarks
from app.services.vision.segmenter import split_segments

logger = logging.getLogger(__name__)


async def _publish(video_id: int, message: str) -> None:
    """진행률 발행. 실패해도 분석은 계속 (warn만)."""
    try:
        await publish_progress(
            ProgressEvent(
                video_id=video_id,
                stage=AnalysisStage.PROCESSING,
                message=message,
            )
        )
    except Exception:
        logger.warning(
            "publish_progress failed (ignored)",
            extra={"video_id": video_id, "progress_message": message},
            exc_info=True,
        )


def _run_vision_pipeline(
    video_path: str,
    target_fps: int,
    model_complexity: int,
    min_detection_confidence: float,
    task_model_path: str | None,
) -> list[AnalysisSegmentPayload]:
    """frames → pose → segments → classify. 동기 / CPU-bound. to_thread로 감싸 호출."""
    frames = iter_frames(video_path, target_fps=target_fps)
    pose_frames = extract_pose_landmarks(
        frames,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        task_model_path=task_model_path,
    )
    segments = split_segments(pose_frames)
    payloads = classify_segments(pose_frames, segments)
    return payloads


async def process_job(request: StreamRequest) -> None:
    """1 job 처리 진입점. 모든 예외는 본 함수가 catch하여 status='failed' 콜백으로 변환.

    Args:
        request: Redis Stream에서 파싱한 StreamRequest.

    Raises:
        AnalysisException(CALLBACK_FAILED): 콜백 자체가 4xx/재시도 소진 — consumer가 dead-letter.
    """
    s = get_settings()
    video_id = request.video_id
    callback_url = request.callback_url
    logger.info(
        "process_job start",
        extra={"video_id": video_id, "gcs_path": request.gcs_path, "callback_url": callback_url},
    )

    # 임시 작업 디렉토리 (job-scoped). gcs_download_dir 하위에 격리.
    os.makedirs(s.gcs_download_dir, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"hola-{video_id}-", dir=s.gcs_download_dir))
    video_path = workdir / "video.mp4"

    failure_reason: AnalysisFailureReason | None = None
    failure_msg: str | None = None
    segments_out: list[AnalysisSegmentPayload] = []

    try:
        # 1. 분석 시작
        await _publish(video_id, "분석 시작")

        # 2. GCS 다운로드
        try:
            await download_blob(s.gcs_bucket, request.gcs_path, str(video_path))
        except AnalysisException as exc:
            failure_reason = exc.reason
            failure_msg = exc.message
            logger.error(
                "download failed",
                extra={"video_id": video_id, "reason": exc.reason.value, "msg": exc.message},
            )
            raise

        await _publish(video_id, "영상 다운로드 완료")

        # 3~6. vision 파이프라인 (동기 CPU-bound → to_thread)
        try:
            segments_out = await asyncio.to_thread(
                _run_vision_pipeline,
                str(video_path),
                s.frame_target_fps,
                s.mp_model_complexity,
                s.mp_min_detection_confidence,
                s.mp_task_model_path,
            )
        except AnalysisException as exc:
            failure_reason = exc.reason
            failure_msg = exc.message
            logger.error(
                "vision pipeline failed",
                extra={"video_id": video_id, "reason": exc.reason.value, "msg": exc.message},
            )
            raise
        except Exception as exc:
            failure_reason = AnalysisFailureReason.INTERNAL
            failure_msg = f"vision internal: {exc!r}"
            logger.exception("vision pipeline internal error", extra={"video_id": video_id})
            raise AnalysisException(AnalysisFailureReason.INTERNAL, failure_msg) from exc

        await _publish(video_id, "포즈 추정 완료")
        await _publish(video_id, "기술 분류 완료, 결과 전송 중")

        # 7. 콜백 — done
        done_body = AnalysisIngestRequest(
            status="done",
            model_version=s.model_version,
            segments=segments_out,
        )
        await post_callback(callback_url, done_body)
        logger.info(
            "process_job done",
            extra={"video_id": video_id, "segments": len(segments_out)},
        )
        return

    except AnalysisException as exc:
        # 분석 실패 → failed 콜백 발송. 콜백 자체 실패는 다시 catch.
        if exc.reason is AnalysisFailureReason.CALLBACK_FAILED:
            # done 콜백이 실패한 경우 — 재시도 무의미, 상위로 raise (consumer가 dead-letter)
            raise
        failed_body = AnalysisIngestRequest(
            status="failed",
            model_version=s.model_version,
            segments=[],
        )
        try:
            await post_callback(callback_url, failed_body)
            logger.warning(
                "process_job reported failure to spring",
                extra={
                    "video_id": video_id,
                    "reason": (failure_reason or exc.reason).value,
                    "msg": failure_msg or exc.message,
                },
            )
        except AnalysisException as cb_exc:
            # failed 콜백마저 실패 → consumer가 dead-letter
            logger.error(
                "failed-callback also failed",
                extra={"video_id": video_id, "callback_url": callback_url},
            )
            raise cb_exc from exc
        return

    except Exception as exc:
        logger.exception("process_job unexpected error", extra={"video_id": video_id})
        try:
            await post_callback(
                callback_url,
                AnalysisIngestRequest(
                    status="failed",
                    model_version=s.model_version,
                    segments=[],
                ),
            )
        except AnalysisException as cb_exc:
            raise cb_exc from exc
        return

    finally:
        # 임시 파일 정리 (항상)
        shutil.rmtree(workdir, ignore_errors=True)

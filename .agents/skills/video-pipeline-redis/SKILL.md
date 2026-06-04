---
name: video-pipeline-redis
description: "GCS Signed URL 영상 다운로드, OpenCV 프레임 추출(iterator), Redis Streams 컨슈머(XREADGROUP), Pub/Sub 진행률 발행, Spring 콜백 호출을 구현한다. 'GCS 다운로드', '프레임 추출', 'Redis Streams', '진행률 발행', '콜백 재시도' 요청 시 반드시 사용."
---

# Video Pipeline & Redis Bus

워커의 **입출력 파이프라인**. pipeline-engineer 전용.

## 언제 사용하는가

- Signed URL로 GCS 영상 다운로드
- OpenCV `VideoCapture`로 프레임 iterator 제공
- Redis Streams `XREADGROUP`으로 작업 소비
- 진행률 발행 (Pub/Sub 또는 Streams)
- Spring 콜백 호출 + 재시도

## 핵심 원칙

1. **스트리밍 처리.** 영상 전체를 메모리에 올리지 않는다. 다운로드도 chunked, 디코드도 frame-by-frame.
2. **idempotent.** 같은 작업 ID가 중복 와도 결과가 같다. Redis Streams의 `XACK` 누락 → 중복 수신 가능성 인지.
3. **명명 일관성 (Spring 확정값).**
   - Stream key: `analysis:requests`
   - Progress channel: `analysis:progress` (Pub/Sub, 단일)
   - Consumer group: `hola-ai-worker` (워커가 정의, Spring은 group 강제 안 함)
4. **callbackUrl은 메시지에서 그대로 사용.** 워커가 path를 구성하지 않는다. Spring이 메시지 페이로드에 절대 URL을 담아준다.
5. **에러는 내부 분류 + Spring 호환 콜백.** 콜백 body는 `{status: "failed", model_version, segments: []}` 형식. 내부 에러 코드는 로깅용.

## GCS 다운로드

### Signed URL 직접 HTTP (권장)

Spring이 이미 Signed URL을 발급해 클라이언트가 업로드했으므로, 워커도 동일 URL로 GET 가능.

```python
import httpx, tempfile, os
from pathlib import Path

async def download_to_tmp(signed_url: str, max_bytes: int = 500 * 1024 * 1024) -> Path:
    """청크 단위로 다운로드. 메모리 폭주 방지."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    total = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", signed_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    tmp.close(); os.unlink(tmp.name)
                    raise AnalysisException(
                        ErrorCode.ANALYSIS_VIDEO_DOWNLOAD_FAILED,
                        f"video exceeds {max_bytes} bytes"
                    )
                tmp.write(chunk)
    tmp.close()
    return Path(tmp.name)
```

### 대안: google-cloud-storage 라이브러리

Service Account 자격증명이 있고 버킷 권한이 있다면:

```python
from google.cloud import storage
client = storage.Client()
bucket = client.bucket("hola-climbing-log-videos")
blob = bucket.blob(object_path)
blob.download_to_filename(tmp_path)
```

> Signed URL이 만료될 가능성이 있고 워커가 자격증명을 가지고 있다면 lib 방식이 더 안전.

## OpenCV 프레임 추출

```python
import cv2
from typing import Iterator
import numpy as np

def iter_frames(video_path: Path, every_n: int = 3) -> Iterator[tuple[int, np.ndarray]]:
    """30fps 영상에서 every_n=3 → 10fps로 다운샘플."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise AnalysisException(ErrorCode.ANALYSIS_DECODE_FAILED, f"cannot open {video_path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % every_n == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
```

샘플링 주기 (`every_n`)는 vision-engineer와 협의. 너무 빠르면 데드포인트 감지 실패, 너무 느리면 처리 시간 폭증.

### 영상 메타데이터 미리 추출

```python
def get_video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    meta["duration_seconds"] = meta["frame_count"] / meta["fps"] if meta["fps"] else 0
    return meta
```

## Redis Streams 컨슈머

> 확정값: stream `analysis:requests`, group `hola-ai-worker`. 페이로드 키는 `videoId` (String), `gcsPath`, `callbackUrl`.

### 컨슈머 그룹 생성 (idempotent)

```python
import redis.asyncio as aioredis

STREAM = "analysis:requests"
GROUP = "hola-ai-worker"

async def ensure_group(r: aioredis.Redis):
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
```

### 메시지 파싱

```python
def parse_job(fields: dict) -> AnalysisJob:
    """Spring AnalysisDispatcher가 XADD한 페이로드 파싱."""
    return AnalysisJob(
        video_id=int(fields[b"videoId"]),     # String → int
        gcs_path=fields[b"gcsPath"].decode(),
        callback_url=fields[b"callbackUrl"].decode(),
    )
```

### 메인 루프

```python
async def consume_loop(r, consumer: str, handle):
    await ensure_group(r)
    while True:
        msgs = await r.xreadgroup(
            GROUP, consumer,
            streams={STREAM: ">"},
            count=1,
            block=5000,
        )
        if not msgs: continue
        for _, entries in msgs:
            for msg_id, fields in entries:
                try:
                    job = parse_job(fields)
                    await handle(msg_id, job)
                    await r.xack(STREAM, GROUP, msg_id)
                except Exception as e:
                    log.exception("handler failed", msg_id=msg_id)
                    # PEL에 남아있음 → 재처리 가능. dead-letter 정책은 별도.
```

### PEL 회수 (재시작 시)

워커 재시작 시 미완료 메시지(PEL) 회수:

```python
async def reclaim_stale(r, stream, group, consumer, min_idle_ms=60000):
    pending = await r.xpending_range(stream, group, "-", "+", count=100)
    for entry in pending:
        if entry["time_since_delivered"] > min_idle_ms:
            await r.xclaim(stream, group, consumer, min_idle_ms, [entry["message_id"]])
```

## 진행률 발행

Spring이 SSE (`/api/videos/{videoId}/analysis/stream`, 이벤트명 `progress`)로 클라이언트에 중계.

**확정값:** Pub/Sub 채널 `analysis:progress` (단일). 페이로드에 `videoId`를 포함하여 Spring listener가 라우팅.

```python
PROGRESS_CHANNEL = "analysis:progress"

async def publish_progress(r, video_id: int, stage: str, percent: int | None = None, message: str | None = None):
    """
    stage: 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' (Spring AnalysisStage enum)
    percent: 0~100 (PROCESSING일 때만 의미). 너무 자주 발행하지 말 것.
    """
    payload = {
        "video_id": video_id,
        "stage": stage,
        "percent": percent,
        "message": message,
        "ts": int(time.time() * 1000),
    }
    await r.publish(PROGRESS_CHANNEL, json.dumps(payload))
```

> 페이로드 정확한 shape은 Spring `AnalysisProgressEvent` 클래스를 확인하여 일치시킨다 (integration-engineer가 검증).

**발행 빈도:** 매 프레임 발행 금지. **5% 단위 또는 1초 간격** + stage 전환 시. 다이노 같은 짧은 이벤트는 PROCESSING 안에서 진행률로만 표현.

**스테이지 라이프사이클:**
1. 메시지 수신 직후 → `QUEUED` (Spring이 이미 발행했을 수 있음. 워커 중복 발행 회피)
2. 다운로드 시작 → `PROCESSING` + percent=5
3. 디코드/추론 진행 → `PROCESSING` + percent=10~95
4. 콜백 성공 → `COMPLETED` + percent=100
5. 어떤 단계든 실패 → `FAILED` + message

## Spring 콜백

**확정값:** 콜백 URL은 메시지 페이로드의 `callbackUrl` 그대로 사용. 워커가 path를 구성하지 않는다.
실제 형식: `POST {callbackUrl}` (예: `http://localhost:8080/api/analysis/videos/123`).

**Body (`AnalysisIngestRequest`):**
```python
{
    "status": "done",                  # or "failed"
    "model_version": "rule_v1",        # vision-engineer가 결정
    "segments": [                      # vision의 finalize_segments() 출력
        {
            "sequence_index": 0,
            "start_time_ms": 1500,
            "end_time_ms": 2800,
            "technique": "high_step",
            "is_dynamic": False,
            "confidence": 0.7,
        },
        ...
    ]
}
```

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class TransientCallbackError(Exception): pass
class PermanentCallbackError(Exception): pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(TransientCallbackError),
    reraise=True,
)
async def callback(callback_url: str, payload: dict):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(callback_url, json=payload)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise TransientCallbackError(f"{resp.status_code} {resp.text[:200]}")
        if resp.status_code >= 400:
            # 4xx = 계약 불일치. 재시도 무의미.
            raise PermanentCallbackError(f"{resp.status_code} {resp.text[:200]}")
        resp.raise_for_status()
```

**failed 콜백:** vision이 실패해도 Spring에는 반드시 콜백을 보낸다.
```python
{"status": "failed", "model_version": "rule_v1", "segments": []}
```

3회 실패 시 dead-letter Streams로 이동:

```python
async def to_dead_letter(r, original_msg_id, job_id, error_code):
    await r.xadd("hola:analysis:dlq", {
        "original_msg_id": original_msg_id,
        "job_id": job_id,
        "error_code": error_code,
        "ts": int(time.time() * 1000),
    })
```

## 임시 파일 관리

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def video_workspace(job_id: int):
    workdir = Path(tempfile.mkdtemp(prefix=f"hola-{job_id}-"))
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
```

## 산출물 체크리스트

- [ ] `_workspace/03_pipeline_gcs_download.md` — 다운로드 전략 + 자격증명
- [ ] `_workspace/03_pipeline_frame_iterator.md` — 프레임 iterator API 확정
- [ ] `_workspace/03_pipeline_redis_consumer.md` — Streams 컨슈머 구현
- [ ] `_workspace/03_pipeline_callback.md` — 콜백 + 재시도 + DLQ
- [ ] 코드: `app/infra/gcs.py`, `app/infra/redis_bus.py`, `app/workers/stream_consumer.py`

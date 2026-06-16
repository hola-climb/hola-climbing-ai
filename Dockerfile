# syntax=docker/dockerfile:1.7
# =============================================================================
# Hola Climbing AI Worker — Dockerfile
# - Base: python:3.11-slim-bookworm (MediaPipe / OpenCV manylinux wheel 호환)
# - Build: multi-stage (uv export → runtime)
# - Platform: linux/amd64 강제 (MediaPipe ARM64 wheel 부재)
# Apple Silicon 사용자: docker-compose.yml의 `platform: linux/amd64`로 Rosetta 동작.
# =============================================================================

# ---------- 1) Builder: uv로 의존성 설치 ----------
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# uv 설치 (공식 standalone — pip 대비 빠른 resolve)
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# 의존성 메타데이터만 먼저 복사 → 레이어 캐시 최적화
COPY pyproject.toml ./
# hatchling metadata가 pyproject.toml의 `readme = "README.md"`를 검증한다.
COPY README.md ./
# uv.lock가 존재하면 frozen 설치, 없으면 일반 sync
COPY uv.lock* ./

# 런타임 deps만 설치 (dev group 제외)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# ---------- 2) Runtime ----------
FROM --platform=linux/amd64 python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    # OpenCV가 GUI 시도하지 않도록
    QT_QPA_PLATFORM=offscreen \
    # 다운로드 임시 디렉토리 (.env에서 override 가능)
    GCS_DOWNLOAD_DIR=/tmp/hola-videos

# system deps — MediaPipe / OpenCV 런타임 라이브러리
# - ffmpeg: 영상 코덱 디코딩
# - libgl1, libglib2.0-0: OpenCV/MediaPipe 공용
# - libgles2, libegl1: MediaPipe Tasks native runtime
# - ca-certificates: TLS (GCS, Spring 콜백)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libgles2 \
        libegl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 비루트 사용자
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && mkdir -p /home/appuser/.config/gcloud /tmp/hola-videos /app \
    && chown -R appuser:appuser /home/appuser /tmp/hola-videos /app

WORKDIR /app

# builder의 venv를 그대로 복사
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# 애플리케이션 코드
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser pyproject.toml ./

# MediaPipe 0.10.35는 mp.solutions가 없어 Tasks 모델 파일이 필요하다.
# gitignored 로컬 models/에 의존하지 않도록 이미지 빌드 시 checksum으로 고정 다운로드한다.
RUN mkdir -p /app/models/mediapipe \
    && chown -R appuser:appuser /app/models
ADD --checksum=sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a \
    --chown=appuser:appuser \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
    /app/models/mediapipe/pose_landmarker_lite.task

# flow gate RF artifact (git 추적됨). 활성화는 FLOW_GATE_MODEL_PATH 환경변수로 결정.
COPY --chown=appuser:appuser models/flow_qa_rf_v2.joblib ./models/flow_qa_rf_v2.joblib

USER appuser

EXPOSE 8000

# /health 엔드포인트 liveness 체크 (curl 없이 python 내장 urllib 사용)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# uvicorn으로 FastAPI 실행 (lifespan에서 Redis Streams consumer task spawn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

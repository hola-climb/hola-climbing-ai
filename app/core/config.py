"""Application settings loaded from environment / .env file.

본 모듈은 `_workspace/01_architect_env.md`의 변수 표와 1:1 대응한다.
변경 시 .env.example과 architect 문서를 동시 수정한다.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Worker runtime settings.

    Spring 서버(hola-climbing-server)의 application.yaml과 공유해야 하는
    Redis/GCS 값은 변경 시 즉시 통합 테스트로 검증.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Worker ---
    worker_host: str = "0.0.0.0"
    worker_port: int = 8000
    log_level: str = "INFO"
    model_version: str = "rule_v3"

    # --- Redis (Spring과 동일) ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_stream_key: str = "analysis:requests"
    redis_consumer_group: str = "hola-ai-worker"
    redis_consumer_name: str = "worker-1"
    redis_progress_channel: str = "analysis:progress"
    redis_block_ms: int = 5000
    redis_dlq_key: str = "analysis:requests:dlq"
    redis_pending_min_idle_ms: int = Field(default=60000, ge=1000)

    # --- GCS ---
    gcs_bucket: str = "hola-climbing-log-videos"
    google_application_credentials: str | None = None
    gcs_download_dir: str = "/tmp/hola-videos"

    # --- Callback ---
    ai_callback_secret: str = ""
    callback_timeout_seconds: float = 10.0
    callback_max_retries: int = 3
    callback_retry_initial_seconds: float = 1.0

    # --- MediaPipe ---
    mp_model_complexity: int = Field(default=1, ge=0, le=2)
    mp_min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    mp_task_model_path: str | None = "models/mediapipe/pose_landmarker_lite.task"
    frame_target_fps: int = Field(default=15, ge=1, le=60)

    # --- Flow gate (optional ML inference, empty string = off) ---
    flow_gate_model_path: str | None = "models/flow_qa_rf_v2.joblib"
    flow_gate_static_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    flow_gate_dynamic_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    flow_gate_label_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    flow_gate_demote_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    flow_gate_version_suffix: str = "flow_rf_v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Memoized settings accessor (1 instance per process)."""
    return Settings()

"""`analysis:progress` Pub/Sub payload model.

Spring `infrastructure/ai/AnalysisProgress.java`와 호환.
Jackson SNAKE_CASE로 직렬화되므로 본 모델의 필드명은 snake_case 그대로 사용.

워커는 `PROCESSING` 단계만 publish. COMPLETED/FAILED는 Spring이 콜백 처리 후 자동 발행.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AnalysisStage(StrEnum):
    """Spring `AnalysisStage` enum. 대문자 문자열로 직렬화."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProgressEvent(BaseModel):
    """`PUBLISH analysis:progress <json>` payload."""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    video_id: int
    stage: AnalysisStage
    message: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

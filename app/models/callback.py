"""Worker → Spring callback body model.

Spring `domain/analysis/dto/request/AnalysisIngestRequest.java`와 1:1 호환.
JSON 직렬화 시 필드명이 이미 snake_case이므로 alias_generator 불필요.

POST {callbackUrl} body로 사용.
"""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CANONICAL_TECHNIQUE_ORDER: Final[tuple[str, ...]] = (
    "high_step",
    "flagging",
    "toe_hook",
    "heel_hook",
    "lock_off",
    "dyno",
    "coordination",
)


def derive_techniques(segments: list[AnalysisSegmentPayload]) -> list[str]:
    """segments에 등장한 기술을 영상 단위 canonical order로 중복 제거."""
    present = {segment.technique for segment in segments}
    return [technique for technique in CANONICAL_TECHNIQUE_ORDER if technique in present]


class AnalysisSegmentPayload(BaseModel):
    """Spring `AnalysisSegmentPayload`와 호환. 한 구간(segment) 한 건."""

    model_config = ConfigDict(populate_by_name=True)

    sequence_index: int = Field(ge=0)
    start_time_ms: int | None = None
    end_time_ms: int | None = None
    technique: str = Field(min_length=1)
    is_dynamic: bool | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalysisIngestRequest(BaseModel):
    """Worker → Spring 콜백 body. `POST {callbackUrl}` 요청 페이로드.

    status 필드는 `"done"` 또는 `"failed"`만 허용된다 (Spring `AnalysisServiceImpl:50`).
    그 외 값은 Spring 측 `INVALID_INPUT` (C001)을 유발한다.
    """

    model_config = ConfigDict(populate_by_name=True)

    status: Literal["done", "failed"]
    model_version: str | None = None
    segments: list[AnalysisSegmentPayload] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    is_dynamic: bool | None = None
    dynamic_probability: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def derive_video_level_techniques(self) -> Self:
        self.techniques = derive_techniques(self.segments)
        return self

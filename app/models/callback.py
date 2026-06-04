"""Worker → Spring callback body model.

Spring `domain/analysis/dto/request/AnalysisIngestRequest.java`와 1:1 호환.
JSON 직렬화 시 필드명이 이미 snake_case이므로 alias_generator 불필요.

POST {callbackUrl} body로 사용.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

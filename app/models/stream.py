"""Redis Stream `analysis:requests` message model.

Spring `RedisStreamAnalysisJobQueue.java`가 XADD하는 raw payload는 **camelCase** 키다.
(Spring Jackson SNAKE_CASE는 HTTP JSON에만 적용되고 Stream payload는 영향 없음.)

본 모델은 입력 전용. 출력(콜백, 진행률)은 snake_case 모델을 별도 정의.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class StreamRequest(BaseModel):
    """Worker-side parsing model for `XREADGROUP analysis:requests`."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    video_id: int = Field(..., alias="videoId")
    gcs_path: str = Field(..., alias="gcsPath", min_length=1)
    callback_url: str = Field(..., alias="callbackUrl", min_length=1)

    @field_validator("video_id", mode="before")
    @classmethod
    def _coerce_video_id(cls, v: object) -> int:
        """XREADGROUP은 bytes를 반환한다 — bytes/str 모두 int로 변환."""
        if isinstance(v, (bytes, bytearray)):
            v = v.decode()
        return int(v)  # type: ignore[arg-type]

    @field_validator("gcs_path", "callback_url", mode="before")
    @classmethod
    def _decode_bytes(cls, v: object) -> object:
        if isinstance(v, (bytes, bytearray)):
            return v.decode()
        return v

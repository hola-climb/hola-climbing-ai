"""ApiResponse wrapper — compatible with Spring `common/response/ApiResponse.java`.

워커가 호출되는 일은 거의 없지만 (`/health` 등 운영 보조 엔드포인트),
Spring과 톤을 맞추기 위해 동일 shape으로 응답한다.

Spring 필드:
  - isSuccess (boolean) -> JSON `is_success`
  - code (string), 성공 시 "OK"
  - message (string, nullable)
  - data (T, nullable)
  - timestamp (Instant ISO-8601 UTC)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper. snake_case 직렬화."""

    model_config = ConfigDict(populate_by_name=True)

    is_success: bool
    code: str = "OK"
    message: str | None = None
    data: T | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def ok(cls, data: T | None = None) -> ApiResponse[T]:
        return cls(is_success=True, code="OK", data=data)

    @classmethod
    def error(cls, code: str, message: str) -> ApiResponse[None]:
        return cls(is_success=False, code=code, message=message, data=None)

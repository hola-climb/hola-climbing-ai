"""Pydantic v2 data models.

케이싱 정책 (architect 결정, `_workspace/01_architect_contract.md` §6):
- `stream.py`: camelCase (Redis Stream raw key) — alias_generator=to_camel
- `callback.py`, `progress.py`, `response.py`: snake_case (Spring SNAKE_CASE 호환)
"""

from app.models.callback import AnalysisIngestRequest, AnalysisSegmentPayload
from app.models.progress import AnalysisStage, ProgressEvent
from app.models.response import ApiResponse
from app.models.stream import StreamRequest

__all__ = [
    "AnalysisIngestRequest",
    "AnalysisSegmentPayload",
    "AnalysisStage",
    "ApiResponse",
    "ProgressEvent",
    "StreamRequest",
]

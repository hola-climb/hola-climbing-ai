"""Logging configuration — structlog + stdlib bridge.

pipeline-engineer가 구조화 로그 컨텍스트 (video_id, message_id) 추가 가능.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger. uvicorn은 별도 logger를 가지므로 둘 다 영향받도록 설정."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
        force=True,
    )
    # TODO(pipeline-engineer): structlog processor chain (JSON renderer in prod)

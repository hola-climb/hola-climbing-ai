"""process_job이 새 callback 계약을 조립하는지 검증."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import get_settings
from app.models.callback import AnalysisSegmentPayload
from app.models.stream import StreamRequest


def _segment(
    idx: int,
    technique: str,
    *,
    is_dynamic: bool,
    confidence: float = 0.8,
) -> AnalysisSegmentPayload:
    return AnalysisSegmentPayload(
        sequence_index=idx,
        start_time_ms=idx * 1000,
        end_time_ms=idx * 1000 + 900,
        technique=technique,
        is_dynamic=is_dynamic,
        confidence=confidence,
    )


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def stream_request() -> StreamRequest:
    return StreamRequest.model_validate(
        {
            "videoId": "42",
            "gcsPath": "videos/uploads/test.mp4",
            "callbackUrl": "http://localhost:8080/api/analysis/videos/42",
        }
    )


def test_flow_gate_model_path_defaults_to_operational_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOW_GATE_MODEL_PATH", raising=False)
    get_settings.cache_clear()

    assert get_settings().flow_gate_model_path == "models/flow_qa_rf_v2.joblib"


async def _run_job_and_capture_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stream_request: StreamRequest,
    segments: list[AnalysisSegmentPayload],
) -> dict[str, Any]:
    import app.services.pipeline.orchestrator as orch_module

    monkeypatch.setenv("GCS_DOWNLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_VERSION", "rule_v3")
    get_settings.cache_clear()

    async def _fake_publish(*args: Any, **kwargs: Any) -> None:
        return None

    async def _fake_download(bucket: str, object_path: str, dest_path: str) -> None:
        with open(dest_path, "wb") as f:
            f.write(b"video")

    def _fake_vision(*args: Any, **kwargs: Any) -> list[AnalysisSegmentPayload]:
        return segments

    captured: dict[str, Any] = {}

    async def _fake_post_callback(url: str, body: Any) -> None:
        captured["url"] = url
        captured["body"] = body.model_dump()

    monkeypatch.setattr(orch_module, "_publish", _fake_publish)
    monkeypatch.setattr(orch_module, "download_blob", _fake_download)
    monkeypatch.setattr(orch_module, "_run_vision_pipeline", _fake_vision)
    monkeypatch.setattr(orch_module, "post_callback", _fake_post_callback)

    await orch_module.process_job(stream_request)
    return captured["body"]


@pytest.mark.asyncio
async def test_video_level_dynamic_stays_null_when_flow_gate_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stream_request: StreamRequest,
) -> None:
    monkeypatch.setenv("FLOW_GATE_MODEL_PATH", "")
    segments = [
        _segment(0, "dyno", is_dynamic=True),
        _segment(1, "high_step", is_dynamic=False),
    ]

    body = await _run_job_and_capture_body(monkeypatch, tmp_path, stream_request, segments)

    assert body["status"] == "done"
    assert body["segments"][0]["is_dynamic"] is True
    assert body["techniques"] == ["high_step", "dyno"]
    assert body["is_dynamic"] is None
    assert body["dynamic_probability"] is None


@pytest.mark.asyncio
async def test_video_level_dynamic_uses_flow_gate_probability_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stream_request: StreamRequest,
) -> None:
    monkeypatch.setenv("FLOW_GATE_MODEL_PATH", str(tmp_path / "gate.joblib"))
    monkeypatch.setenv("FLOW_GATE_LABEL_THRESHOLD", "0.5")
    segments = [_segment(0, "lock_off", is_dynamic=False)]

    import app.services.vision.flow_gate as flow_gate_module

    def _fake_apply_flow_gate(*args: Any, **kwargs: Any):
        return segments, 0.5

    monkeypatch.setattr(flow_gate_module, "apply_flow_gate", _fake_apply_flow_gate)

    body = await _run_job_and_capture_body(monkeypatch, tmp_path, stream_request, segments)

    assert body["model_version"] == "rule_v3+flow_rf_v2"
    assert body["techniques"] == ["lock_off"]
    assert body["is_dynamic"] is True
    assert body["dynamic_probability"] == 0.5


@pytest.mark.asyncio
async def test_video_level_dynamic_stays_null_when_flow_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    stream_request: StreamRequest,
) -> None:
    monkeypatch.setenv("FLOW_GATE_MODEL_PATH", str(tmp_path / "gate.joblib"))
    segments = [_segment(0, "coordination", is_dynamic=True)]

    import app.services.vision.flow_gate as flow_gate_module

    def _fake_apply_flow_gate(*args: Any, **kwargs: Any):
        raise RuntimeError("flow unavailable")

    monkeypatch.setattr(flow_gate_module, "apply_flow_gate", _fake_apply_flow_gate)

    body = await _run_job_and_capture_body(monkeypatch, tmp_path, stream_request, segments)

    assert body["model_version"] == "rule_v3"
    assert body["segments"][0]["is_dynamic"] is True
    assert body["techniques"] == ["coordination"]
    assert body["is_dynamic"] is None
    assert body["dynamic_probability"] is None

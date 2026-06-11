"""flow_gate 단위 테스트 — 보정 정책 + artifact 검증 + fallback 경계."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from app.models.callback import AnalysisSegmentPayload
from app.services.vision.flow_gate import (
    _artifact_cache,
    _load_artifact,
    adjust_segments,
    apply_flow_gate,
    predict_prob_dynamic,
)

STATIC_TH = 0.30
DYNAMIC_TH = 0.70
DEMOTE_CONF = 0.55


def _seg(
    idx: int,
    technique: str,
    confidence: float,
) -> AnalysisSegmentPayload:
    return AnalysisSegmentPayload(
        sequence_index=idx,
        start_time_ms=idx * 1000,
        end_time_ms=idx * 1000 + 900,
        technique=technique,
        is_dynamic=technique in ("dyno", "coordination"),
        confidence=confidence,
    )


def _adjust(segments, prob):
    return adjust_segments(
        segments,
        prob,
        static_threshold=STATIC_TH,
        dynamic_threshold=DYNAMIC_TH,
        demote_confidence=DEMOTE_CONF,
    )


class TestAdjustSegments:
    def test_static_verdict_drops_weak_dynamic_segments(self):
        segments = [
            _seg(0, "high_step", 0.8),
            _seg(1, "dyno", 0.40),  # 약한 dynamic → drop
            _seg(2, "lock_off", 0.6),
        ]
        out = _adjust(segments, prob=0.10)
        assert [s.technique for s in out] == ["high_step", "lock_off"]

    def test_static_verdict_keeps_confident_dynamic_segments(self):
        # rule confidence 높음 → flow가 static이라 해도 유지 (백다이노 보호)
        segments = [_seg(0, "dyno", 0.85)]
        out = _adjust(segments, prob=0.05)
        assert len(out) == 1
        assert out[0].technique == "dyno"

    def test_dropped_segments_resequence_index(self):
        segments = [
            _seg(0, "coordination", 0.40),
            _seg(1, "flagging", 0.7),
            _seg(2, "dyno", 0.30),
            _seg(3, "heel_hook", 0.5),
        ]
        out = _adjust(segments, prob=0.20)
        assert [s.technique for s in out] == ["flagging", "heel_hook"]
        assert [s.sequence_index for s in out] == [0, 1]
        # 원본은 불변
        assert [s.sequence_index for s in segments] == [0, 1, 2, 3]

    def test_dynamic_verdict_is_noop(self):
        segments = [_seg(0, "dyno", 0.40)]
        out = _adjust(segments, prob=0.90)
        assert out is segments

    def test_uncertain_verdict_is_noop(self):
        segments = [_seg(0, "dyno", 0.40)]
        out = _adjust(segments, prob=0.50)
        assert out is segments

    def test_static_verdict_never_touches_static_techniques(self):
        segments = [
            _seg(0, "high_step", 0.36),  # 약한 confidence지만 static 기술 → 유지
            _seg(1, "toe_hook", 0.36),
        ]
        out = _adjust(segments, prob=0.01)
        assert len(out) == 2

    def test_no_drop_returns_same_list(self):
        segments = [_seg(0, "lock_off", 0.9)]
        out = _adjust(segments, prob=0.10)
        assert out is segments


class TestLoadArtifact:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _load_artifact(str(tmp_path / "nope.joblib"))

    def test_wrong_shape_raises(self, tmp_path: Path):
        p = tmp_path / "bad.joblib"
        joblib.dump(["not", "a", "dict"], p)
        with pytest.raises(ValueError, match="artifact shape"):
            _load_artifact(str(p))

    def test_wrong_classes_raises(self, tmp_path: Path):
        p = tmp_path / "bad_classes.joblib"
        joblib.dump({"model": object(), "classes": ["a", "b"]}, p)
        with pytest.raises(ValueError, match="classes mismatch"):
            _load_artifact(str(p))

    def test_valid_artifact_is_cached(self, tmp_path: Path):
        p = tmp_path / "ok.joblib"
        joblib.dump(
            {"model": object(), "classes": ["static", "dynamic"], "feature_dim": 42},
            p,
        )
        art1 = _load_artifact(str(p))
        art2 = _load_artifact(str(p))
        assert art1 is art2
        _artifact_cache.pop(str(p), None)


class TestApplyFlowGate:
    def test_returns_adjusted_segments_and_prob(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.vision.flow_gate.predict_prob_dynamic",
            lambda video_path, model_path: 0.10,
        )
        segments = [_seg(0, "dyno", 0.40), _seg(1, "high_step", 0.8)]
        out, prob = apply_flow_gate(
            "video.mp4",
            segments,
            model_path="unused.joblib",
            static_threshold=STATIC_TH,
            dynamic_threshold=DYNAMIC_TH,
            demote_confidence=DEMOTE_CONF,
        )
        assert prob == 0.10
        assert [s.technique for s in out] == ["high_step"]


class _RecordingModel:
    def __init__(self) -> None:
        self.seen_shape: tuple[int, int] | None = None

    def predict_proba(self, x):
        self.seen_shape = x.shape
        return np.asarray([[0.8, 0.2]], dtype=np.float64)


class TestPredictProbDynamic:
    def test_uses_legacy_42_dim_features_for_v2_artifacts(self, monkeypatch):
        model = _RecordingModel()
        monkeypatch.setattr(
            "app.services.vision.flow_gate._load_artifact",
            lambda model_path: {"model": model, "classes": ["static", "dynamic"], "feature_dim": 42},
        )
        monkeypatch.setattr(
            "app.services.vision.flow_features.extract_flow_series",
            lambda video_path: (
                np.stack(
                    [
                        np.linspace(0.1, 1.0, num=90, dtype=np.float32),
                        np.zeros(90, dtype=np.float32),
                    ],
                    axis=1,
                ),
                30.0,
                3.0,
            ),
        )

        prob = predict_prob_dynamic("video.mp4", "v2.joblib")

        assert prob == 0.2
        assert model.seen_shape == (1, 42)

    def test_uses_v3_46_dim_features_for_v3_artifacts(self, monkeypatch):
        model = _RecordingModel()
        monkeypatch.setattr(
            "app.services.vision.flow_gate._load_artifact",
            lambda model_path: {"model": model, "classes": ["static", "dynamic"], "feature_dim": 46},
        )
        monkeypatch.setattr(
            "app.services.vision.flow_features.extract_flow_series",
            lambda video_path: (
                np.stack(
                    [
                        np.linspace(0.1, 1.0, num=90, dtype=np.float32),
                        np.zeros(90, dtype=np.float32),
                    ],
                    axis=1,
                ),
                30.0,
                3.0,
            ),
        )

        prob = predict_prob_dynamic("video.mp4", "v3.joblib")

        assert prob == 0.2
        assert model.seen_shape == (1, 46)

    def test_uses_v4_58_dim_features_for_v4_artifacts(self, monkeypatch):
        model = _RecordingModel()
        monkeypatch.setattr(
            "app.services.vision.flow_gate._load_artifact",
            lambda model_path: {"model": model, "classes": ["static", "dynamic"], "feature_dim": 58},
        )
        monkeypatch.setattr(
            "app.services.vision.flow_features.extract_flow_series",
            lambda video_path: (
                np.stack(
                    [
                        np.linspace(0.1, 1.0, num=90, dtype=np.float32),
                        np.linspace(-0.2, 0.2, num=90, dtype=np.float32),
                    ],
                    axis=1,
                ),
                30.0,
                3.0,
            ),
        )

        prob = predict_prob_dynamic("video.mp4", "v4.joblib")

        assert prob == 0.2
        assert model.seen_shape == (1, 58)

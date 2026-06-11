"""영상 단위 dynamic/static flow 게이트 — optional ML 추론.

flow-only RF (group-kfold balanced accuracy 0.8381, 2026-06-10 재학습)가
영상 전체의 dynamic/static 경향을 판정하고, 그 판정을 rule 기반 segment
출력의 사후 보정 prior로 사용한다. 콜백 계약(AnalysisIngestRequest)은
변경하지 않는다.

정책 — 보수적 앙상블. 두 신호가 모두 약할 때만 개입한다:

- prob_dynamic < flow_gate_static_threshold:
    rule이 약하게 잡은 dynamic 기술 segment
    (confidence < flow_gate_demote_confidence)를 drop.
    rule confidence가 높은 segment는 유지 — flow는 백다이노처럼 화면상
    움직임이 작은 동작을 놓치는 약점이 확인됐다 (miss review 2026-06-10).
- prob_dynamic > flow_gate_dynamic_threshold: 개입 없음 (segment 생성 안 함).
- 그 외 (uncertain 구간): 개입 없음.

FLOW_GATE_MODEL_PATH 미설정 시 본 모듈은 호출되지 않는다. 모델 로딩·추론
실패는 호출자(orchestrator)가 catch하여 rule 출력 그대로 fallback한다.
sklearn/scipy는 optional `ml` 그룹 의존성이므로 모든 import는 lazy.

2026-06-10 v3 feature 실험은 v2 대비 성능 보합이라 운영 기본 artifact는 v2를
유지한다. v3 artifact는 feature_dim=46에 맞춰 burst-aware feature를 사용하고,
v4 artifact는 feature_dim=58에 맞춰 magnitude + vy direction feature를 사용한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models.callback import AnalysisSegmentPayload
from app.services.vision._thresholds import DYNAMIC_TECHNIQUES

_EXPECTED_CLASSES = ["static", "dynamic"]

# 모델 artifact 캐시 (프로세스 수명 동안 1회 로드)
_artifact_cache: dict[str, dict[str, Any]] = {}


def _load_artifact(model_path: str) -> dict[str, Any]:
    """joblib artifact 로드 + shape 검증. 캐시됨."""
    cached = _artifact_cache.get(model_path)
    if cached is not None:
        return cached

    import joblib  # lazy: ml optional group

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"flow gate model not found: {model_path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"unexpected flow gate artifact shape: {model_path}")
    if artifact.get("classes") != _EXPECTED_CLASSES:
        raise ValueError(
            f"flow gate artifact classes mismatch: {artifact.get('classes')!r}"
        )
    _artifact_cache[model_path] = artifact
    return artifact


def predict_prob_dynamic(video_path: str, model_path: str) -> float:
    """영상에서 flow feature를 추출해 dynamic 확률을 반환.

    학습 시 build_flow_dataset.py와 동일한 전처리를 사용한다. v2 artifact(42-dim)는
    legacy feature를, v3 artifact(46-dim)는 burst-aware feature를 사용한다.

    Raises:
        FileNotFoundError: 모델 artifact 없음.
        ValueError: artifact shape 불일치, feature dim 불일치, 영상 디코딩 실패.
    """
    # lazy: scipy 의존 (ml optional group)
    from app.services.vision.flow_features import (
        FLOW_FEATURE_DIM,
        LEGACY_FLOW_FEATURE_DIM,
        V3_FLOW_FEATURE_DIM,
        extract_flow_series,
        extract_flow_stats,
        extract_flow_stats_legacy,
        extract_flow_stats_v3,
        remove_fall_end,
    )

    artifact = _load_artifact(model_path)
    expected_dim = int(artifact.get("feature_dim", 0))
    flow_series, _src_fps, _duration = extract_flow_series(Path(video_path))
    trimmed = remove_fall_end(flow_series)
    magnitude = trimmed[:, 0] if trimmed.ndim == 2 else trimmed
    if expected_dim == LEGACY_FLOW_FEATURE_DIM:
        features = extract_flow_stats_legacy(magnitude)
    elif expected_dim == V3_FLOW_FEATURE_DIM:
        features = extract_flow_stats_v3(magnitude)
    elif expected_dim == FLOW_FEATURE_DIM or expected_dim == 0:
        features = extract_flow_stats(trimmed)
    else:
        raise ValueError(f"unsupported flow feature dim: {expected_dim}")

    expected_dim = expected_dim or int(features.shape[0])
    if features.shape[0] != expected_dim:
        raise ValueError(
            f"flow feature dim mismatch: got {features.shape[0]}, expected {expected_dim}"
        )

    model = artifact["model"]
    proba = model.predict_proba(features.reshape(1, -1))[0]
    # classes=["static","dynamic"] → index 1이 dynamic
    return float(proba[1])


def adjust_segments(
    segments: list[AnalysisSegmentPayload],
    prob_dynamic: float,
    *,
    static_threshold: float,
    dynamic_threshold: float,
    demote_confidence: float,
) -> list[AnalysisSegmentPayload]:
    """flow 판정을 prior로 segment 목록을 보정한 새 리스트를 반환.

    static 확신 구간에서만 약한 dynamic segment를 drop하고 sequence_index를
    재부여한다. 그 외 구간에서는 입력을 그대로 반환한다.
    """
    if prob_dynamic >= static_threshold:
        # dynamic 또는 uncertain — 개입 없음
        return segments

    kept: list[AnalysisSegmentPayload] = []
    for seg in segments:
        is_weak_dynamic = (
            seg.technique in DYNAMIC_TECHNIQUES
            and (seg.confidence or 0.0) < demote_confidence
        )
        if is_weak_dynamic:
            continue
        kept.append(seg)

    if len(kept) == len(segments):
        return segments

    return [
        seg.model_copy(update={"sequence_index": i}) for i, seg in enumerate(kept)
    ]


def apply_flow_gate(
    video_path: str,
    segments: list[AnalysisSegmentPayload],
    *,
    model_path: str,
    static_threshold: float,
    dynamic_threshold: float,
    demote_confidence: float,
) -> tuple[list[AnalysisSegmentPayload], float]:
    """flow 추론 + segment 보정. (보정된 segments, prob_dynamic) 반환.

    동기 / CPU-bound — orchestrator가 asyncio.to_thread로 감싸 호출한다.
    예외는 호출자가 catch하여 rule 출력으로 fallback.
    """
    prob_dynamic = predict_prob_dynamic(video_path, model_path)
    adjusted = adjust_segments(
        segments,
        prob_dynamic,
        static_threshold=static_threshold,
        dynamic_threshold=dynamic_threshold,
        demote_confidence=demote_confidence,
    )
    return adjusted, prob_dynamic

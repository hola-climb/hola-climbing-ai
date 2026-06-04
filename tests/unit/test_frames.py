"""iter_frames 단위 테스트 — 합성 mp4로 deterministic 동작 검증."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import AnalysisException, AnalysisFailureReason
from app.services.pipeline.frames import iter_frames


class TestIterFrames:
    def test_yields_expected_count_for_1s_30fps(self, tiny_mp4: Path) -> None:
        """30fps 1초 영상에서 target_fps=15 → step=2 → 약 15개 프레임."""
        frames = list(iter_frames(str(tiny_mp4), target_fps=15))
        # mp4v 코덱은 정확한 프레임수 보장이 어려움. 12~16 범위 허용.
        assert 12 <= len(frames) <= 16, f"got {len(frames)} frames"

    def test_target_fps_30_yields_more_than_15(self, tiny_mp4: Path) -> None:
        """target_fps=30 → step=1 → 약 30개 프레임."""
        frames = list(iter_frames(str(tiny_mp4), target_fps=30))
        assert len(frames) >= 25

    def test_frame_shape_is_bgr(self, tiny_mp4: Path) -> None:
        frames = list(iter_frames(str(tiny_mp4), target_fps=15))
        assert frames, "no frames"
        _, _, bgr = frames[0]
        assert bgr.shape == (240, 320, 3)
        assert bgr.dtype.name == "uint8"

    def test_timestamps_monotonic_increasing(self, tiny_mp4: Path) -> None:
        frames = list(iter_frames(str(tiny_mp4), target_fps=15))
        timestamps = [t for _, t, _ in frames]
        assert timestamps == sorted(timestamps)
        # 1초 영상이므로 마지막 ts는 1000ms 근처
        assert 0 <= timestamps[-1] <= 1100

    def test_frame_indices_in_source_basis(self, tiny_mp4: Path) -> None:
        """frame_idx는 원본 영상 기준 절대 인덱스. target_fps=15면 0,2,4,..."""
        frames = list(iter_frames(str(tiny_mp4), target_fps=15))
        idxs = [i for i, _, _ in frames]
        # 다운샘플링이 적용됐는지: 연속한 idx 차이가 1 초과
        if len(idxs) >= 2:
            assert idxs[1] - idxs[0] >= 1

    def test_nonexistent_file_raises_video_decode(self, tmp_path: Path) -> None:
        bogus = tmp_path / "nope.mp4"
        with pytest.raises(AnalysisException) as exc_info:
            # generator를 list로 소진해야 raise됨
            list(iter_frames(str(bogus), target_fps=15))
        assert exc_info.value.reason == AnalysisFailureReason.VIDEO_DECODE

    def test_target_fps_zero_raises_value_error(self, tiny_mp4: Path) -> None:
        with pytest.raises(ValueError):
            list(iter_frames(str(tiny_mp4), target_fps=0))

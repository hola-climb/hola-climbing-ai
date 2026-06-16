"""Video encoder dataset builder tests."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.build_video_encoder_dataset import (
    build_video_encoder_dataset,
    make_dinov2_embedding_fn,
    make_person_crop_fn,
    make_videomae_embedding_fn,
    sample_video_clips,
    sample_video_frames,
    select_burst_clip_starts,
)


def test_sample_video_frames_returns_fixed_rgb_frames(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(
        video,
        [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 255),
        ],
    )

    frames = sample_video_frames(video, num_frames=6, resize=(8, 6))

    assert frames.shape == (6, 6, 8, 3)
    assert frames.dtype == np.uint8
    first_pixel = frames[0, 0, 0]
    assert int(first_pixel[0]) > 240
    assert int(first_pixel[1]) < 10
    assert int(first_pixel[2]) < 10


def test_sample_video_clips_returns_even_contiguous_clips(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _write_video(
        video,
        [
            (0, 0, 20),
            (0, 0, 40),
            (0, 0, 60),
            (0, 0, 80),
            (0, 0, 100),
            (0, 0, 120),
        ],
    )

    clips = sample_video_clips(
        video,
        num_clips=2,
        frames_per_clip=3,
        frame_stride=1,
        resize=(8, 6),
    )

    assert clips.shape == (2, 3, 6, 8, 3)
    assert int(clips[0, 0, 0, 0, 0]) < int(clips[0, -1, 0, 0, 0])
    assert int(clips[1, 0, 0, 0, 0]) < int(clips[1, -1, 0, 0, 0])
    assert int(clips[1, 0, 0, 0, 0]) > int(clips[0, 0, 0, 0, 0])


def test_select_burst_clip_starts_prefers_non_overlapping_bursts_and_context() -> None:
    flow = np.full(120, 0.1, dtype=np.float32)
    flow[20:30] = 5.0
    flow[80:90] = 4.0

    starts = select_burst_clip_starts(
        flow,
        num_clips=3,
        clip_window_size=16,
    )

    assert len(starts) == 3
    burst_starts = starts[:2]
    assert any(15 <= start <= 25 for start in burst_starts)
    assert any(75 <= start <= 85 for start in burst_starts)
    assert abs(burst_starts[0] - burst_starts[1]) >= 16
    assert starts[-1] == 52


def test_make_person_crop_fn_selects_largest_person_box_and_resizes() -> None:
    frames = _gradient_clip(width=30, height=20, length=2)
    detector = _FakeDetector(
        [
            {
                "boxes": np.asarray([[2, 2, 8, 8], [10, 4, 26, 18]], dtype=np.float32),
                "labels": np.asarray([1, 1], dtype=np.int64),
                "scores": np.asarray([0.99, 0.80], dtype=np.float32),
            }
        ]
    )
    crop_fn = make_person_crop_fn(
        detector=detector,
        output_size=(8, 8),
        margin_ratio=0.0,
    )

    result = crop_fn(frames)

    assert result.frames.shape == (2, 8, 8, 3)
    assert result.used_fallback is False
    assert result.box == (10, 3, 26, 19)
    assert int(result.frames[..., 0].mean()) > 120


def test_make_person_crop_fn_falls_back_to_previous_box() -> None:
    frames = _gradient_clip(width=30, height=20, length=2)
    detector = _FakeDetector(
        [
            {
                "boxes": np.asarray([[10, 4, 26, 18]], dtype=np.float32),
                "labels": np.asarray([1], dtype=np.int64),
                "scores": np.asarray([0.9], dtype=np.float32),
            },
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int64),
                "scores": np.zeros(0, dtype=np.float32),
            },
        ]
    )
    crop_fn = make_person_crop_fn(
        detector=detector,
        output_size=(8, 8),
        margin_ratio=0.0,
    )

    first = crop_fn(frames)
    second = crop_fn(frames)

    assert first.used_fallback is False
    assert second.used_fallback is True
    assert second.box == first.box


def test_make_videomae_embedding_fn_mean_pools_hidden_state() -> None:
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def __call__(self, frames: list[np.ndarray], *, return_tensors: str) -> dict[str, object]:
            assert len(frames) == 4
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros((1, 4, 3, 224, 224), dtype=torch.float32)}

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

        def to(self, _device: object) -> FakeModel:
            return self

        def __call__(self, **_inputs: object) -> object:
            hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]], dtype=torch.float32)
            return SimpleNamespace(last_hidden_state=hidden)

    embed = make_videomae_embedding_fn(
        model_id="fake",
        device_name="cpu",
        processor=FakeProcessor(),
        model=FakeModel(),
    )
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)

    out = embed(frames)

    assert out.dtype == np.float32
    assert out.tolist() == [2.0, 4.0]


def test_make_dinov2_embedding_fn_mean_pools_frame_outputs() -> None:
    torch = pytest.importorskip("torch")

    class FakeProcessor:
        def __call__(self, *, images: list[np.ndarray], return_tensors: str) -> dict[str, object]:
            assert len(images) == 4
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros((4, 3, 224, 224), dtype=torch.float32)}

    class FakeModel:
        def eval(self) -> FakeModel:
            return self

        def to(self, _device: object) -> FakeModel:
            return self

        def __call__(self, **_inputs: object) -> object:
            pooled = torch.tensor(
                [[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]],
                dtype=torch.float32,
            )
            return SimpleNamespace(pooler_output=pooled)

    embed = make_dinov2_embedding_fn(
        model_id="fake",
        device_name="cpu",
        processor=FakeProcessor(),
        model=FakeModel(),
    )
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)

    out = embed(frames)

    assert out.dtype == np.float32
    assert out.tolist() == [4.0, 6.0]


def test_build_video_encoder_dataset_writes_npz_and_manifest(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    videos = tmp_path / "videos"
    out = tmp_path / "encoder_dataset"
    videos.mkdir()
    _write_labels(labels, [("A.mp4", 1), ("B.mp4", 0)])
    _write_video(videos / "A.mp4", [(0, 0, 255), (0, 255, 0)])
    _write_video(videos / "B.mp4", [(255, 0, 0), (255, 255, 255)])

    result = build_video_encoder_dataset(
        labels_csv=labels,
        videos_dir=videos,
        out_dir=out,
        embedding_fn=lambda frames: np.asarray(
            [frames.shape[0], frames.shape[1], frames.mean()],
            dtype=np.float32,
        ),
        encoder_model="fake_encoder",
        encoder_weights="fake_weights",
        num_frames=4,
        num_clips=2,
        frame_stride=1,
        resize=(8, 6),
    )

    assert result.written == 2
    assert result.reused == 0
    assert result.missing == []
    assert result.failed == []
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert data["x"].shape == (6,)
        assert int(data["label"]) == 1
        assert str(data["variant"]) == "video_encoder"
        assert str(data["encoder_model"]) == "fake_encoder"
        assert str(data["encoder_weights"]) == "fake_weights"
        assert int(data["num_clips"]) == 2
    with result.manifest_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["stem"] for row in rows] == ["A", "B"]
    assert {row["feature_dim"] for row in rows} == {"6"}
    assert {row["num_clips"] for row in rows} == {"2"}

    reused = build_video_encoder_dataset(
        labels_csv=labels,
        videos_dir=videos,
        out_dir=out,
        embedding_fn=lambda frames: np.asarray(
            [frames.shape[0], frames.shape[1], frames.mean()],
            dtype=np.float32,
        ),
        encoder_model="fake_encoder",
        encoder_weights="fake_weights",
        num_frames=4,
        num_clips=2,
        frame_stride=1,
        resize=(8, 6),
    )

    assert reused.written == 0
    assert reused.reused == 2
    with reused.manifest_path.open(encoding="utf-8", newline="") as f:
        reused_rows = list(csv.DictReader(f))
    assert {row["crop_fallback_count"] for row in reused_rows} == {"0"}


def test_build_video_encoder_dataset_rebuilds_stale_npz(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    videos = tmp_path / "videos"
    out = tmp_path / "encoder_dataset"
    videos.mkdir()
    _write_labels(labels, [("A.mp4", 1)])
    _write_video(videos / "A.mp4", [(0, 0, 255), (0, 255, 0)])

    first = build_video_encoder_dataset(
        labels_csv=labels,
        videos_dir=videos,
        out_dir=out,
        embedding_fn=lambda frames: np.asarray([frames.shape[0]], dtype=np.float32),
        encoder_model="fake_encoder",
        encoder_weights="fake_weights",
        num_frames=4,
        resize=(8, 6),
    )
    second = build_video_encoder_dataset(
        labels_csv=labels,
        videos_dir=videos,
        out_dir=out,
        embedding_fn=lambda frames: np.asarray([frames.shape[0]], dtype=np.float32),
        encoder_model="fake_encoder",
        encoder_weights="fake_weights",
        num_frames=5,
        resize=(8, 6),
    )

    assert first.written == 1
    assert second.written == 1
    assert second.reused == 0
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert int(data["num_frames"]) == 5
        assert data["x"].tolist() == [5.0]


def _write_labels(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for filename, label in rows:
            writer.writerow({"filename": filename, "label": label})


def _write_video(path: Path, bgr_frames: list[tuple[int, int, int]]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 12),
    )
    for color in bgr_frames:
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[:, :] = color
        writer.write(frame)
    writer.release()


def _gradient_clip(*, width: int, height: int, length: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = x[None, :]
    frame[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    return np.stack([frame.copy() for _ in range(length)], axis=0)


class _FakeDetector:
    def __init__(self, outputs: list[dict[str, np.ndarray]]) -> None:
        self._outputs = outputs
        self.calls = 0

    def __call__(self, _images: object) -> list[dict[str, np.ndarray]]:
        output = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return [output]

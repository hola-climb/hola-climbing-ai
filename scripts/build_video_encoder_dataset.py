"""Build pretrained video-encoder embedding datasets from labeled videos.

Usage:
    uv run python scripts/build_video_encoder_dataset.py \
        --labels data/review/labels_gcs_flow_reviewed_round3.csv \
        --videos-dir data/gcs_cache/videos/original \
        --out data/video_encoder_dataset/gcs_r3d18_k400_v1
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from app.services.vision.flow_features import extract_flow_magnitude
from app.services.vision.pose_dataset import match_labeled_videos

EmbeddingFn = Callable[[NDArray[np.uint8]], NDArray[np.float32]]
ClipTransformFn = Callable[[NDArray[np.uint8]], "ClipTransformResult"]


@dataclass(frozen=True)
class ClipTransformResult:
    frames: NDArray[np.uint8]
    used_fallback: bool
    box: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class BuildVideoEncoderDatasetResult:
    written: int
    reused: int
    missing: list[str]
    failed: list[tuple[str, str]]
    manifest_path: Path


def build_video_encoder_dataset(
    *,
    labels_csv: Path,
    videos_dir: Path,
    out_dir: Path,
    embedding_fn: EmbeddingFn,
    encoder_model: str,
    encoder_weights: str,
    num_frames: int = 16,
    num_clips: int = 1,
    frame_stride: int = 1,
    resize: tuple[int, int] = (171, 128),
    sampling: str = "uniform",
    clip_span_sec: float = 2.0,
    clip_transform_fn: ClipTransformFn | None = None,
    overwrite: bool = False,
) -> BuildVideoEncoderDatasetResult:
    """Build one compressed `.npz` encoder embedding file per labeled video."""
    matched, missing = match_labeled_videos(labels_csv, videos_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.with_name(f"{out_dir.name}_manifest.csv")

    written = 0
    reused = 0
    failed: list[tuple[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    total = len(matched)
    for index, item in enumerate(matched, start=1):
        out_path = out_dir / f"{item.stem}.npz"
        try:
            if out_path.exists() and not overwrite:
                if _npz_matches_config(
                    out_path,
                    encoder_model=encoder_model,
                    encoder_weights=encoder_weights,
                    num_frames=num_frames,
                    num_clips=num_clips,
                    frame_stride=frame_stride,
                    sampling=sampling,
                    person_crop=clip_transform_fn is not None,
                ):
                    print(f"[encoder] {index}/{total} reuse {item.stem}", flush=True)
                    feature_dim, crop_fallback_count = _cached_manifest_values_from_npz(out_path)
                    reused += 1
                    manifest_rows.append(
                        _manifest_row(
                            stem=item.stem,
                            label=item.label,
                            source_path=item.video_path,
                            out_path=out_path,
                            encoder_model=encoder_model,
                            encoder_weights=encoder_weights,
                            num_frames=num_frames,
                            num_clips=num_clips,
                            frame_stride=frame_stride,
                            sampling=sampling,
                            person_crop=clip_transform_fn is not None,
                            crop_fallback_count=crop_fallback_count,
                            feature_dim=feature_dim,
                            reused=True,
                        )
                    )
                    continue
                print(f"[encoder] {index}/{total} rebuild {item.stem} (metadata mismatch)", flush=True)
            else:
                print(f"[encoder] {index}/{total} encode {item.stem}", flush=True)

            clips = _sample_clips(
                item.video_path,
                sampling=sampling,
                num_clips=num_clips,
                frames_per_clip=num_frames,
                frame_stride=frame_stride,
                clip_span_sec=clip_span_sec,
                resize=resize,
            )
            crop_fallback_count = 0
            transformed_clips: list[NDArray[np.uint8]] = []
            for clip in clips:
                if clip_transform_fn is None:
                    transformed_clips.append(clip)
                    continue
                transformed = clip_transform_fn(clip)
                crop_fallback_count += int(transformed.used_fallback)
                transformed_clips.append(transformed.frames)
            embedding = _pool_clip_embeddings([embedding_fn(clip) for clip in transformed_clips]).astype(np.float32)
            if embedding.ndim != 1:
                raise ValueError(f"embedding must be 1D, got {embedding.shape}")
            np.savez_compressed(
                out_path,
                x=embedding,
                label=np.asarray(item.label, dtype=np.int64),
                stem=np.asarray(item.stem),
                source_path=np.asarray(str(item.video_path)),
                variant=np.asarray("video_encoder"),
                encoder_model=np.asarray(encoder_model),
                encoder_weights=np.asarray(encoder_weights),
                num_frames=np.asarray(num_frames, dtype=np.int64),
                num_clips=np.asarray(num_clips, dtype=np.int64),
                frame_stride=np.asarray(frame_stride, dtype=np.int64),
                sampling=np.asarray(sampling),
                person_crop=np.asarray(clip_transform_fn is not None),
                crop_fallback_count=np.asarray(crop_fallback_count, dtype=np.int64),
                feature_dim=np.asarray(embedding.shape[0], dtype=np.int64),
            )
            written += 1
            manifest_rows.append(
                _manifest_row(
                    stem=item.stem,
                    label=item.label,
                    source_path=item.video_path,
                    out_path=out_path,
                    encoder_model=encoder_model,
                    encoder_weights=encoder_weights,
                    num_frames=num_frames,
                    num_clips=num_clips,
                    frame_stride=frame_stride,
                    sampling=sampling,
                    person_crop=clip_transform_fn is not None,
                    crop_fallback_count=crop_fallback_count,
                    feature_dim=int(embedding.shape[0]),
                    reused=False,
                )
            )
        except Exception as exc:
            failed.append((item.stem, repr(exc)))

    _write_manifest(manifest_path, manifest_rows)
    return BuildVideoEncoderDatasetResult(
        written=written,
        reused=reused,
        missing=missing,
        failed=failed,
        manifest_path=manifest_path,
    )


def sample_video_frames(
    video_path: Path,
    *,
    num_frames: int,
    resize: tuple[int, int],
) -> NDArray[np.uint8]:
    """Uniformly sample RGB frames from a video as `(T, H, W, C)`."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            frames = _read_all_frames(cap, resize=resize)
            if not frames:
                raise ValueError("video has no readable frames")
            indices = np.linspace(0, len(frames) - 1, num=num_frames).round().astype(np.int64)
            return np.stack([frames[int(idx)] for idx in indices], axis=0).astype(np.uint8)

        indices = np.linspace(0, max(0, total_frames - 1), num=num_frames).round().astype(np.int64)
        sampled: list[NDArray[np.uint8]] = []
        last_frame: NDArray[np.uint8] | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if last_frame is None:
                    continue
                sampled.append(last_frame)
                continue
            rgb = _resize_bgr_to_rgb(cast(NDArray[np.uint8], frame), resize=resize)
            sampled.append(rgb)
            last_frame = rgb
    finally:
        cap.release()

    if not sampled:
        raise ValueError("video has no readable frames")
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])
    return np.stack(sampled[:num_frames], axis=0).astype(np.uint8)


def sample_video_clips(
    video_path: Path,
    *,
    num_clips: int,
    frames_per_clip: int,
    frame_stride: int,
    resize: tuple[int, int],
) -> NDArray[np.uint8]:
    """Uniformly sample contiguous RGB clips as `(N, T, H, W, C)`."""
    if num_clips < 1:
        raise ValueError("num_clips must be >= 1")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be >= 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    try:
        all_frames = _read_all_frames(cap, resize=resize)
    finally:
        cap.release()
    if not all_frames:
        raise ValueError("video has no readable frames")

    clip_span = (frames_per_clip - 1) * frame_stride + 1
    max_start = max(0, len(all_frames) - clip_span)
    starts = np.linspace(0, max_start, num=num_clips).round().astype(np.int64)
    clips: list[NDArray[np.uint8]] = []
    for start in starts:
        frames: list[NDArray[np.uint8]] = []
        for offset in range(frames_per_clip):
            idx = min(len(all_frames) - 1, int(start) + offset * frame_stride)
            frames.append(all_frames[idx])
        clips.append(np.stack(frames, axis=0).astype(np.uint8))
    return np.stack(clips, axis=0).astype(np.uint8)


def sample_burst_guided_clips(
    video_path: Path,
    *,
    num_clips: int,
    frames_per_clip: int,
    frame_stride: int,
    clip_span_sec: float,
    resize: tuple[int, int],
) -> NDArray[np.uint8]:
    """Sample clips around the strongest optical-flow bursts plus one context clip."""
    if num_clips < 1:
        raise ValueError("num_clips must be >= 1")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be >= 1")
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    flow_mag, src_fps, _duration = extract_flow_magnitude(video_path)
    clip_window_size = max(1, int(clip_span_sec * 30))
    starts = select_burst_clip_starts(flow_mag, num_clips=num_clips, clip_window_size=clip_window_size)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"failed to open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise ValueError("video has no readable frames")
        flow_to_video = max(src_fps / 30.0, 1.0 / 30.0)
        clips = [
            _read_clip_at_frame(
                cap,
                start_frame=round(start * flow_to_video),
                total_frames=total_frames,
                frames_per_clip=frames_per_clip,
                frame_stride=frame_stride,
                resize=resize,
            )
            for start in starts
        ]
    finally:
        cap.release()
    return np.stack(clips, axis=0).astype(np.uint8)


def select_burst_clip_starts(
    flow_mag: NDArray[np.float32],
    *,
    num_clips: int,
    clip_window_size: int,
) -> list[int]:
    """Select non-overlapping burst windows, reserving the final clip for context."""
    signal = np.asarray(flow_mag, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"flow_mag must be 1D, got {signal.shape}")
    if num_clips < 1:
        raise ValueError("num_clips must be >= 1")
    if clip_window_size < 1:
        raise ValueError("clip_window_size must be >= 1")
    max_start = max(0, len(signal) - clip_window_size)
    context_start = max_start // 2
    if len(signal) <= clip_window_size or num_clips == 1:
        return [context_start for _ in range(num_clips)]

    smoothed = _moving_average(signal, window=15)
    window_means = np.asarray(
        [smoothed[start : start + clip_window_size].mean() for start in range(max_start + 1)],
        dtype=np.float32,
    )
    masked = window_means.copy()
    burst_starts: list[int] = []
    for _ in range(num_clips - 1):
        if not np.isfinite(masked).any():
            break
        start = int(np.nanargmax(masked))
        burst_starts.append(start)
        mask_start = max(0, start - clip_window_size)
        mask_end = min(len(masked), start + clip_window_size + 1)
        masked[mask_start:mask_end] = np.nan

    while len(burst_starts) < num_clips - 1:
        burst_starts.append(context_start)
    return [*burst_starts, context_start]


def make_person_crop_fn(
    *,
    detector: Any | None = None,
    device_name: str = "auto",
    score_threshold: float = 0.5,
    margin_ratio: float = 0.30,
    output_size: tuple[int, int] = (224, 224),
) -> ClipTransformFn:
    """Create a clip transform that crops every frame to the detected person box."""
    torch_module: Any | None = None
    device: Any | None = None
    if detector is None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        torch_module = torch
        device = _resolve_device(torch, device_name)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        detector.eval().to(device)

    last_box: tuple[int, int, int, int] | None = None

    def crop_clip(frames: NDArray[np.uint8]) -> ClipTransformResult:
        nonlocal last_box
        if frames.ndim != 4:
            raise ValueError(f"frames must have shape (T, H, W, C), got {frames.shape}")
        center_frame = frames[len(frames) // 2]
        raw_box = _detect_person_box(
            detector=detector,
            frame=center_frame,
            torch_module=torch_module,
            device=device,
            score_threshold=score_threshold,
        )
        used_fallback = raw_box is None
        if raw_box is None:
            box = last_box
        else:
            height, width = center_frame.shape[:2]
            box = _expand_square_box(raw_box, width=width, height=height, margin_ratio=margin_ratio)
            last_box = box

        if box is None:
            resized = np.stack([cv2.resize(frame, output_size) for frame in frames], axis=0).astype(np.uint8)
            return ClipTransformResult(frames=resized, used_fallback=True, box=None)

        x1, y1, x2, y2 = box
        cropped = np.stack(
            [cv2.resize(frame[y1:y2, x1:x2], output_size) for frame in frames],
            axis=0,
        ).astype(np.uint8)
        return ClipTransformResult(frames=cropped, used_fallback=used_fallback, box=box)

    return crop_clip


def make_torchvision_embedding_fn(
    *,
    encoder_model: str,
    weights_name: str,
    device_name: str,
) -> EmbeddingFn:
    """Create a torchvision video encoder embedding function."""
    import torch
    from torch import nn
    from torchvision.models import video as video_models

    device = _resolve_device(torch, device_name)
    if encoder_model != "r3d_18":
        raise ValueError(f"unsupported encoder_model: {encoder_model}")
    weights_enum = video_models.R3D_18_Weights
    weights = weights_enum.DEFAULT if weights_name == "DEFAULT" else weights_enum[weights_name]
    model = video_models.r3d_18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()

    def embed(frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0)
        tensor = transform(tensor).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model(tensor).detach().cpu().numpy()[0].astype(np.float32)
        return cast(NDArray[np.float32], out)

    return embed


def make_videomae_embedding_fn(
    *,
    model_id: str,
    device_name: str,
    processor: Any | None = None,
    model: Any | None = None,
) -> EmbeddingFn:
    """Create a HuggingFace VideoMAE embedding function."""
    import torch

    device = _resolve_device(torch, device_name)
    if processor is None or model is None:
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        processor = VideoMAEImageProcessor.from_pretrained(model_id)
        model = VideoMAEModel.from_pretrained(model_id)
    model.eval().to(device)

    def embed(frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        inputs = processor([frame for frame in frames], return_tensors="pt")
        tensor_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**tensor_inputs)
            pooled = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0].astype(np.float32)
        return cast(NDArray[np.float32], pooled)

    return embed


def make_dinov2_embedding_fn(
    *,
    model_id: str,
    device_name: str,
    processor: Any | None = None,
    model: Any | None = None,
) -> EmbeddingFn:
    """Create a HuggingFace DINOv2 frame embedding function."""
    import torch

    device = _resolve_device(torch, device_name)
    if processor is None or model is None:
        from transformers import AutoImageProcessor, AutoModel

        processor = cast(Any, AutoImageProcessor).from_pretrained(model_id)
        model = cast(Any, AutoModel).from_pretrained(model_id)
    model.eval().to(device)

    def embed(frames: NDArray[np.uint8]) -> NDArray[np.float32]:
        inputs = processor(images=[frame for frame in frames], return_tensors="pt")
        tensor_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**tensor_inputs)
            pooled_output = getattr(outputs, "pooler_output", None)
            if pooled_output is None:
                pooled_output = outputs.last_hidden_state[:, 0]
            pooled = pooled_output.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return cast(NDArray[np.float32], pooled)

    return embed


def make_embedding_fn(
    *,
    encoder_model: str,
    encoder_weights: str,
    device_name: str,
) -> EmbeddingFn:
    """Create an embedding function by encoder model name."""
    if encoder_model == "r3d_18":
        return make_torchvision_embedding_fn(
            encoder_model=encoder_model,
            weights_name=encoder_weights,
            device_name=device_name,
        )
    if encoder_model == "videomae":
        model_id = (
            "MCG-NJU/videomae-base-finetuned-kinetics"
            if encoder_weights == "DEFAULT"
            else encoder_weights
        )
        return make_videomae_embedding_fn(model_id=model_id, device_name=device_name)
    if encoder_model == "dinov2":
        model_id = "facebook/dinov2-base" if encoder_weights == "DEFAULT" else encoder_weights
        return make_dinov2_embedding_fn(model_id=model_id, device_name=device_name)
    raise ValueError(f"unsupported encoder_model: {encoder_model}")


def _sample_clips(
    video_path: Path,
    *,
    sampling: str,
    num_clips: int,
    frames_per_clip: int,
    frame_stride: int,
    clip_span_sec: float,
    resize: tuple[int, int],
) -> NDArray[np.uint8]:
    if sampling == "uniform":
        return sample_video_clips(
            video_path,
            num_clips=num_clips,
            frames_per_clip=frames_per_clip,
            frame_stride=frame_stride,
            resize=resize,
        )
    if sampling == "burst":
        return sample_burst_guided_clips(
            video_path,
            num_clips=num_clips,
            frames_per_clip=frames_per_clip,
            frame_stride=frame_stride,
            clip_span_sec=clip_span_sec,
            resize=resize,
        )
    raise ValueError(f"unsupported sampling: {sampling}")


def _read_clip_at_frame(
    cap: Any,
    *,
    start_frame: int,
    total_frames: int,
    frames_per_clip: int,
    frame_stride: int,
    resize: tuple[int, int],
) -> NDArray[np.uint8]:
    frames: list[NDArray[np.uint8]] = []
    last_frame: NDArray[np.uint8] | None = None
    for offset in range(frames_per_clip):
        frame_idx = min(total_frames - 1, max(0, start_frame + offset * frame_stride))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            if last_frame is not None:
                frames.append(last_frame)
            continue
        rgb = _resize_bgr_to_rgb(cast(NDArray[np.uint8], frame), resize=resize)
        frames.append(rgb)
        last_frame = rgb
    if not frames:
        raise ValueError("clip has no readable frames")
    while len(frames) < frames_per_clip:
        frames.append(frames[-1])
    return np.stack(frames[:frames_per_clip], axis=0).astype(np.uint8)


def _moving_average(signal: NDArray[np.float32], *, window: int) -> NDArray[np.float32]:
    if len(signal) < 3:
        return signal.astype(np.float32)
    window_size = max(1, min(window, len(signal)))
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return cast(NDArray[np.float32], np.convolve(signal, kernel, mode="same").astype(np.float32))


def _detect_person_box(
    *,
    detector: Any,
    frame: NDArray[np.uint8],
    torch_module: Any | None,
    device: Any | None,
    score_threshold: float,
) -> tuple[float, float, float, float] | None:
    if torch_module is None:
        outputs = detector([frame])
    else:
        tensor = torch_module.from_numpy(frame).permute(2, 0, 1).float().div(255.0).to(device)
        with torch_module.inference_mode():
            outputs = detector([tensor])
    if not outputs:
        return None
    output = outputs[0]
    boxes = _to_numpy(output.get("boxes", np.zeros((0, 4), dtype=np.float32))).astype(np.float32)
    labels = _to_numpy(output.get("labels", np.zeros(0, dtype=np.int64))).astype(np.int64)
    scores = _to_numpy(output.get("scores", np.zeros(0, dtype=np.float32))).astype(np.float32)
    if len(boxes) == 0:
        return None
    keep = (labels == 1) & (scores >= score_threshold)
    if not np.any(keep):
        return None
    kept_boxes = boxes[keep]
    areas = np.maximum(0.0, kept_boxes[:, 2] - kept_boxes[:, 0]) * np.maximum(0.0, kept_boxes[:, 3] - kept_boxes[:, 1])
    best = kept_boxes[int(np.argmax(areas))]
    return (float(best[0]), float(best[1]), float(best[2]), float(best[3]))


def _expand_square_box(
    box: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = round(max(box_width, box_height) * (1.0 + 2.0 * margin_ratio))
    side = max(1, min(side, width, height))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = round(cx - side / 2.0)
    top = round(cy - side / 2.0)
    left = min(max(0, left), max(0, width - side))
    top = min(max(0, top), max(0, height - side))
    return (left, top, left + side, top + side)


def _to_numpy(value: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pool_clip_embeddings(embeddings: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    if not embeddings:
        raise ValueError("embeddings must not be empty")
    stacked = np.stack(embeddings, axis=0).astype(np.float32)
    if stacked.ndim != 2:
        raise ValueError(f"clip embeddings must stack to 2D, got {stacked.shape}")
    if stacked.shape[0] == 1:
        return cast(NDArray[np.float32], stacked[0].astype(np.float32))
    pooled = np.concatenate([stacked.mean(axis=0), stacked.std(axis=0)], axis=0).astype(np.float32)
    return cast(NDArray[np.float32], pooled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--encoder-model", default="r3d_18")
    parser.add_argument("--encoder-weights", default="DEFAULT")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-clips", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=171)
    parser.add_argument("--resize-height", type=int, default=128)
    parser.add_argument("--sampling", choices=["uniform", "burst"], default="uniform")
    parser.add_argument("--clip-span-sec", type=float, default=2.0)
    parser.add_argument("--person-crop", action="store_true")
    parser.add_argument("--crop-score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    embedding_fn = make_embedding_fn(
        encoder_model=args.encoder_model,
        encoder_weights=args.encoder_weights,
        device_name=args.device,
    )
    clip_transform_fn = (
        make_person_crop_fn(
            device_name=args.device,
            score_threshold=args.crop_score_threshold,
        )
        if args.person_crop
        else None
    )
    result = build_video_encoder_dataset(
        labels_csv=args.labels,
        videos_dir=args.videos_dir,
        out_dir=args.out,
        embedding_fn=embedding_fn,
        encoder_model=args.encoder_model,
        encoder_weights=args.encoder_weights,
        num_frames=args.num_frames,
        num_clips=args.num_clips,
        frame_stride=args.frame_stride,
        resize=(args.resize_width, args.resize_height),
        sampling=args.sampling,
        clip_span_sec=args.clip_span_sec,
        clip_transform_fn=clip_transform_fn,
        overwrite=args.overwrite,
    )
    print(
        "[summary] "
        f"written={result.written} reused={result.reused} missing={len(result.missing)} "
        f"failed={len(result.failed)} manifest={result.manifest_path}"
    )
    if result.missing:
        print("[missing sample]", ", ".join(result.missing[:10]))
    if result.failed:
        print("[failed sample]", result.failed[:5])
    return 0 if result.written + result.reused > 0 else 1


def _read_all_frames(
    cap: Any,
    *,
    resize: tuple[int, int],
) -> list[NDArray[np.uint8]]:
    frames: list[NDArray[np.uint8]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(_resize_bgr_to_rgb(cast(NDArray[np.uint8], frame), resize=resize))
    return frames


def _resize_bgr_to_rgb(frame: NDArray[np.uint8], *, resize: tuple[int, int]) -> NDArray[np.uint8]:
    resized = cv2.resize(frame, resize)
    return cast(NDArray[np.uint8], cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def _resolve_device(torch_module: Any, device_name: str) -> Any:
    if device_name != "auto":
        return torch_module.device(device_name)
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _cached_manifest_values_from_npz(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        crop_fallback_count = int(data["crop_fallback_count"]) if "crop_fallback_count" in data else -1
        return int(data["x"].shape[0]), crop_fallback_count


def _npz_matches_config(
    path: Path,
    *,
    encoder_model: str,
    encoder_weights: str,
    num_frames: int,
    num_clips: int,
    frame_stride: int,
    sampling: str,
    person_crop: bool,
) -> bool:
    expected: dict[str, object] = {
        "encoder_model": encoder_model,
        "encoder_weights": encoder_weights,
        "num_frames": num_frames,
        "num_clips": num_clips,
        "frame_stride": frame_stride,
        "sampling": sampling,
        "person_crop": person_crop,
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            for key, expected_value in expected.items():
                if key not in data:
                    return False
                actual_value = np.asarray(data[key]).item()
                if isinstance(expected_value, bool):
                    if bool(actual_value) != expected_value:
                        return False
                    continue
                if isinstance(expected_value, int):
                    if int(actual_value) != expected_value:
                        return False
                    continue
                if str(actual_value) != expected_value:
                    return False
    except Exception:
        return False
    return True


def _manifest_row(
    *,
    stem: str,
    label: int,
    source_path: Path,
    out_path: Path,
    encoder_model: str,
    encoder_weights: str,
    num_frames: int,
    num_clips: int,
    frame_stride: int,
    sampling: str,
    person_crop: bool,
    crop_fallback_count: int,
    feature_dim: int,
    reused: bool,
) -> dict[str, object]:
    return {
        "stem": stem,
        "label": label,
        "variant": "video_encoder",
        "encoder_model": encoder_model,
        "encoder_weights": encoder_weights,
        "num_frames": num_frames,
        "num_clips": num_clips,
        "frame_stride": frame_stride,
        "sampling": sampling,
        "person_crop": str(person_crop),
        "crop_fallback_count": crop_fallback_count,
        "feature_dim": feature_dim,
        "source_path": str(source_path),
        "out_path": str(out_path),
        "reused": str(reused),
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "stem",
        "label",
        "variant",
        "encoder_model",
        "encoder_weights",
        "num_frames",
        "num_clips",
        "frame_stride",
        "sampling",
        "person_crop",
        "crop_fallback_count",
        "feature_dim",
        "source_path",
        "out_path",
        "reused",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

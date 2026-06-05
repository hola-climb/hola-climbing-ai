# Pose Dynamic Static Classifier Design

## Goal

Add a video-level `dynamic`/`static` binary classifier that learns from MediaPipe Pose sequences instead of rule thresholds.

## Scope

- Input labels come from `/Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv`.
- Input videos are matched by filename stem from `/Users/minjoun/Movies/Original`.
- The first learned model predicts only video-level `dynamic` vs `static`.
- Segment-level technique labels remain handled by the existing heuristic pipeline.

## Architecture

The pipeline has three independent parts:

1. Pose dataset cache builder: video -> MediaPipe Pose frames -> normalized/resampled `.npz`.
2. Sequence trainer: `.npz` cache -> GRU binary classifier checkpoint.
3. Inference helper: pose sequence + checkpoint -> `dynamic`/`static` probability.

The existing worker does not switch to the learned model by default. This keeps the current heuristic flow stable while the model is trained and evaluated.

## Data Format

Each cached sample stores:

- `x`: `float32` array shaped `(target_frames, 132)` where `132 = 33 landmarks * 4 values`.
- `label`: `int64`, `0=static`, `1=dynamic`.
- `stem`: source filename stem.
- `source_path`: source video path.
- `raw_pose_frames`: original number of detected pose frames.

## Model

Use a small GRU classifier:

- input size: `132`
- hidden size: default `64`
- layers: default `1`
- output: one logit, sigmoid probability means dynamic probability

PyTorch is optional at app import time. Training/inference commands fail with a clear install message if `torch` is missing.

## Testing

- Unit-test label loading, video matching, pose normalization, resampling, and model input shape without requiring `torch`.
- Unit-test model inference missing-dependency behavior.
- Keep existing `pytest`, `ruff`, and `mypy` gates green.

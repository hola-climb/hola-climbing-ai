# Pose Dynamic Static Diagnostics Design

## Goal

Improve the learned dynamic/static workflow by making it diagnosable and by comparing raw pose features against motion-aware pose features.

## Current Problem

The first full run used 208 cached samples and reached validation accuracy `0.5122`, below the majority baseline around `0.538`. The label distribution is not severely imbalanced, so the next useful step is to identify whether the bottleneck is the feature representation, split variance, low-quality samples, or model capacity.

## Design

Add a feature transformation layer that can produce either:

- `raw`: existing `(x, y, z, visibility)` flattened pose sequence, shape `(T, 132)`.
- `motion`: hip-centered, torso-scale-normalized coordinates plus visibility, velocity, acceleration, and speed, shape `(T, 363)`.

Keep the GRU classifier but make `input_size` checkpoint metadata explicit so old raw checkpoints and new motion checkpoints both load correctly.

Enhance `scripts/train_pose_sequence.py` to:

- filter samples with too few detected pose frames via `--min-raw-pose-frames`;
- train with `--feature-set raw|motion`;
- print train and validation metrics each epoch;
- support `--folds N` stratified k-fold evaluation;
- write prediction-level CSV and metrics JSON under `--report-dir`.

## Trade-offs

Motion features add engineered input structure, but the model still learns the final decision. This keeps the workflow aligned with the goal of learned classification while reducing the amount of physics the model must discover from only ~200 videos.

K-fold evaluation costs more training time, but on 208 samples it is still cheap and reduces the chance of trusting a lucky or unlucky 41-sample validation split.

Filtering low-pose samples removes noisy data but may hide a real production case. For training, this is acceptable; production should later handle low-confidence pose extraction explicitly.

## Success Criteria

- Existing tests keep passing.
- New unit tests cover motion feature shape/invariance, checkpoint input size metadata, and k-fold/data filtering helpers.
- A motion-feature training run can produce `models/pose_dynamic_static_motion.pt` and report artifacts.
- The result is reported honestly, including if it does not improve over raw pose.

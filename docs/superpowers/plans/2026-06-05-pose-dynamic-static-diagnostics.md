# Pose Dynamic Static Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add motion-aware features, k-fold evaluation, and diagnostic reports for the dynamic/static pose classifier.

**Architecture:** Keep cached `.npz` files as raw pose features. Add a deterministic feature transformer used by training, then make the training script report per-split predictions and aggregate metrics. Preserve raw mode and checkpoint compatibility.

**Tech Stack:** Python, NumPy, PyTorch optional `ml` group, pytest, CSV/JSON reports.

---

### Task 1: Motion Feature Transformer

**Files:**
- Create: `app/services/vision/pose_features.py`
- Test: `tests/unit/test_pose_features.py`

- [x] Add failing tests for `raw` passthrough and `motion` feature shape.
- [x] Add failing test that translated/scaled poses produce equivalent motion features.
- [x] Implement `feature_size()` and `prepare_pose_features()`.
- [x] Run `uv run pytest tests/unit/test_pose_features.py`.

### Task 2: Checkpoint Metadata

**Files:**
- Modify: `app/services/vision/model_classifier.py`
- Test: `tests/unit/test_model_classifier.py`

- [x] Add failing test that checkpoint roundtrip preserves `input_size` and `feature_set`.
- [x] Add backward-compatible defaults for old checkpoints.
- [x] Run `uv run pytest tests/unit/test_model_classifier.py`.

### Task 3: Training Diagnostics

**Files:**
- Modify: `scripts/train_pose_sequence.py`
- Test: `tests/unit/test_train_pose_sequence.py`

- [x] Add failing tests for low-frame filtering and stratified k-fold split coverage.
- [x] Add failing tests for metrics/report payload helpers.
- [x] Implement `--feature-set`, `--min-raw-pose-frames`, `--folds`, and `--report-dir`.
- [x] Run `uv run pytest tests/unit/test_train_pose_sequence.py`.

### Task 4: Re-run And Compare

**Files:**
- Modify: `README.md`
- Local artifacts: `models/pose_dynamic_static_motion.pt`, `models/reports/`

- [x] Document the diagnostic training commands.
- [x] Run full verification: `uv run pytest`, `uv run ruff check app tests scripts`, `uv run mypy app`.
- [x] Run `motion` training with 5-fold evaluation and compare against raw baseline.

**Result:** raw 5-fold balanced accuracy mean `0.6160`; motion 5-fold balanced accuracy mean `0.5672`. Motion features overfit quickly and should not be promoted as-is.

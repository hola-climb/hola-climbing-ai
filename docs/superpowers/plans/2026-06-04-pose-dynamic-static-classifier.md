# Pose Dynamic Static Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a video-level `dynamic`/`static` learned classifier over MediaPipe Pose sequences.

**Architecture:** Add reusable pose dataset preprocessing helpers, a dataset cache builder script, a PyTorch GRU trainer script, and an optional inference helper. Keep the existing heuristic worker unchanged by default.

**Tech Stack:** Python 3.11+, NumPy, MediaPipe/OpenCV via existing pipeline, optional PyTorch for training/inference.

---

### Task 1: Pose Dataset Helpers

**Files:**
- Create: `app/services/vision/pose_dataset.py`
- Test: `tests/unit/test_pose_dataset.py`

- [x] Write tests for label parsing, video matching, normalization, resampling, and model input shape.
- [x] Implement pure NumPy helpers.
- [x] Run `uv run pytest tests/unit/test_pose_dataset.py`.

### Task 2: Dataset Cache Builder

**Files:**
- Create: `scripts/build_pose_dataset.py`

- [x] Implement CLI that reads labels, matches videos, extracts pose, and writes `.npz` files.
- [x] Support `--limit`, `--target-fps`, and `--target-frames`.
- [x] Keep failed videos non-fatal and summarize counts.

### Task 3: GRU Trainer And Inference Helper

**Files:**
- Create: `app/services/vision/model_classifier.py`
- Create: `scripts/train_pose_sequence.py`
- Test: `tests/unit/test_model_classifier.py`

- [x] Add optional PyTorch loader with clear install error.
- [x] Implement GRU model factory, checkpoint load, and probability prediction.
- [x] Implement trainer with stratified split, metrics, and best checkpoint saving.

### Task 4: Verification And Docs

**Files:**
- Modify: `README.md`

- [x] Document dataset build and training commands.
- [x] Run `uv run pytest`, `uv run ruff check app tests scripts`, and `uv run mypy app`.

**Verification:** `uv run pytest` 72 passed, `uv run ruff check app tests scripts` passed, `uv run mypy app` passed.

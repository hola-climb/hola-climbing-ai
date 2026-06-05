# Tabular Flow Baseline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `/Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind` tabular pose and optical-flow experiments into `hola-climbing-ai`, then compare them fairly against the current GRU baseline.

**Architecture:** Treat `/hola_ind/feature_extraction.py` as a strong baseline, not the final direction. Implement deterministic feature extractors, dataset builders, and sklearn training reports so pose tabular, optical flow, and pose+flow fusion can be evaluated with identical labels and split policies. Keep production worker behavior unchanged until the comparison proves a candidate is stable.

**Tech Stack:** Python, NumPy, OpenCV, SciPy, scikit-learn, joblib, pytest, CSV/JSON reports.

---

## Current Facts To Preserve

- Current GRU QA baseline: `models/reports/pose_dynamic_static_raw_qa_kfold_metrics.json`
  - samples `205`
  - 5-fold balanced accuracy mean `0.6547`
  - 5-fold accuracy mean `0.6631`
  - dynamic recall mean `0.5509`
- `/hola_ind feature_extraction.py + train.py` was reproduced from git blob `6379977`.
  - source feature shape `(425, 536)`
  - labels: static `218`, dynamic `207`
  - holdout accuracy `0.7765`
  - holdout balanced accuracy `0.7733`
  - holdout F1(dynamic) `0.7467`
  - 5-fold balanced accuracy mean `0.7289`
  - 5-fold F1(dynamic) mean `0.7142`
- User context: the previous experimenter reported that `feature_extraction.py` plateaued, then started optical-flow experiments and stopped. Therefore, feature engineering should be used as a baseline and ablation tool, not as the only path forward.

## Next Session Kickoff

When Minjoun says “다음 세션 하자” or “진행해”, start here:

1. Read this plan.
2. Use `ai-worker-qa`, `hola-ai-orchestrator`, and either `superpowers:executing-plans` or `superpowers:subagent-driven-development`.
3. Run the preflight commands below.
4. Execute Task 1 first with TDD.

Preflight commands:

```bash
pwd
git status --short
uv run pytest tests/unit/test_apply_dynamic_static_review.py tests/unit/test_train_pose_sequence.py
python3 - <<'PY'
import json
from pathlib import Path
p = Path("models/reports/pose_dynamic_static_raw_qa_kfold_metrics.json")
m = json.loads(p.read_text())
print(m["aggregate_valid"])
PY
```

Expected:

- working directory: `/Users/minjoun/Workspace/projects/Hola-Climbing/hola-climbing-ai`
- current GRU aggregate contains `balanced_accuracy_mean: 0.6547`

---

## File Map

- Create: `app/services/vision/tabular_features.py`
  - Converts pose landmarks into fixed-width tabular features.
  - Provides variants: `exact`, `normalized`, `velocity_only`.
  - `exact` must intentionally reproduce `/hola_ind/feature_extraction.py`, including its `all_landmarks[:, :99]` velocity slicing behavior.
- Create: `scripts/build_pose_tabular_dataset.py`
  - Builds `.npz` tabular datasets from `pose_json` files or existing pose cache.
  - Writes manifest CSV for traceability.
- Create: `scripts/train_tabular_dynamic_static.py`
  - Trains RandomForest, SVM, LogisticRegression.
  - Evaluates holdout, stratified 5-fold, and group 5-fold.
  - Writes metrics JSON, predictions CSV, and optional `.joblib` model.
- Create: `scripts/build_flow_dataset.py`
  - Extracts optical-flow features from videos under `/Users/minjoun/Movies/Original`.
  - Reuses `/hola_ind` flow feature idea, but evaluates with current QA labels.
- Create: `scripts/build_fusion_dataset.py`
  - Joins pose tabular features and flow features by `stem`.
- Create tests:
  - `tests/unit/test_tabular_features.py`
  - `tests/unit/test_build_pose_tabular_dataset.py`
  - `tests/unit/test_train_tabular_dynamic_static.py`
  - `tests/unit/test_flow_features.py`
- Modify:
  - `pyproject.toml`: add `scikit-learn`, `joblib`, `scipy` to the `ml` dependency group.
  - `.gitignore`: add ignored local artifacts if missing: `data/tabular_dataset/`, `data/flow_dataset/`, `data/fusion_dataset/`.
  - `README.md`: document comparison commands after results are produced.

---

### Task 1: Tabular Pose Feature Extractor

**Files:**
- Create: `app/services/vision/tabular_features.py`
- Test: `tests/unit/test_tabular_features.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
import numpy as np

from app.services.vision.tabular_features import (
    extract_tabular_pose_features,
    pose_json_frames_to_array,
)


def test_exact_feature_shape_is_hola_ind_compatible() -> None:
    pose = np.zeros((5, 33, 4), dtype=np.float32)
    pose[:, :, 0] = np.arange(33, dtype=np.float32)
    pose[:, :, 1] = np.arange(33, dtype=np.float32) + 100
    pose[:, :, 2] = np.arange(33, dtype=np.float32) + 200
    pose[:, :, 3] = 1.0

    features = extract_tabular_pose_features(pose, variant="exact")

    assert features.shape == (536,)


def test_velocity_only_removes_position_summary() -> None:
    pose = np.zeros((6, 33, 4), dtype=np.float32)
    pose[:, :, :3] = np.linspace(0.0, 1.0, num=6, dtype=np.float32)[:, None, None]
    pose[:, :, 3] = 1.0

    features = extract_tabular_pose_features(pose, variant="velocity_only")

    assert features.shape == (8,)
    assert features[2] > 0.0


def test_pose_json_frames_to_array_reads_keypoints() -> None:
    frames = [
        {"keypoints": [{"x": 1.0, "y": 2.0, "z": 3.0, "v": 0.5} for _ in range(33)]},
        {"keypoints": [{"x": 2.0, "y": 3.0, "z": 4.0, "v": 0.6} for _ in range(33)]},
    ]

    arr = pose_json_frames_to_array(frames)

    assert arr.shape == (2, 33, 4)
    assert arr[0, 0].tolist() == [1.0, 2.0, 3.0, 0.5]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_tabular_features.py
```

Expected: fails because `app.services.vision.tabular_features` does not exist.

- [ ] **Step 3: Implement minimal feature extractor**

Implementation requirements:

- `pose_json_frames_to_array(frames: list[dict[str, object]]) -> np.ndarray`
- `extract_tabular_pose_features(pose: np.ndarray, variant: Literal["exact", "normalized", "velocity_only"]) -> np.ndarray`
- `exact`:
  - flatten to `(T, 132)`
  - position summary: mean/std/min/max over time, 528 dims
  - velocity summary: reproduce `/hola_ind` behavior with `coords_xyz = all_landmarks[:, :99]`, then 8 velocity stats
- `normalized`:
  - hip-center and torso-scale normalize `x,y,z`
  - keep visibility
  - apply the same 528 + 8 summary structure, but velocity must use true xyz coordinates, not `[:, :99]`
- `velocity_only`:
  - true xyz velocity stats only, 8 dims

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_tabular_features.py
```

Expected: passes.

---

### Task 2: Pose Tabular Dataset Builder

**Files:**
- Create: `scripts/build_pose_tabular_dataset.py`
- Test: `tests/unit/test_build_pose_tabular_dataset.py`

- [ ] **Step 1: Write failing tests**

Test behavior:

```python
import csv
import json
from pathlib import Path

import numpy as np

from scripts.build_pose_tabular_dataset import build_tabular_dataset


def _write_pose_json(path: Path, frames: int = 4) -> None:
    payload = [
        {"keypoints": [{"x": float(i), "y": float(i + 1), "z": float(i + 2), "v": 1.0} for i in range(33)]}
        for _ in range(frames)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_tabular_dataset_from_pose_json(tmp_path: Path) -> None:
    pose_dir = tmp_path / "pose_json"
    pose_dir.mkdir()
    _write_pose_json(pose_dir / "A.json")
    _write_pose_json(pose_dir / "B.json")
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nA.json,0\nB.json,1\nC.json,\n", encoding="utf-8")
    out = tmp_path / "out"

    result = build_tabular_dataset(
        labels_csv=labels,
        pose_json_dir=pose_dir,
        out_dir=out,
        variant="exact",
    )

    assert result.written == 2
    assert result.missing == []
    assert sorted(p.name for p in out.glob("*.npz")) == ["A.npz", "B.npz"]
    with np.load(out / "A.npz", allow_pickle=False) as data:
        assert data["x"].shape == (536,)
        assert int(data["label"]) == 0
```

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_build_pose_tabular_dataset.py
```

Expected: fails because `scripts.build_pose_tabular_dataset` does not exist.

- [ ] **Step 3: Implement builder**

CLI must support:

```bash
uv run python scripts/build_pose_tabular_dataset.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/labels.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/hola_ind_exact \
  --variant exact
```

Required output per sample:

- `x`: feature vector
- `label`: `0` or `1`
- `stem`
- `source_path`
- `variant`

Also write:

- `data/tabular_dataset/hola_ind_exact_manifest.csv`

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_build_pose_tabular_dataset.py
```

Expected: passes.

---

### Task 3: Tabular Trainer And Split Reports

**Files:**
- Create: `scripts/train_tabular_dynamic_static.py`
- Test: `tests/unit/test_train_tabular_dynamic_static.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies**

Add to the `ml` group:

```toml
"scikit-learn>=1.5",
"joblib>=1.4",
"scipy>=1.13",
```

Then run:

```bash
uv sync --group ml
```

- [ ] **Step 2: Write failing tests**

Test behavior:

```python
import numpy as np

from scripts.train_tabular_dynamic_static import canonical_group, evaluate_predictions, stratified_group_splits


def test_canonical_group_strips_duplicate_suffix() -> None:
    assert canonical_group("IMG_3445 (1)") == "IMG_3445"
    assert canonical_group("IMG_3445") == "IMG_3445"


def test_evaluate_predictions_reports_balanced_accuracy() -> None:
    metrics = evaluate_predictions(
        y_true=np.asarray([0, 0, 1, 1]),
        y_prob=np.asarray([0.1, 0.8, 0.7, 0.2]),
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision_dynamic"] == 0.5
    assert metrics["recall_dynamic"] == 0.5
```

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run pytest tests/unit/test_train_tabular_dynamic_static.py
```

Expected: fails because trainer module does not exist.

- [ ] **Step 4: Implement trainer**

CLI must support:

```bash
uv run python scripts/train_tabular_dynamic_static.py \
  --data data/tabular_dataset/hola_ind_exact \
  --out models/tabular_hola_ind_exact_rf.joblib \
  --run-name tabular_hola_ind_exact \
  --splits holdout,kfold,group-kfold
```

Required models:

- `rf`: `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`
- `svm`: `Pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))`
- `logreg`: `Pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))`

Required reports:

- `models/reports/tabular_hola_ind_exact_metrics.json`
- `models/reports/tabular_hola_ind_exact_predictions.csv`

Metrics per split:

- accuracy
- balanced_accuracy
- precision_dynamic
- recall_dynamic
- specificity_static
- f1_dynamic
- tp/tn/fp/fn

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest tests/unit/test_train_tabular_dynamic_static.py
uv run ruff check scripts/train_tabular_dynamic_static.py tests/unit/test_train_tabular_dynamic_static.py
```

Expected: passes.

---

### Task 4: Reproduce `/hola_ind feature_extraction.py`

**Files:**
- Use: `scripts/build_pose_tabular_dataset.py`
- Use: `scripts/train_tabular_dynamic_static.py`
- Local artifacts: `data/tabular_dataset/hola_ind_exact/`, `models/reports/tabular_hola_ind_exact_*`

- [ ] **Step 1: Build exact dataset**

Run:

```bash
uv run python scripts/build_pose_tabular_dataset.py \
  --labels /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/labels.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/hola_ind_exact \
  --variant exact
```

Expected:

- written `425`
- feature dimension `536`
- static `218`, dynamic `207`

- [ ] **Step 2: Train and report**

Run:

```bash
uv run python scripts/train_tabular_dynamic_static.py \
  --data data/tabular_dataset/hola_ind_exact \
  --out models/tabular_hola_ind_exact_rf.joblib \
  --run-name tabular_hola_ind_exact \
  --splits holdout,kfold,group-kfold
```

Expected sanity range:

- holdout balanced accuracy near `0.7733`
- 5-fold balanced accuracy near `0.7289`

If the reproduced numbers differ by more than `0.03`, inspect exact compatibility first, especially the `all_landmarks[:, :99]` velocity slicing.

---

### Task 5: Evaluate With Current QA Labels

**Files:**
- Use: `data/review/labels_완료_qa.csv`
- Use: `scripts/build_pose_tabular_dataset.py`
- Use: `scripts/train_tabular_dynamic_static.py`

- [ ] **Step 1: Build exact QA dataset**

Run:

```bash
uv run python scripts/build_pose_tabular_dataset.py \
  --labels data/review/labels_완료_qa.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/qa_exact \
  --variant exact
```

Expected:

- written roughly `206`
- labels exclude blank/ambiguous rows

- [ ] **Step 2: Build normalized and velocity-only QA datasets**

Run:

```bash
uv run python scripts/build_pose_tabular_dataset.py \
  --labels data/review/labels_완료_qa.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/qa_normalized \
  --variant normalized

uv run python scripts/build_pose_tabular_dataset.py \
  --labels data/review/labels_완료_qa.csv \
  --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
  --out data/tabular_dataset/qa_velocity_only \
  --variant velocity_only
```

- [ ] **Step 3: Train all QA variants**

Run:

```bash
for variant in exact normalized velocity_only; do
  uv run python scripts/train_tabular_dynamic_static.py \
    --data "data/tabular_dataset/qa_${variant}" \
    --out "models/tabular_qa_${variant}_rf.joblib" \
    --run-name "tabular_qa_${variant}" \
    --splits holdout,kfold,group-kfold
done
```

Decision rule:

- If `exact` is high but `normalized` and `group-kfold` drop sharply, suspect camera/position shortcut.
- If `normalized` or `velocity_only` stays above GRU `0.6547` by at least `0.03`, tabular features are a strong candidate.

---

### Task 6: Optical Flow Baseline

**Files:**
- Create: `app/services/vision/flow_features.py`
- Create: `scripts/build_flow_dataset.py`
- Test: `tests/unit/test_flow_features.py`
- Local artifacts: `data/flow_dataset/qa_flow/`, `models/reports/flow_qa_*`

- [ ] **Step 1: Write failing flow feature tests**

Test behavior:

```python
import numpy as np

from app.services.vision.flow_features import extract_flow_stats, remove_fall_end


def test_remove_fall_end_trims_tail_spike() -> None:
    signal = np.asarray([1.0] * 20 + [100.0, 120.0], dtype=np.float32)
    trimmed = remove_fall_end(signal, tail_ratio=0.1)
    assert len(trimmed) < len(signal)


def test_extract_flow_stats_returns_42_features() -> None:
    signal = np.linspace(0.1, 1.0, num=90, dtype=np.float32)
    features = extract_flow_stats(signal)
    assert features.shape == (42,)
    assert np.isfinite(features).all()
```

- [ ] **Step 2: Implement flow feature functions**

Implement:

- `extract_flow_magnitude(video_path: Path, resize=(320, 240), target_fps=30) -> tuple[np.ndarray, float, float]`
- `remove_fall_end(flow_mag: np.ndarray, tail_ratio=0.05) -> np.ndarray`
- `extract_flow_stats(flow_mag: np.ndarray) -> np.ndarray`

Use `/hola_ind/flow_feature_extraction.py` as the source behavior.

- [ ] **Step 3: Build QA flow dataset**

Run:

```bash
uv run python scripts/build_flow_dataset.py \
  --labels data/review/labels_완료_qa.csv \
  --videos-dir /Users/minjoun/Movies/Original \
  --out data/flow_dataset/qa_flow
```

Expected:

- sample count close to QA labeled videos with existing source videos
- feature dimension `42`

- [ ] **Step 4: Train flow baseline**

Run:

```bash
uv run python scripts/train_tabular_dynamic_static.py \
  --data data/flow_dataset/qa_flow \
  --out models/flow_qa_rf.joblib \
  --run-name flow_qa \
  --splits holdout,kfold,group-kfold
```

Decision rule:

- If flow 5-fold balanced accuracy is near or above tabular pose, motion signal is stronger than pose coordinates.
- If flow catches dynamic false negatives from GRU, fusion is worth doing.

---

### Task 7: Pose + Flow Fusion

**Files:**
- Create: `scripts/build_fusion_dataset.py`
- Local artifacts: `data/fusion_dataset/qa_exact_flow/`, `models/reports/fusion_qa_*`

- [ ] **Step 1: Build fusion dataset**

Run:

```bash
uv run python scripts/build_fusion_dataset.py \
  --left data/tabular_dataset/qa_normalized \
  --right data/flow_dataset/qa_flow \
  --out data/fusion_dataset/qa_normalized_flow
```

Expected:

- only stems present in both datasets are included
- labels must match between both inputs
- feature dimension equals pose dimension + flow dimension

- [ ] **Step 2: Train fusion baseline**

Run:

```bash
uv run python scripts/train_tabular_dynamic_static.py \
  --data data/fusion_dataset/qa_normalized_flow \
  --out models/fusion_qa_normalized_flow_rf.joblib \
  --run-name fusion_qa_normalized_flow \
  --splits holdout,kfold,group-kfold
```

Decision rule:

- If fusion group-kfold balanced accuracy is at least `0.70`, promote fusion to integration candidate.
- If only holdout is high, keep it as experiment only.

---

### Task 8: Comparison Report And Next Decision

**Files:**
- Create: `models/reports/dynamic_static_baseline_comparison.md`
- Modify: `README.md`
- Update vault session log after results.

- [ ] **Step 1: Write comparison report**

Include table:

| Model | Dataset | Samples | Holdout Bal Acc | 5-fold Bal Acc | Group 5-fold Bal Acc | Dynamic Recall | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| GRU raw QA | `data/pose_dataset_reviewed` | 205 | n/a | 0.6547 | n/a | 0.5509 | Current baseline |
| Tabular exact | `qa_exact` | record after run | record after run | record after run | record after run | record after run | Shortcut risk |
| Tabular normalized | `qa_normalized` | record after run | record after run | record after run | record after run | record after run | Better generalization signal |
| Flow | `qa_flow` | record after run | record after run | record after run | record after run | record after run | Motion signal |
| Fusion | `qa_normalized_flow` | record after run | record after run | record after run | record after run | record after run | Candidate if group split holds |

- [ ] **Step 2: Decide next path**

Choose one:

- Promote tabular/fusion optional inference if group-kfold balanced accuracy is stable above `0.70`.
- Run second QA review if dynamic recall stays under `0.60`.
- Continue GRU only if tabular/fusion fails under group split.

- [ ] **Step 3: Final verification**

Run:

```bash
uv run pytest
uv run ruff check app tests scripts
uv run mypy app
```

Expected:

- pytest passes with only known Redis integration skips
- ruff passes
- mypy passes

---

## Risks And How To Read Results

- `/hola_ind exact` may learn camera/position shortcuts because it includes absolute `pos_mean`, `pos_std`, `pos_min`, and `pos_max`.
- `GroupKFold` is the guardrail for near-duplicate names such as `IMG_3445`, `IMG_3445 (1)`, `IMG_3445 (2)`.
- Optical flow may overreact to camera motion, falls, or crop changes. Keep prediction CSV and review high-confidence misses.
- If a score only looks good on single holdout, do not promote it.
- If normalized pose or flow retains most of the score under group split, that signal is more likely to generalize.

## Done Criteria

- `/hola_ind feature_extraction.py` baseline reproduced or explained with a delta under/over `0.03`.
- QA label comparisons generated for exact, normalized, velocity-only, flow, and fusion.
- A comparison report exists at `models/reports/dynamic_static_baseline_comparison.md`.
- Final recommendation is written in README or vault: promote candidate, run more QA, or keep current GRU baseline.

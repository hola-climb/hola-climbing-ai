"""Train a GRU dynamic/static classifier from cached pose `.npz` files.

Usage:
    uv run python scripts/train_pose_sequence.py \
        --data data/pose_dataset \
        --out models/pose_dynamic_static.pt \
        --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.vision.model_classifier import (
    PoseSequenceClassifier,
    require_torch,
    save_checkpoint,
)
from app.services.vision.pose_features import feature_size, prepare_pose_features


@dataclass(frozen=True)
class DatasetSplit:
    train_idx: list[int]
    valid_idx: list[int]


@dataclass(frozen=True)
class CachedPoseDataset:
    x: np.ndarray
    y: np.ndarray
    stems: list[str]
    raw_pose_frames: np.ndarray
    skipped_low_pose_frames: list[tuple[str, int]]


@dataclass(frozen=True)
class SplitTrainingResult:
    fold: int
    train_metrics: dict[str, float]
    valid_metrics: dict[str, float]
    prediction_rows: list[dict[str, object]]
    model_state: dict[str, object]


@dataclass(frozen=True)
class ReportOutputs:
    predictions_csv: Path
    metrics_json: Path


def load_npz_dataset(
    data_dir: Path,
    *,
    feature_set: str = "raw",
    min_raw_pose_frames: int = 0,
) -> CachedPoseDataset:
    """Load cached pose samples from `data_dir`."""
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise ValueError(f"no .npz files found under {data_dir}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    stems: list[str] = []
    raw_pose_frames: list[int] = []
    skipped_low_pose_frames: list[tuple[str, int]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            raw_x = data["x"].astype(np.float32)
            label = int(data["label"])
            stem = str(data["stem"])
            frame_count = int(data["raw_pose_frames"]) if "raw_pose_frames" in data.files else raw_x.shape[0]
        if raw_x.ndim != 2 or raw_x.shape[1] != feature_size("raw"):
            raise ValueError(f"{path} has invalid x shape {raw_x.shape}")
        if frame_count < min_raw_pose_frames:
            skipped_low_pose_frames.append((stem, frame_count))
            continue
        xs.append(prepare_pose_features(raw_x, feature_set=feature_set))
        ys.append(label)
        stems.append(stem)
        raw_pose_frames.append(frame_count)
    if not xs:
        raise ValueError(f"all samples were filtered out under {data_dir}")
    return CachedPoseDataset(
        x=np.stack(xs, axis=0),
        y=np.asarray(ys, dtype=np.float32),
        stems=stems,
        raw_pose_frames=np.asarray(raw_pose_frames, dtype=np.int64),
        skipped_low_pose_frames=skipped_low_pose_frames,
    )


def stratified_split(labels: np.ndarray, valid_ratio: float, seed: int) -> DatasetSplit:
    """Simple deterministic stratified split."""
    rng = random.Random(seed)
    train_idx: list[int] = []
    valid_idx: list[int] = []
    for label in (0, 1):
        idxs = [i for i, y in enumerate(labels) if int(y) == label]
        rng.shuffle(idxs)
        valid_count = max(1, round(len(idxs) * valid_ratio)) if len(idxs) >= 2 else 0
        valid_idx.extend(idxs[:valid_count])
        train_idx.extend(idxs[valid_count:])
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    if not train_idx or not valid_idx:
        raise ValueError("dataset needs at least two classes with enough samples for validation")
    return DatasetSplit(train_idx=train_idx, valid_idx=valid_idx)


def stratified_kfold(labels: np.ndarray, folds: int, seed: int) -> list[DatasetSplit]:
    """Build deterministic stratified k-fold splits."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for label in (0, 1):
        idxs = [i for i, y in enumerate(labels) if int(y) == label]
        if len(idxs) < folds:
            raise ValueError(f"not enough label={label} samples for {folds} folds")
        rng.shuffle(idxs)
        by_label[label] = idxs

    splits: list[DatasetSplit] = []
    all_indices = set(range(len(labels)))
    for fold in range(folds):
        valid_idx: list[int] = []
        for label in (0, 1):
            valid_idx.extend(by_label[label][fold::folds])
        valid_set = set(valid_idx)
        train_idx = sorted(all_indices - valid_set)
        rng.shuffle(train_idx)
        rng.shuffle(valid_idx)
        splits.append(DatasetSplit(train_idx=train_idx, valid_idx=valid_idx))
    return splits


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    y_int = y_true.astype(np.int64)
    accuracy = float((y_pred == y_int).mean()) if y_int.size else 0.0
    tp = int(((y_pred == 1) & (y_int == 1)).sum())
    tn = int(((y_pred == 0) & (y_int == 0)).sum())
    fp = int(((y_pred == 1) & (y_int == 0)).sum())
    fn = int(((y_pred == 0) & (y_int == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "accuracy": round(accuracy, 4),
        "precision_dynamic": round(precision, 4),
        "recall_dynamic": round(recall, 4),
        "specificity_static": round(specificity, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def write_reports(
    report_dir: Path,
    *,
    run_name: str,
    prediction_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> ReportOutputs:
    """Write prediction-level CSV and summary metrics JSON."""
    report_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = report_dir / f"{run_name}_predictions.csv"
    metrics_json = report_dir / f"{run_name}_metrics.json"
    fieldnames = [
        "fold",
        "split",
        "stem",
        "label",
        "prob_dynamic",
        "pred",
        "correct",
        "raw_pose_frames",
    ]
    with predictions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    metrics_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReportOutputs(predictions_csv=predictions_csv, metrics_json=metrics_json)


def _prediction_rows(
    *,
    fold: int,
    split_name: str,
    indices: list[int],
    stems: list[str],
    raw_pose_frames: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_pred = (y_prob >= 0.5).astype(np.int64)
    for local_idx, sample_idx in enumerate(indices):
        label = int(y_true[local_idx])
        pred = int(y_pred[local_idx])
        rows.append(
            {
                "fold": fold,
                "split": split_name,
                "stem": stems[sample_idx],
                "label": label,
                "prob_dynamic": round(float(y_prob[local_idx]), 6),
                "pred": pred,
                "correct": pred == label,
                "raw_pose_frames": int(raw_pose_frames[sample_idx]),
            }
        )
    return rows


def _aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "accuracy",
        "precision_dynamic",
        "recall_dynamic",
        "specificity_static",
        "balanced_accuracy",
    ]
    aggregate: dict[str, float] = {}
    for key in keys:
        values = np.asarray([m[key] for m in fold_metrics], dtype=np.float32)
        aggregate[f"{key}_mean"] = round(float(values.mean()), 4)
        aggregate[f"{key}_std"] = round(float(values.std()), 4)
    return aggregate


def _train_one_split(
    *,
    torch: Any,
    x_np: np.ndarray,
    y_np: np.ndarray,
    stems: list[str],
    raw_pose_frames: np.ndarray,
    split: DatasetSplit,
    fold: int,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    lr: float,
) -> SplitTrainingResult:
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    input_size = int(x_np.shape[2])
    model = PoseSequenceClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_metrics: dict[str, float] = {"balanced_accuracy": -1.0}
    best_state: dict[str, object] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        order = split.train_idx[:]
        random.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            train_prob = torch.sigmoid(model(x[split.train_idx])).cpu().numpy()
            valid_prob = torch.sigmoid(model(x[split.valid_idx])).cpu().numpy()
        train_metrics = binary_metrics(y_np[split.train_idx], train_prob)
        valid_metrics = binary_metrics(y_np[split.valid_idx], valid_prob)
        train_loss = sum(losses) / len(losses) if losses else 0.0
        print(
            f"[fold {fold:02d} epoch {epoch:03d}] loss={train_loss:.4f} "
            f"train={train_metrics} valid={valid_metrics}"
        )
        if valid_metrics["balanced_accuracy"] >= best_metrics.get("balanced_accuracy", -1.0):
            best_metrics = valid_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(x[split.train_idx])).cpu().numpy()
        valid_prob = torch.sigmoid(model(x[split.valid_idx])).cpu().numpy()
    final_train_metrics = binary_metrics(y_np[split.train_idx], train_prob)
    final_valid_metrics = binary_metrics(y_np[split.valid_idx], valid_prob)
    rows = _prediction_rows(
        fold=fold,
        split_name="train",
        indices=split.train_idx,
        stems=stems,
        raw_pose_frames=raw_pose_frames,
        y_true=y_np[split.train_idx],
        y_prob=train_prob,
    )
    rows.extend(
        _prediction_rows(
            fold=fold,
            split_name="valid",
            indices=split.valid_idx,
            stems=stems,
            raw_pose_frames=raw_pose_frames,
            y_true=y_np[split.valid_idx],
            y_prob=valid_prob,
        )
    )
    return SplitTrainingResult(
        fold=fold,
        train_metrics=final_train_metrics,
        valid_metrics=final_valid_metrics,
        prediction_rows=rows,
        model_state=best_state,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/pose_dataset"))
    parser.add_argument("--out", type=Path, default=Path("models/pose_dynamic_static.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", choices=["raw", "motion"], default="raw")
    parser.add_argument("--min-raw-pose-frames", type=int, default=30)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=Path("models/reports"))
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()

    torch = require_torch()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = load_npz_dataset(
        args.data,
        feature_set=args.feature_set,
        min_raw_pose_frames=args.min_raw_pose_frames,
    )
    if args.folds > 1:
        splits = stratified_kfold(dataset.y, folds=args.folds, seed=args.seed)
    else:
        splits = [stratified_split(dataset.y, valid_ratio=args.valid_ratio, seed=args.seed)]

    results: list[SplitTrainingResult] = []
    for fold_idx, split in enumerate(splits, start=1):
        results.append(
            _train_one_split(
                torch=torch,
                x_np=dataset.x,
                y_np=dataset.y,
                stems=dataset.stems,
                raw_pose_frames=dataset.raw_pose_frames,
                split=split,
                fold=fold_idx,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                lr=args.lr,
            )
        )

    best_result = max(results, key=lambda r: r.valid_metrics["balanced_accuracy"])
    model = PoseSequenceClassifier(
        input_size=int(dataset.x.shape[2]),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    model.load_state_dict(best_result.model_state)
    target_frames = int(dataset.x.shape[1])
    save_checkpoint(
        args.out,
        model=model,
        target_frames=target_frames,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        input_size=int(dataset.x.shape[2]),
        feature_set=args.feature_set,
        metrics=best_result.valid_metrics,
    )
    valid_metrics = [r.valid_metrics for r in results]
    prediction_rows = [row for result in results for row in result.prediction_rows]
    label_counts = {
        "static": int((dataset.y == 0).sum()),
        "dynamic": int((dataset.y == 1).sum()),
    }
    summary: dict[str, object] = {
        "data": str(args.data),
        "out": str(args.out),
        "samples": len(dataset.stems),
        "label_counts": label_counts,
        "feature_set": args.feature_set,
        "input_size": int(dataset.x.shape[2]),
        "target_frames": target_frames,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "folds": len(splits),
        "min_raw_pose_frames": args.min_raw_pose_frames,
        "skipped_low_pose_frames": dataset.skipped_low_pose_frames,
        "fold_metrics": [
            {
                "fold": result.fold,
                "train": result.train_metrics,
                "valid": result.valid_metrics,
            }
            for result in results
        ],
        "aggregate_valid": _aggregate_metrics(valid_metrics),
        "best_fold": best_result.fold,
        "best_valid": best_result.valid_metrics,
    }
    run_name = args.run_name or args.out.stem
    report_outputs = write_reports(
        args.report_dir,
        run_name=run_name,
        prediction_rows=prediction_rows,
        summary=summary,
    )
    print(
        f"\n[done] samples={len(dataset.stems)} folds={len(splits)} "
        f"target_frames={target_frames} out={args.out}"
    )
    print(f"[aggregate] {summary['aggregate_valid']}")
    print(f"[best] fold={best_result.fold} {best_result.valid_metrics}")
    print(f"[reports] predictions={report_outputs.predictions_csv} metrics={report_outputs.metrics_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

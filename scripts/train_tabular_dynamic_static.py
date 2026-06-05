"""Train sklearn dynamic/static classifiers from tabular `.npz` datasets.

Usage:
    uv run python scripts/train_tabular_dynamic_static.py \
        --data data/tabular_dataset/hola_ind_exact \
        --out models/tabular_hola_ind_exact_rf.joblib \
        --run-name tabular_hola_ind_exact \
        --splits holdout,kfold,group-kfold
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class DatasetSplit:
    train_idx: list[int]
    valid_idx: list[int]


@dataclass(frozen=True)
class TabularDataset:
    x: NDArray[np.float32]
    y: NDArray[np.int64]
    stems: list[str]
    source_paths: list[str]


@dataclass(frozen=True)
class ReportOutputs:
    predictions_csv: Path
    metrics_json: Path
    model_path: Path


def canonical_group(stem: str) -> str:
    """Collapse near-duplicate filenames like `IMG_3445 (1)` to `IMG_3445`."""
    return re.sub(r"\s+\(\d+\)$", "", stem)


def evaluate_predictions(y_true: NDArray[np.int64], y_prob: NDArray[np.float64]) -> dict[str, float | int]:
    """Compute binary dynamic/static metrics at threshold 0.5."""
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
    f1_dynamic = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round((recall + specificity) / 2.0, 4),
        "precision_dynamic": round(precision, 4),
        "recall_dynamic": round(recall, 4),
        "specificity_static": round(specificity, 4),
        "f1_dynamic": round(f1_dynamic, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def stratified_group_splits(
    labels: NDArray[np.int64],
    groups: list[str],
    *,
    folds: int,
    seed: int,
) -> list[DatasetSplit]:
    """Create group-disjoint stratified folds using canonicalized stems."""
    canonical_groups = [canonical_group(g) for g in groups]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    x_dummy = np.zeros(len(labels), dtype=np.float32)
    return [
        DatasetSplit(train_idx=train_idx.tolist(), valid_idx=valid_idx.tolist())
        for train_idx, valid_idx in splitter.split(x_dummy, labels, groups=canonical_groups)
    ]


def load_tabular_dataset(data_dir: Path) -> TabularDataset:
    """Load tabular `.npz` samples from a dataset directory."""
    files = _ordered_dataset_files(data_dir)
    if not files:
        raise ValueError(f"no .npz files found under {data_dir}")

    xs: list[NDArray[np.float32]] = []
    labels: list[int] = []
    stems: list[str] = []
    source_paths: list[str] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32)
            if x.ndim != 1:
                raise ValueError(f"{path} x must be a 1D feature vector, got {x.shape}")
            xs.append(x)
            labels.append(int(data["label"]))
            stems.append(str(data["stem"]))
            source_paths.append(str(data["source_path"]) if "source_path" in data.files else str(path))

    feature_dims = {x.shape[0] for x in xs}
    if len(feature_dims) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(feature_dims)}")
    return TabularDataset(
        x=np.stack(xs, axis=0).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        stems=stems,
        source_paths=source_paths,
    )


def _ordered_dataset_files(data_dir: Path) -> list[Path]:
    manifest_path = data_dir.with_name(f"{data_dir.name}_manifest.csv")
    if not manifest_path.exists():
        return sorted(data_dir.glob("*.npz"))

    ordered: list[Path] = []
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out_path = Path(row.get("out_path") or "")
            if not out_path.exists():
                out_path = data_dir / f"{row.get('stem')}.npz"
            if out_path.exists():
                ordered.append(out_path)
    return ordered if ordered else sorted(data_dir.glob("*.npz"))


def train_and_report(
    *,
    data_dir: Path,
    out_path: Path,
    run_name: str,
    split_names: list[str],
    model_names: list[str],
    folds: int = 5,
    valid_ratio: float = 0.2,
    seed: int = 42,
    report_dir: Path = Path("models/reports"),
) -> ReportOutputs:
    """Train configured models and write prediction/metric reports."""
    dataset = load_tabular_dataset(data_dir)
    prediction_rows: list[dict[str, object]] = []
    model_reports: dict[str, dict[str, object]] = {}

    for model_name in model_names:
        split_reports: dict[str, object] = {}
        for split_name in split_names:
            splits = _make_splits(
                split_name,
                labels=dataset.y,
                groups=dataset.stems,
                folds=folds,
                valid_ratio=valid_ratio,
                seed=seed,
            )
            metrics_by_fold: list[dict[str, float | int]] = []
            for fold, split in enumerate(splits):
                model = _build_model(model_name, seed=seed)
                model.fit(dataset.x[split.train_idx], dataset.y[split.train_idx])
                y_prob = _predict_dynamic_probability(model, dataset.x[split.valid_idx])
                metrics = evaluate_predictions(dataset.y[split.valid_idx], y_prob)
                metrics_by_fold.append(metrics)
                prediction_rows.extend(
                    _prediction_rows(
                        model_name=model_name,
                        split_name=split_name,
                        fold=fold,
                        split=split,
                        dataset=dataset,
                        y_prob=y_prob,
                    )
                )
            split_reports[split_name] = {
                "folds": metrics_by_fold,
                "aggregate_valid": _aggregate_metrics(metrics_by_fold),
            }
        model_reports[model_name] = split_reports

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_name = model_names[0]
    final_model = _build_model(final_model_name, seed=seed)
    final_model.fit(dataset.x, dataset.y)
    joblib.dump(
        {
            "model": final_model,
            "model_name": final_model_name,
            "run_name": run_name,
            "feature_dim": int(dataset.x.shape[1]),
            "classes": ["static", "dynamic"],
        },
        out_path,
    )

    predictions_csv = report_dir / f"{run_name}_predictions.csv"
    metrics_json = report_dir / f"{run_name}_metrics.json"
    _write_predictions(predictions_csv, prediction_rows)
    metrics_json.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "data_dir": str(data_dir),
                "samples": int(dataset.y.shape[0]),
                "feature_dim": int(dataset.x.shape[1]),
                "label_counts": {
                    "static": int((dataset.y == 0).sum()),
                    "dynamic": int((dataset.y == 1).sum()),
                },
                "models": model_reports,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReportOutputs(predictions_csv=predictions_csv, metrics_json=metrics_json, model_path=out_path)


def _make_splits(
    split_name: str,
    *,
    labels: NDArray[np.int64],
    groups: list[str],
    folds: int,
    valid_ratio: float,
    seed: int,
) -> list[DatasetSplit]:
    if split_name == "holdout":
        idx = np.arange(len(labels))
        train_idx, valid_idx = train_test_split(
            idx,
            test_size=valid_ratio,
            random_state=seed,
            stratify=labels,
        )
        return [DatasetSplit(train_idx=train_idx.tolist(), valid_idx=valid_idx.tolist())]
    if split_name == "kfold":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        x_dummy = np.zeros(len(labels), dtype=np.float32)
        return [
            DatasetSplit(train_idx=train_idx.tolist(), valid_idx=valid_idx.tolist())
            for train_idx, valid_idx in splitter.split(x_dummy, labels)
        ]
    if split_name == "group-kfold":
        return stratified_group_splits(labels, groups, folds=folds, seed=seed)
    raise ValueError(f"unsupported split: {split_name}")


def _build_model(model_name: str, *, seed: int) -> Any:
    if model_name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if model_name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, random_state=seed),
        )
    if model_name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    raise ValueError(f"unsupported model: {model_name}")


def _predict_dynamic_probability(model: Any, x: NDArray[np.float32]) -> NDArray[np.float64]:
    probabilities = model.predict_proba(x)
    return probabilities[:, 1].astype(np.float64)


def _prediction_rows(
    *,
    model_name: str,
    split_name: str,
    fold: int,
    split: DatasetSplit,
    dataset: TabularDataset,
    y_prob: NDArray[np.float64],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_pred = (y_prob >= 0.5).astype(np.int64)
    for local_idx, sample_idx in enumerate(split.valid_idx):
        label = int(dataset.y[sample_idx])
        pred = int(y_pred[local_idx])
        stem = dataset.stems[sample_idx]
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "fold": fold,
                "stem": stem,
                "group": canonical_group(stem),
                "label": label,
                "prob_dynamic": round(float(y_prob[local_idx]), 6),
                "pred": pred,
                "correct": pred == label,
                "source_path": dataset.source_paths[sample_idx],
            }
        )
    return rows


def _aggregate_metrics(metrics_by_fold: list[dict[str, float | int]]) -> dict[str, float]:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_dynamic",
        "recall_dynamic",
        "specificity_static",
        "f1_dynamic",
    ]
    aggregate: dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(m[key]) for m in metrics_by_fold], dtype=np.float32)
        aggregate[f"{key}_mean"] = round(float(values.mean()), 4)
        aggregate[f"{key}_std"] = round(float(values.std()), 4)
    return aggregate


def _write_predictions(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "split",
        "fold",
        "stem",
        "group",
        "label",
        "prob_dynamic",
        "pred",
        "correct",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--splits", default="holdout,kfold,group-kfold")
    parser.add_argument("--models", default="rf,svm,logreg")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=Path("models/reports"))
    args = parser.parse_args()

    outputs = train_and_report(
        data_dir=args.data,
        out_path=args.out,
        run_name=args.run_name,
        split_names=_parse_csv(args.splits),
        model_names=_parse_csv(args.models),
        folds=args.folds,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        report_dir=args.report_dir,
    )
    print(
        "[summary] "
        f"model={outputs.model_path} metrics={outputs.metrics_json} "
        f"predictions={outputs.predictions_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

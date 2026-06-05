"""Build tabular pose feature datasets from `/hola_ind` pose JSON files.

Usage:
    uv run python scripts/build_pose_tabular_dataset.py \
        --labels /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/labels.csv \
        --pose-json-dir /Users/minjoun/Workspace/projects/Hola-Climbing/hola_ind/pose_json \
        --out data/tabular_dataset/hola_ind_exact \
        --variant exact
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from app.services.vision.pose_dataset import load_label_rows
from app.services.vision.tabular_features import (
    TabularFeatureVariant,
    extract_tabular_pose_features,
    pose_json_frames_to_array,
)


@dataclass(frozen=True)
class BuildTabularDatasetResult:
    written: int
    missing: list[str]
    manifest_path: Path


def build_tabular_dataset(
    *,
    labels_csv: Path,
    pose_json_dir: Path,
    out_dir: Path,
    variant: TabularFeatureVariant,
) -> BuildTabularDatasetResult:
    """Build one compressed `.npz` feature file per labeled pose JSON."""
    rows = load_label_rows(labels_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.with_name(f"{out_dir.name}_manifest.csv")

    written = 0
    missing: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for stem, label in rows:
        pose_path = pose_json_dir / f"{stem}.json"
        if not pose_path.exists():
            missing.append(stem)
            continue

        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{pose_path} must contain a JSON array")
        pose = pose_json_frames_to_array(cast(list[dict[str, object]], payload))
        features = extract_tabular_pose_features(pose, variant=variant)
        out_path = out_dir / f"{stem}.npz"
        np.savez_compressed(
            out_path,
            x=features,
            label=np.asarray(label, dtype=np.int64),
            stem=np.asarray(stem),
            source_path=np.asarray(str(pose_path)),
            variant=np.asarray(variant),
        )
        written += 1
        manifest_rows.append(
            {
                "stem": stem,
                "label": label,
                "variant": variant,
                "feature_dim": features.shape[0],
                "source_path": str(pose_path),
                "out_path": str(out_path),
            }
        )

    _write_manifest(manifest_path, manifest_rows)
    return BuildTabularDatasetResult(written=written, missing=missing, manifest_path=manifest_path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["stem", "label", "variant", "feature_dim", "source_path", "out_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pose-json-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("exact", "normalized", "velocity_only"),
        default="exact",
    )
    args = parser.parse_args()

    result = build_tabular_dataset(
        labels_csv=args.labels,
        pose_json_dir=args.pose_json_dir,
        out_dir=args.out,
        variant=cast(TabularFeatureVariant, args.variant),
    )
    print(
        "[summary] "
        f"written={result.written} missing={len(result.missing)} manifest={result.manifest_path}"
    )
    if result.missing:
        print("[missing sample]", ", ".join(result.missing[:10]))
    return 0 if result.written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

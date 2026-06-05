"""Build fused tabular datasets by joining two `.npz` feature directories.

Usage:
    uv run python scripts/build_fusion_dataset.py \
        --left data/tabular_dataset/qa_normalized \
        --right data/flow_dataset/qa_flow \
        --out data/fusion_dataset/qa_normalized_flow
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BuildFusionDatasetResult:
    written: int
    missing_right: list[str]
    label_mismatch: list[str]
    manifest_path: Path


def build_fusion_dataset(
    *,
    left_dir: Path,
    right_dir: Path,
    out_dir: Path,
) -> BuildFusionDatasetResult:
    """Join feature vectors by `stem`, requiring matching labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir.with_name(f"{out_dir.name}_manifest.csv")
    right_by_stem = {_read_stem(path): path for path in _ordered_dataset_files(right_dir)}

    written = 0
    missing_right: list[str] = []
    label_mismatch: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    for left_path in _ordered_dataset_files(left_dir):
        left = _load_sample(left_path)
        right_path = right_by_stem.get(left["stem"])
        if right_path is None:
            missing_right.append(str(left["stem"]))
            continue
        right = _load_sample(right_path)
        if left["label"] != right["label"]:
            label_mismatch.append(str(left["stem"]))
            continue

        x = np.concatenate(
            [
                np.asarray(left["x"], dtype=np.float32),
                np.asarray(right["x"], dtype=np.float32),
            ]
        ).astype(np.float32)
        out_path = out_dir / f"{left['stem']}.npz"
        np.savez_compressed(
            out_path,
            x=x,
            label=np.asarray(left["label"], dtype=np.int64),
            stem=np.asarray(left["stem"]),
            source_path=np.asarray(left["source_path"]),
            right_source_path=np.asarray(right["source_path"]),
            variant=np.asarray("fusion"),
        )
        written += 1
        manifest_rows.append(
            {
                "stem": left["stem"],
                "label": left["label"],
                "variant": "fusion",
                "feature_dim": x.shape[0],
                "source_path": left["source_path"],
                "right_source_path": right["source_path"],
                "out_path": str(out_path),
            }
        )

    _write_manifest(manifest_path, manifest_rows)
    return BuildFusionDatasetResult(
        written=written,
        missing_right=missing_right,
        label_mismatch=label_mismatch,
        manifest_path=manifest_path,
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


def _read_stem(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        return str(data["stem"])


def _load_sample(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "x": data["x"].astype(np.float32),
            "label": int(data["label"]),
            "stem": str(data["stem"]),
            "source_path": str(data["source_path"]) if "source_path" in data.files else str(path),
        }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "stem",
        "label",
        "variant",
        "feature_dim",
        "source_path",
        "right_source_path",
        "out_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = build_fusion_dataset(left_dir=args.left, right_dir=args.right, out_dir=args.out)
    print(
        "[summary] "
        f"written={result.written} missing_right={len(result.missing_right)} "
        f"label_mismatch={len(result.label_mismatch)} manifest={result.manifest_path}"
    )
    if result.missing_right:
        print("[missing_right sample]", ", ".join(result.missing_right[:10]))
    if result.label_mismatch:
        print("[label_mismatch sample]", ", ".join(result.label_mismatch[:10]))
    return 0 if result.written > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

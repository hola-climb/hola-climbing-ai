"""Build a local video cache from GCS objects matched to dynamic/static labels.

Usage:
    uv run python scripts/cache_gcs_labeled_videos.py \
        --labels /Users/minjoun/Workspace/projects/Hola-Climbing/labels_완료.csv \
        --bucket hola-climbing-log-videos \
        --prefix videos/Original/ \
        --cache-dir data/gcs_cache/videos/original \
        --matched-labels-out data/gcs_cache/labels_gcs_matched.csv \
        --manifest-out data/gcs_cache/gcs_original_manifest.csv \
        --backend gsutil \
        --download
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from google.cloud import storage

from app.services.vision.pose_dataset import VIDEO_EXTS, normalize_label


@dataclass(frozen=True)
class GcsVideoObject:
    bucket: str
    object_path: str
    filename: str
    stem: str
    size: int
    md5_hash: str
    crc32c: str
    updated: str


@dataclass(frozen=True)
class LabelRecord:
    filename: str
    stem: str
    label: int


@dataclass(frozen=True)
class MatchedGcsVideo:
    label_filename: str
    stem: str
    canonical_group: str
    label: int
    gcs_object: GcsVideoObject


@dataclass(frozen=True)
class MatchResult:
    matches: list[MatchedGcsVideo]
    missing_stems: list[str]
    unlabeled_stems: list[str]
    duplicate_object_stems: dict[str, list[str]]


@dataclass(frozen=True)
class CacheSummary:
    downloaded: int
    skipped_existing: int
    failed: list[tuple[str, str]]


class _BlobLike(Protocol):
    def download_to_filename(self, dest: str) -> None: ...


class _BucketLike(Protocol):
    def blob(self, name: str) -> _BlobLike: ...


class _ClientLike(Protocol):
    def bucket(self, bucket: str) -> _BucketLike: ...


def canonical_group(stem: str) -> str:
    """Collapse near-duplicate labels like `IMG_3445 (1)` to `IMG_3445`."""
    return re.sub(r"\s+\(\d+\)$", "", stem)


def parse_gsutil_ls_l_line(line: str) -> GcsVideoObject | None:
    """Parse one `gsutil ls -l` object line into a GcsVideoObject."""
    stripped = line.strip()
    if not stripped or stripped.startswith("TOTAL:"):
        return None
    parts = stripped.split(maxsplit=2)
    if len(parts) != 3 or not parts[0].isdigit() or not parts[2].startswith("gs://"):
        return None
    size = int(parts[0])
    updated = parts[1]
    uri = parts[2]
    without_scheme = uri[len("gs://") :]
    bucket, _, object_path = without_scheme.partition("/")
    if not bucket or not object_path:
        return None
    filename = Path(object_path).name
    if Path(filename).suffix.lower() not in {ext.lower() for ext in VIDEO_EXTS}:
        return None
    return GcsVideoObject(
        bucket=bucket,
        object_path=object_path,
        filename=filename,
        stem=Path(filename).stem,
        size=size,
        md5_hash="",
        crc32c="",
        updated=updated,
    )


def list_gsutil_video_objects(*, bucket: str, prefix: str) -> list[GcsVideoObject]:
    """List video objects using the already-authenticated gsutil CLI."""
    uri = f"gs://{bucket}/{prefix.rstrip('/')}/**"
    result = subprocess.run(
        ["gsutil", "ls", "-l", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    objects: list[GcsVideoObject] = []
    for line in result.stdout.splitlines():
        obj = parse_gsutil_ls_l_line(line)
        if obj is not None:
            objects.append(obj)
    return objects


def list_gcs_video_objects(
    *,
    bucket: str,
    prefix: str,
    client: Any | None = None,
) -> list[GcsVideoObject]:
    """List supported video objects under a GCS prefix."""
    storage_client = client or storage.Client()
    allowed_suffixes = {ext.lower() for ext in VIDEO_EXTS}
    objects: list[GcsVideoObject] = []
    for blob in storage_client.list_blobs(bucket, prefix=prefix):
        path = str(blob.name)
        filename = Path(path).name
        if not filename or Path(filename).suffix.lower() not in allowed_suffixes:
            continue
        updated = blob.updated.isoformat() if getattr(blob, "updated", None) else ""
        objects.append(
            GcsVideoObject(
                bucket=bucket,
                object_path=path,
                filename=filename,
                stem=Path(filename).stem,
                size=int(blob.size or 0),
                md5_hash=str(blob.md5_hash or ""),
                crc32c=str(blob.crc32c or ""),
                updated=updated,
            )
        )
    return objects


def load_label_records(labels_csv: Path) -> tuple[list[LabelRecord], list[str]]:
    """Load labeled rows and separately return unlabeled stems."""
    records: list[LabelRecord] = []
    unlabeled: list[str] = []
    with labels_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            stem = Path(filename).stem
            label = normalize_label(row.get("label") or "")
            if label is None:
                unlabeled.append(stem)
                continue
            records.append(LabelRecord(filename=filename, stem=stem, label=label))
    return records, unlabeled


def match_labels_to_gcs(labels_csv: Path, objects: list[GcsVideoObject]) -> MatchResult:
    """Match labeled CSV rows to GCS videos by filename stem."""
    labels, unlabeled_stems = load_label_records(labels_csv)
    objects_by_stem: dict[str, list[GcsVideoObject]] = {}
    for obj in objects:
        objects_by_stem.setdefault(obj.stem, []).append(obj)

    duplicate_object_stems = {
        stem: [obj.object_path for obj in candidates]
        for stem, candidates in objects_by_stem.items()
        if len(candidates) > 1
    }

    matches: list[MatchedGcsVideo] = []
    missing_stems: list[str] = []
    for label in labels:
        candidates = objects_by_stem.get(label.stem)
        if not candidates:
            missing_stems.append(label.stem)
            continue
        # If the same stem appears more than once in GCS, choose the largest object.
        # The full candidate list is still reported in duplicate_object_stems.
        obj = sorted(candidates, key=lambda candidate: (-candidate.size, candidate.object_path))[0]
        matches.append(
            MatchedGcsVideo(
                label_filename=label.filename,
                stem=label.stem,
                canonical_group=canonical_group(label.stem),
                label=label.label,
                gcs_object=obj,
            )
        )
    return MatchResult(
        matches=matches,
        missing_stems=missing_stems,
        unlabeled_stems=unlabeled_stems,
        duplicate_object_stems=duplicate_object_stems,
    )


def write_matched_labels_csv(path: Path, matches: list[MatchedGcsVideo]) -> None:
    """Write `filename,label` CSV compatible with build_flow_dataset.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for match in matches:
            writer.writerow({"filename": match.label_filename, "label": match.label})


def write_match_manifest(path: Path, result: MatchResult, *, cache_dir: Path) -> None:
    """Write a GCS/label/cache mapping manifest for audit and reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem",
        "canonical_group",
        "label",
        "label_filename",
        "bucket",
        "object_path",
        "object_filename",
        "size",
        "md5_hash",
        "crc32c",
        "updated",
        "cache_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for match in result.matches:
            obj = match.gcs_object
            writer.writerow(
                {
                    "stem": match.stem,
                    "canonical_group": match.canonical_group,
                    "label": match.label,
                    "label_filename": match.label_filename,
                    "bucket": obj.bucket,
                    "object_path": obj.object_path,
                    "object_filename": obj.filename,
                    "size": obj.size,
                    "md5_hash": obj.md5_hash,
                    "crc32c": obj.crc32c,
                    "updated": obj.updated,
                    "cache_path": str(cache_dir / obj.filename),
                }
            )


def cache_matched_videos(
    matches: list[MatchedGcsVideo],
    *,
    cache_dir: Path,
    client: _ClientLike | None = None,
    backend: str = "python",
) -> CacheSummary:
    """Download matched GCS videos to a flat local cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    storage_client = client or (None if backend == "gsutil" else storage.Client())
    downloaded = 0
    skipped_existing = 0
    failed: list[tuple[str, str]] = []
    for match in matches:
        obj = match.gcs_object
        dest = cache_dir / obj.filename
        if dest.exists() and dest.stat().st_size == obj.size:
            skipped_existing += 1
            continue
        try:
            if backend == "gsutil":
                subprocess.run(
                    [
                        "gsutil",
                        "-o",
                        "GSUtil:sliced_object_download_threshold=0",
                        "cp",
                        f"gs://{obj.bucket}/{obj.object_path}",
                        str(dest),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                assert storage_client is not None
                storage_client.bucket(obj.bucket).blob(obj.object_path).download_to_filename(str(dest))
            downloaded += 1
        except Exception as exc:
            failed.append((obj.object_path, repr(exc)))
    return CacheSummary(
        downloaded=downloaded,
        skipped_existing=skipped_existing,
        failed=failed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--bucket", default="hola-climbing-log-videos")
    parser.add_argument("--prefix", default="videos/Original/")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/gcs_cache/videos/original"))
    parser.add_argument("--matched-labels-out", type=Path, default=Path("data/gcs_cache/labels_gcs_matched.csv"))
    parser.add_argument("--manifest-out", type=Path, default=Path("data/gcs_cache/gcs_original_manifest.csv"))
    parser.add_argument("--download", action="store_true", help="download matched videos into --cache-dir")
    parser.add_argument("--limit", type=int, help="limit matched rows for smoke runs")
    parser.add_argument(
        "--backend",
        choices=["python", "gsutil"],
        default="python",
        help="python uses google-cloud-storage ADC; gsutil uses existing gcloud login",
    )
    args = parser.parse_args()

    if args.backend == "gsutil":
        objects = list_gsutil_video_objects(bucket=args.bucket, prefix=args.prefix)
    else:
        objects = list_gcs_video_objects(bucket=args.bucket, prefix=args.prefix)
    result = match_labels_to_gcs(args.labels, objects)
    matches = result.matches[: args.limit] if args.limit else result.matches
    limited_result = MatchResult(
        matches=matches,
        missing_stems=result.missing_stems,
        unlabeled_stems=result.unlabeled_stems,
        duplicate_object_stems=result.duplicate_object_stems,
    )
    write_matched_labels_csv(args.matched_labels_out, matches)
    write_match_manifest(args.manifest_out, limited_result, cache_dir=args.cache_dir)
    print(
        "[summary] "
        f"objects={len(objects)} labeled_matches={len(result.matches)} "
        f"written_matches={len(matches)} missing={len(result.missing_stems)} "
        f"unlabeled={len(result.unlabeled_stems)} duplicate_object_stems={len(result.duplicate_object_stems)}"
    )
    if result.missing_stems:
        print("[missing sample]", ", ".join(result.missing_stems[:10]))
    if result.duplicate_object_stems:
        print("[duplicate sample]", next(iter(result.duplicate_object_stems.items())))

    if args.download:
        cache_summary = cache_matched_videos(matches, cache_dir=args.cache_dir, backend=args.backend)
        print(
            "[download] "
            f"downloaded={cache_summary.downloaded} "
            f"skipped_existing={cache_summary.skipped_existing} "
            f"failed={len(cache_summary.failed)}"
        )
        if cache_summary.failed:
            print("[failed sample]", cache_summary.failed[:5])
            return 2
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())

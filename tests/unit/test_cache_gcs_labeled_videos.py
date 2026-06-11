"""GCS labeled-video cache builder."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from scripts.cache_gcs_labeled_videos import (
    GcsVideoObject,
    cache_matched_videos,
    match_labels_to_gcs,
    parse_gsutil_ls_l_line,
    write_matched_labels_csv,
)


def _write_labels(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerow(["IMG_0001.json", "1"])
        writer.writerow(["IMG_0002.json", "0"])
        writer.writerow(["IMG_0003.json", ""])
        writer.writerow(["IMG_9999.json", "1"])
        writer.writerow(["IMG_3445 (1).json", "1"])


def test_match_labels_to_gcs_by_stem_and_groups_duplicates(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    objects = [
        GcsVideoObject(
            bucket="hola-climbing-log-videos",
            object_path="videos/videos/Original/IMG_0001.MOV",
            filename="IMG_0001.MOV",
            stem="IMG_0001",
            size=10,
            md5_hash="m1",
            crc32c="c1",
            updated="2026-06-10T00:00:00Z",
        ),
        GcsVideoObject(
            bucket="hola-climbing-log-videos",
            object_path="videos/videos/Original/IMG_0002.mp4",
            filename="IMG_0002.mp4",
            stem="IMG_0002",
            size=20,
            md5_hash="m2",
            crc32c="c2",
            updated="2026-06-10T00:00:00Z",
        ),
        GcsVideoObject(
            bucket="hola-climbing-log-videos",
            object_path="videos/videos/Original/IMG_3445 (1).MOV",
            filename="IMG_3445 (1).MOV",
            stem="IMG_3445 (1)",
            size=30,
            md5_hash="m3",
            crc32c="c3",
            updated="2026-06-10T00:00:00Z",
        ),
    ]

    result = match_labels_to_gcs(labels, objects)

    assert [m.stem for m in result.matches] == ["IMG_0001", "IMG_0002", "IMG_3445 (1)"]
    assert [m.label for m in result.matches] == [1, 0, 1]
    assert result.matches[2].canonical_group == "IMG_3445"
    assert result.missing_stems == ["IMG_9999"]
    assert result.unlabeled_stems == ["IMG_0003"]


def test_parse_gsutil_ls_l_line_builds_video_object() -> None:
    line = "  64711134  2026-04-22T08:36:43Z  gs://hola-climbing-log-videos/videos/Original/IMG_0028.MOV"

    obj = parse_gsutil_ls_l_line(line)

    assert obj == GcsVideoObject(
        bucket="hola-climbing-log-videos",
        object_path="videos/Original/IMG_0028.MOV",
        filename="IMG_0028.MOV",
        stem="IMG_0028",
        size=64711134,
        md5_hash="",
        crc32c="",
        updated="2026-04-22T08:36:43Z",
    )


def test_parse_gsutil_ls_l_line_ignores_total_and_non_video() -> None:
    assert parse_gsutil_ls_l_line("TOTAL: 1 objects, 1 bytes") is None
    assert (
        parse_gsutil_ls_l_line(
            "  12  2026-04-22T08:36:43Z  gs://hola-climbing-log-videos/videos/Original/readme.txt"
        )
        is None
    )


def test_write_matched_labels_csv_preserves_label_filenames(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(labels)
    result = match_labels_to_gcs(
        labels,
        [
            GcsVideoObject(
                bucket="bucket",
                object_path="prefix/IMG_0001.MOV",
                filename="IMG_0001.MOV",
                stem="IMG_0001",
                size=10,
                md5_hash="",
                crc32c="",
                updated="",
            )
        ],
    )
    out = tmp_path / "matched.csv"

    write_matched_labels_csv(out, result.matches)

    assert out.read_text(encoding="utf-8").splitlines() == [
        "filename,label",
        "IMG_0001.json,1",
    ]


class _FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.downloaded_to: str | None = None

    def download_to_filename(self, dest: str) -> None:
        self.downloaded_to = dest
        Path(dest).write_bytes(b"video")


class _FakeBucket:
    def __init__(self) -> None:
        self.blobs: dict[str, _FakeBlob] = {}

    def blob(self, name: str) -> _FakeBlob:
        blob = self.blobs.setdefault(name, _FakeBlob(name))
        return blob


class _FakeClient:
    def __init__(self) -> None:
        self.bucket_obj = _FakeBucket()

    def bucket(self, _bucket: str) -> _FakeBucket:
        return self.bucket_obj


def test_cache_matched_videos_downloads_missing_and_skips_existing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    existing = cache_dir / "IMG_0001.MOV"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"x" * 10)
    client = _FakeClient()
    # Build matches through the public matcher to keep field semantics realistic.
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nIMG_0001.json,1\nIMG_0002.json,0\n", encoding="utf-8")
    result = match_labels_to_gcs(
        labels,
        [
            GcsVideoObject("bucket", "prefix/IMG_0001.MOV", "IMG_0001.MOV", "IMG_0001", 10, "", "", ""),
            GcsVideoObject("bucket", "prefix/IMG_0002.MOV", "IMG_0002.MOV", "IMG_0002", 20, "", "", ""),
        ],
    )

    summary = cache_matched_videos(result.matches, cache_dir=cache_dir, client=client)

    assert summary.downloaded == 1
    assert summary.skipped_existing == 1
    assert (cache_dir / "IMG_0002.MOV").exists()
    assert client.bucket_obj.blobs["prefix/IMG_0002.MOV"].downloaded_to == str(cache_dir / "IMG_0002.MOV")


def test_cache_matched_videos_gsutil_disables_sliced_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    labels = tmp_path / "labels.csv"
    labels.write_text("filename,label\nIMG_5193.json,1\n", encoding="utf-8")
    result = match_labels_to_gcs(
        labels,
        [
            GcsVideoObject(
                "hola-climbing-log-videos",
                "videos/Original/IMG_5193.MOV",
                "IMG_5193.MOV",
                "IMG_5193",
                5,
                "",
                "",
                "",
            )
        ],
    )
    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert check is True
        assert capture_output is True
        assert text is True
        Path(cmd[-1]).write_bytes(b"video")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("scripts.cache_gcs_labeled_videos.subprocess.run", fake_run)

    summary = cache_matched_videos(result.matches, cache_dir=cache_dir, backend="gsutil")

    assert summary.downloaded == 1
    assert calls == [
        [
            "gsutil",
            "-o",
            "GSUtil:sliced_object_download_threshold=0",
            "cp",
            "gs://hola-climbing-log-videos/videos/Original/IMG_5193.MOV",
            str(cache_dir / "IMG_5193.MOV"),
        ]
    ]

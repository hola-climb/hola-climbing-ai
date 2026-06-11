"""ROI flow diagnostic probe."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.probe_roi_flow_diagnostics import (
    ProbeSample,
    RoiProbeResult,
    read_results_csv,
    select_probe_samples,
    summarize_probe,
    write_results_csv,
)


def test_select_probe_samples_picks_default_dynamic_probe_cohorts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(
        predictions,
        [
            ("HARD_FN", 1, 0, 0.10, "/videos/HARD_FN.mp4"),
            ("LOW_CONF_FN", 1, 0, 0.40, "/videos/LOW_CONF_FN.mp4"),
            ("EASY_1", 1, 1, 0.95, "/videos/EASY_1.mp4"),
            ("EASY_2", 1, 1, 0.90, "/videos/EASY_2.mp4"),
            ("STATIC", 0, 0, 0.10, "/videos/STATIC.mp4"),
        ],
    )

    samples = select_probe_samples(predictions, high_confidence_threshold=0.85, control_limit=1)

    assert [(sample.stem, sample.cohort) for sample in samples] == [
        ("HARD_FN", "high_conf_fn"),
        ("EASY_1", "correct_dynamic"),
    ]


def test_select_probe_samples_can_add_static_gate_cohorts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(
        predictions,
        [
            ("HARD_FN", 1, 0, 0.10, "/videos/HARD_FN.mp4"),
            ("EASY_DYN", 1, 1, 0.95, "/videos/EASY_DYN.mp4"),
            ("STATIC_QUIET", 0, 0, 0.02, "/videos/STATIC_QUIET.mp4"),
            ("STATIC_BORDER", 0, 0, 0.30, "/videos/STATIC_BORDER.mp4"),
            ("STATIC_FP_HI", 0, 1, 0.91, "/videos/STATIC_FP_HI.mp4"),
            ("STATIC_FP_LOW", 0, 1, 0.70, "/videos/STATIC_FP_LOW.mp4"),
        ],
    )

    samples = select_probe_samples(
        predictions,
        high_confidence_threshold=0.85,
        control_limit=1,
        include_static_cohorts=True,
        static_control_limit=1,
    )

    assert [(sample.stem, sample.cohort) for sample in samples] == [
        ("HARD_FN", "high_conf_fn"),
        ("EASY_DYN", "correct_dynamic"),
        ("STATIC_QUIET", "correct_static"),
        ("STATIC_FP_HI", "high_conf_fp_static"),
    ]


def test_select_probe_samples_can_run_only_static_gate_cohorts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(
        predictions,
        [
            ("HARD_FN", 1, 0, 0.10, "/videos/HARD_FN.mp4"),
            ("EASY_DYN", 1, 1, 0.95, "/videos/EASY_DYN.mp4"),
            ("STATIC_QUIET", 0, 0, 0.02, "/videos/STATIC_QUIET.mp4"),
            ("STATIC_FP_HI", 0, 1, 0.91, "/videos/STATIC_FP_HI.mp4"),
        ],
    )

    samples = select_probe_samples(
        predictions,
        include_static_cohorts=True,
        include_dynamic_cohorts=False,
        static_control_limit=1,
    )

    assert [(sample.stem, sample.cohort) for sample in samples] == [
        ("STATIC_QUIET", "correct_static"),
        ("STATIC_FP_HI", "high_conf_fp_static"),
    ]


def test_summarize_probe_reports_largest_effect() -> None:
    hard_1 = _result("HARD_1", "high_conf_fn", roi_vy_mean=3.0, adj_vy_mean=10.0)
    hard_2 = _result("HARD_2", "high_conf_fn", roi_vy_mean=3.5, adj_vy_mean=11.0)
    easy_1 = _result("EASY_1", "correct_dynamic", roi_vy_mean=1.0, adj_vy_mean=1.0)
    easy_2 = _result("EASY_2", "correct_dynamic", roi_vy_mean=1.5, adj_vy_mean=2.0)

    summary = summarize_probe([hard_1, hard_2, easy_1, easy_2], failures=[])

    assert summary.high_conf_fn_count == 2
    assert summary.correct_dynamic_count == 2
    assert summary.completed_count == 4
    assert summary.max_abs_effect_feature == "adj_vy_mean"
    assert summary.max_abs_effect_size > 1.0


def test_summarize_probe_reports_fn_vs_static_gate_direction() -> None:
    hard_1 = _result("HARD_1", "high_conf_fn", roi_vy_mean=2.0, adj_vy_mean=5.0)
    hard_2 = _result("HARD_2", "high_conf_fn", roi_vy_mean=2.5, adj_vy_mean=6.0)
    dynamic_1 = _result("DYN_1", "correct_dynamic", roi_vy_mean=3.0, adj_vy_mean=9.0)
    dynamic_2 = _result("DYN_2", "correct_dynamic", roi_vy_mean=3.5, adj_vy_mean=10.0)
    static_1 = _result("STATIC_1", "correct_static", roi_vy_mean=0.0, adj_vy_mean=0.0)
    static_2 = _result("STATIC_2", "correct_static", roi_vy_mean=1.0, adj_vy_mean=1.0)

    summary = summarize_probe([hard_1, hard_2, dynamic_1, dynamic_2, static_1, static_2], failures=[])

    static_comparison = next(
        comparison
        for comparison in summary.comparisons
        if comparison.name == "high_conf_fn_vs_correct_static"
    )
    assert static_comparison.max_abs_effect_feature == "adj_vy_mean"
    assert static_comparison.max_abs_effect_size > 1.0
    assert static_comparison.top_effects[0]["left_is_dynamic_side"] is True
    assert summary.static_gate["passed"] is True
    assert summary.static_gate["feature"] == "adj_vy_mean"


def test_read_results_csv_round_trips_probe_results(tmp_path: Path) -> None:
    path = tmp_path / "probe.csv"
    expected = _result("STATIC_1", "correct_static", roi_vy_mean=0.5, adj_vy_mean=0.25)

    write_results_csv(path, [expected])
    actual = read_results_csv(path)

    assert len(actual) == 1
    assert actual[0].sample.stem == "STATIC_1"
    assert actual[0].sample.cohort == "correct_static"
    assert actual[0].features["roi_vy_mean"] == 0.5
    assert actual[0].features["adj_vy_mean"] == 0.25


def _result(
    stem: str,
    cohort: str,
    *,
    roi_vy_mean: float,
    adj_vy_mean: float,
) -> RoiProbeResult:
    features = {
        name: 0.0
        for name in (
            "roi_mag_mean",
            "roi_mag_p95",
            "roi_mag_max",
            "roi_vy_mean",
            "roi_vy_std",
            "roi_vy_min",
            "roi_vy_max",
            "roi_vy_p10",
            "roi_vy_p90",
            "roi_upward_ratio",
            "roi_downward_ratio",
            "roi_max_upward_window_mean",
            "roi_max_downward_window_mean",
            "adj_mag_mean",
            "adj_mag_p95",
            "adj_mag_max",
            "adj_vy_mean",
            "adj_vy_std",
            "adj_vy_min",
            "adj_vy_max",
            "adj_vy_p10",
            "adj_vy_p90",
            "adj_upward_ratio",
            "adj_downward_ratio",
            "adj_max_upward_window_mean",
            "adj_max_downward_window_mean",
        )
    }
    features["roi_vy_mean"] = roi_vy_mean
    features["adj_vy_mean"] = adj_vy_mean
    return RoiProbeResult(
        sample=ProbeSample(
            stem=stem,
            cohort=cohort,
            label=0 if "static" in cohort else 1,
            pred=0 if cohort in {"high_conf_fn", "correct_static"} else 1,
            prob_dynamic=0.1 if cohort in {"high_conf_fn", "correct_static"} else 0.9,
            video_path=Path(f"/videos/{stem}.mp4"),
        ),
        features=features,
        sampled_frames=10,
        flow_frames=9,
        pose_frames=10,
        roi_flow_frames=9,
        fallback_flow_frames=0,
    )


def _write_predictions(path: Path, rows: list[tuple[str, int, int, float, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for stem, label, pred, prob_dynamic, source_path in rows:
            writer.writerow(
                {
                    "model": "rf",
                    "split": "group-kfold",
                    "fold": "0",
                    "stem": stem,
                    "group": stem,
                    "label": str(label),
                    "prob_dynamic": str(prob_dynamic),
                    "pred": str(pred),
                    "correct": str(label == pred),
                    "source_path": source_path,
                }
            )

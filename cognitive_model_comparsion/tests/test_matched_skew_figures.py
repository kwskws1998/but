"""Tests for trajectory-matched six-checkpoint table generation."""

import json
from argparse import Namespace

import pandas as pd

from cognitive_model_comparsion.scripts.plot_llama3_8b_ob1_mean_profiles import (
    aggregate_metrics,
    load_trajectory_matched_tables,
    run as run_figure_generation,
    write_reviewer_metric_table,
)


def write_analysis_fixture(path, skew, fixation_digest):
    """Write one minimal trajectory-matched six-checkpoint analysis."""
    path.mkdir()
    profile_rows = []
    metric_rows = []
    methods = (
        "raw_delta",
        "support_centered_sd_symmetric",
        "learned_asymmetric",
    )
    for index in range(1, 7):
        checkpoint_id = f"s{index:02d}_fixture"
        for offset, ob1, raw, symmetric, asymmetric in (
            (-1, 0.1, 0.0, 0.25, 0.1),
            (0, 0.5, 1.0, 0.5, 0.4),
            (1, 0.4, 0.0, 0.25, 0.5),
        ):
            profile_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "ob1_attention_skew": float(skew),
                    "relative_t5_token_offset": offset,
                    "ob1_attention_profile": ob1,
                    "raw_delta": raw,
                    "support_centered_sd_symmetric": symmetric,
                    "learned_asymmetric": asymmetric,
                    "requested_skew_matches_trajectory": True,
                }
            )
        for method_index, method in enumerate(methods):
            metric_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "ob1_attention_skew": float(skew),
                    "method": method,
                    "profile_spearman": 0.5 + method_index * 0.1,
                    "js_divergence": 0.2 - method_index * 0.02,
                    "requested_skew_matches_trajectory": True,
                }
            )
    pd.DataFrame(profile_rows).to_csv(
        path / "kernel_profiles.csv",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(
        path / "reviewer_kernel_summary.csv",
        index=False,
    )
    (path / "attention_profile_audit.json").write_text(
        json.dumps(
            {
                "trajectory_attention_skew": float(skew),
                "checkpoint_count": 6,
                "seed_count": 100,
                "passage_count": 55,
                "ob1_fixations_sha256": fixation_digest,
                "ob1_worker_manifest_sha256": f"manifest-{skew}",
            }
        ),
        encoding="utf-8",
    )


def test_matched_skew_inputs_create_four_column_reviewer_table(tmp_path):
    """Skew-specific trajectory analyses remain separate through averaging."""
    skew3_dir = tmp_path / "skew3"
    skew4_dir = tmp_path / "skew4"
    write_analysis_fixture(skew3_dir, 3.0, "fixations-skew3")
    write_analysis_fixture(skew4_dir, 4.0, "fixations-skew4")

    _, metrics, audit = load_trajectory_matched_tables(
        skew3_dir,
        skew4_dir,
        expected_runs=6,
    )
    metric_summary = aggregate_metrics(metrics)
    csv_path = tmp_path / "table.csv"
    markdown_path = tmp_path / "table.md"
    write_reviewer_metric_table(
        metric_summary,
        csv_path,
        markdown_path,
    )

    table = pd.read_csv(csv_path)
    assert table.columns.tolist() == [
        "Condition",
        "Skew=3 Spearman",
        "Skew=3 JS",
        "Skew=4 Spearman",
        "Skew=4 JS",
    ]
    assert len(table) == 3
    assert audit["comparison_design"] == (
        "trajectory_matched_skew3_and_skew4"
    )
    assert "Skew=4 Spearman ↑" in markdown_path.read_text(encoding="utf-8")

    figure_output = tmp_path / "figures"
    result = run_figure_generation(
        Namespace(
            input=None,
            skew3_analysis_dir=skew3_dir,
            skew4_analysis_dir=skew4_dir,
            expected_runs=6,
            output_dir=figure_output,
            x_min=-1,
            x_max=1,
        )
    )
    assert result["comparison_design"] == (
        "trajectory_matched_skew3_and_skew4"
    )
    assert (figure_output / "llama3_8b_ob1_mean_profiles.png").is_file()
    assert (figure_output / "llama3_8b_ob1_region_mass.png").is_file()

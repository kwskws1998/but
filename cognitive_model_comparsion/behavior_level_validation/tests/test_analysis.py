"""Validate reviewer-table selection and checkpoint safeguards."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from cognitive_model_comparsion.behavior_level_validation.analysis import (
    METHOD_ORDER,
    build_behavior_contrast_table,
    build_behavior_result_table,
    load_evaluation_outputs,
    write_behavior_report,
)
from cognitive_model_comparsion.behavior_level_validation.main import (
    select_et1_checkpoints,
)


def summary_fixture() -> pd.DataFrame:
    """Return a complete four-method behavior summary."""
    records = []
    for index, method in enumerate(METHOD_ORDER):
        record = {
            "method": method,
            "display_name": method,
            "passages": 55,
        }
        js_value = 0.4 - index / 20
        ob1_js_value = 0.3 - index / 20
        for metric, value in {
            "human_spearman": 0.1 + index / 10,
            "js_divergence": js_value,
            "hellinger_distance": js_value,
            "total_variation_distance": js_value,
            "overlap_coefficient": 1.0 - js_value,
            "ob1_spearman": 0.2 + index / 10,
            "ob1_js_divergence": ob1_js_value,
            "ob1_hellinger_distance": ob1_js_value,
            "ob1_total_variation_distance": ob1_js_value,
            "ob1_overlap_coefficient": 1.0 - ob1_js_value,
        }.items():
            record[metric] = value
            record[f"{metric}_ci_low"] = value - 0.01
            record[f"{metric}_ci_high"] = value + 0.01
        records.append(record)
    return pd.DataFrame(records)


def contrast_fixture() -> pd.DataFrame:
    """Return all compact-table behavior contrasts."""
    records = []
    for candidate, baseline in (
        ("et1_symmetric", "et1_raw"),
        ("et1_asymmetric", "et1_raw"),
        ("et1_asymmetric", "et1_symmetric"),
    ):
        for metric in (
            "human_spearman",
            "js_divergence",
            "hellinger_distance",
            "total_variation_distance",
            "overlap_coefficient",
            "ob1_spearman",
            "ob1_js_divergence",
            "ob1_hellinger_distance",
            "ob1_total_variation_distance",
            "ob1_overlap_coefficient",
        ):
            records.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "positive_means_improvement": True,
                    "passages": 55,
                    "mean_paired_improvement": 0.1,
                    "ci_low": 0.05,
                    "ci_high": 0.15,
                    "permutation_p_two_sided": 0.001,
                }
            )
    return pd.DataFrame(records)


def write_evaluation_fixture(tmp_path, human_target: str):
    """Write a minimal completed evaluation directory."""
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    summary_fixture().to_csv(
        evaluation_dir / "result_table.csv",
        index=False,
    )
    contrast_fixture().to_csv(
        evaluation_dir / "bootstrap_summary.csv",
        index=False,
    )
    pd.DataFrame({"word": ["alpha"]}).to_csv(
        evaluation_dir / "word_level_values.csv",
        index=False,
    )
    pd.DataFrame({"passage": [0]}).to_csv(
        evaluation_dir / "passage_metrics.csv",
        index=False,
    )
    with (evaluation_dir / "evaluation_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "human_target": human_target,
                "passages": 55,
                "bootstrap_samples": 10000,
            },
            handle,
        )
    return evaluation_dir


def test_behavior_result_table_uses_expected_methods_and_metrics():
    """Keep only the planned Human and OB1 behavior columns."""
    result = build_behavior_result_table(summary_fixture())
    assert result["method"].tolist() == list(METHOD_ORDER)
    assert "human_js_divergence" in result
    assert "ob1_js_divergence" in result
    assert "human_hellinger_distance" in result
    assert "ob1_total_variation_distance" in result
    assert "ob1_overlap_coefficient" in result
    assert "word_order_wasserstein" not in result


def test_behavior_contrasts_omit_pvalues_and_normalize_metric_names():
    """Keep paired effect estimates while omitting reviewer-facing p-values."""
    result = build_behavior_contrast_table(contrast_fixture())
    assert len(result) == 30
    assert "permutation_p_two_sided" not in result
    assert "human_js_divergence" in set(result["metric"])
    assert "ob1_hellinger_distance" in set(result["metric"])
    assert "human_total_variation_distance" in set(result["metric"])
    assert "ob1_overlap_coefficient" in set(result["metric"])
    assert result["positive_means_improvement"].all()


def test_report_requires_conditional_human_trt(tmp_path):
    """Reject an unconditional target for the paper-aligned primary report."""
    evaluation_dir = write_evaluation_fixture(
        tmp_path,
        "human_trt_unconditional",
    )
    with pytest.raises(ValueError, match="human_trt_conditional"):
        load_evaluation_outputs(evaluation_dir)


def test_write_behavior_report_records_actual_et1_scope(tmp_path):
    """Write compact files and explicit unit-compatibility metadata."""
    evaluation_dir = write_evaluation_fixture(
        tmp_path,
        "human_trt_conditional",
    )
    output_dir = tmp_path / "report"
    audit = write_behavior_report(evaluation_dir, output_dir)
    assert (output_dir / "behavior_result_table.csv").is_file()
    assert (output_dir / "behavior_paired_contrasts.csv").is_file()
    assert (output_dir / "RESULTS.md").is_file()
    assert audit["actual_passage_specific_et1_values_used"] is True
    assert audit["unit_impulse_kernel_profiles_used"] is False
    assert audit["raw_millisecond_rmse_computed"] is False
    assert "passage_normalized_overlap_coefficient" in audit[
        "derived_metrics"
    ]
    assert "passage_normalized_overlap_coefficient" not in audit[
        "primary_metrics"
    ]


def test_checkpoint_selection_refuses_implicit_sweep_average():
    """Require an explicit checkpoint choice when multiple sigmas are present."""
    values = pd.DataFrame(
        {
            "checkpoint_id": ["a", "b"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="multiple checkpoints"):
        select_et1_checkpoints(values, None)
    selected, selected_ids = select_et1_checkpoints(
        values,
        "b",
    )
    assert selected_ids == ["b"]
    assert selected["checkpoint_id"].tolist() == ["b"]

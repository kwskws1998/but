"""Tests for the frozen 12-checkpoint Provo attention analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognitive_model_comparsion.scripts import (
    run_provo_12_checkpoint_attention_analysis as runner,
)
from cognitive_model_comparsion.scripts.summarize_provo_12_checkpoint_attention import (
    METHODS,
    METRIC_DIRECTIONS,
    aggregate_paired_contrasts,
    checkpoint_passage_matrix,
    crossed_bootstrap_mean,
    paired_improvement,
)


def synthetic_passage_metrics() -> pd.DataFrame:
    """Build a complete deterministic checkpoint-passage metric grid."""
    method_values = {
        "raw_delta": (0.10, 0.50),
        "fixed_symmetric_sigma1": (0.55, 0.35),
        "support_centered_sd_symmetric": (0.60, 0.30),
        "support_centered_sd_ratio4": (0.65, 0.25),
        "support_rms_displacement_symmetric": (0.58, 0.32),
        "support_rms_displacement_ratio4": (0.63, 0.27),
        "mirrored_learned": (0.45, 0.40),
        "learned_asymmetric": (0.70, 0.20),
    }
    records = []
    for checkpoint_index in range(12):
        for passage_id in range(3):
            adjustment = checkpoint_index * 0.0001 + passage_id * 0.00001
            for method in METHODS:
                spearman, distance = method_values[method]
                records.append(
                    {
                        "checkpoint_id": f"s{checkpoint_index + 1:02d}",
                        "ob1_attention_skew": 3.0,
                        "passage_id_zero_based": passage_id,
                        "method": method,
                        "profile_spearman": spearman + adjustment,
                        "js_divergence": distance + adjustment,
                        "hellinger_distance": distance + adjustment,
                        "total_variation_distance": distance + adjustment,
                    }
                )
    return pd.DataFrame(records)


def test_validate_sigma_json_rejects_nonfinite_value(tmp_path: Path) -> None:
    """Require finite values in the exact 12-record sigma configuration."""
    records = runner.validate_sigma_json(runner.DEFAULT_SIGMA_JSON)
    records[0]["sigma_left"] = float("nan")
    path = tmp_path / "sigmas.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        runner.validate_sigma_json(path)


def test_python_path_preserves_virtualenv_symlink(tmp_path: Path) -> None:
    """Keep the venv launcher path instead of resolving to its base Python."""
    base_python = tmp_path / "base-python"
    base_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)

    selected = runner.absolute_path_preserving_symlinks(venv_python)

    assert selected == venv_python.absolute()
    assert selected != venv_python.resolve()


def test_crossed_bootstrap_preserves_point_estimate() -> None:
    """Return the cell mean while resampling both crossed axes."""
    matrix = np.arange(24, dtype=float).reshape(4, 6)
    mean, low, high = crossed_bootstrap_mean(
        matrix,
        1000,
        np.random.default_rng(7),
    )
    assert mean == pytest.approx(matrix.mean())
    assert low < mean < high


def test_checkpoint_passage_matrix_rejects_missing_cell() -> None:
    """Reject an incomplete checkpoint-by-passage coordinate grid."""
    values = pd.DataFrame(
        {
            "checkpoint_id": ["s01", "s01", "s02"],
            "passage_id_zero_based": [0, 1, 0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="Incomplete"):
        checkpoint_passage_matrix(values, "value")


def test_paired_improvement_orients_all_metrics_candidate_positive() -> None:
    """Orient similarity and distance differences as better-is-positive."""
    keys = {
        "checkpoint_id": ["s01"],
        "ob1_attention_skew": [3.0],
        "passage_id_zero_based": [0],
    }
    for metric, direction in METRIC_DIRECTIONS.items():
        if direction == "higher":
            candidate_value, baseline_value = 0.8, 0.6
        else:
            candidate_value, baseline_value = 0.2, 0.4
        candidate = pd.DataFrame({**keys, metric: [candidate_value]})
        baseline = pd.DataFrame({**keys, metric: [baseline_value]})
        result = paired_improvement(candidate, baseline, metric)
        assert result["improvement"].iloc[0] == pytest.approx(0.2)


def test_paired_aggregation_uses_checkpoint_and_passage_axes() -> None:
    """Aggregate paired deltas and retain every checkpoint-passage cell."""
    metrics = synthetic_passage_metrics()
    aggregate, checkpoints, cells, passages = aggregate_paired_contrasts(
        metrics,
        bootstrap_samples=500,
        seed=11,
    )
    primary = aggregate.loc[
        aggregate["candidate"].eq("learned_asymmetric")
        & aggregate["baseline"].eq("support_centered_sd_symmetric")
    ]
    assert len(primary) == len(METRIC_DIRECTIONS)
    assert primary["checkpoint_count"].eq(12).all()
    assert primary["passage_count"].eq(3).all()
    assert primary["candidate_wins"].eq(12).all()
    assert np.allclose(
        primary["equal_checkpoint_mean_paired_improvement"],
        0.1,
    )
    assert np.allclose(primary["crossed_bootstrap_ci_low"], 0.1)
    assert np.allclose(primary["crossed_bootstrap_ci_high"], 0.1)
    assert len(checkpoints) == len(aggregate) * 12
    assert len(cells) == len(aggregate) * 12 * 3
    assert len(passages) == len(aggregate) * 3

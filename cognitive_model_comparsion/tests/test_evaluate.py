"""Hand-calculated tests for cognitive-comparison metrics and bootstrap."""

import numpy as np
import pandas as pd
import pytest

from cognitive_model_comparsion.src.evaluate import (
    bootstrap_mean,
    normalized_allocation,
    paired_contrasts,
    passage_metric_values,
    summarize_methods_by_checkpoint,
)


def test_identical_allocations_have_perfect_rank_and_zero_distance():
    """Identical vectors yield rho one and zero JS/Wasserstein distance."""
    values = np.array([1.0, 2.0, 4.0])
    metrics = passage_metric_values(values, values, values)

    assert metrics["human_spearman"] == pytest.approx(1.0)
    assert metrics["ob1_spearman"] == pytest.approx(1.0)
    assert metrics["js_divergence"] == pytest.approx(0.0)
    assert metrics["wasserstein"] == pytest.approx(0.0)


def test_negative_values_are_clipped_only_for_distribution_metrics():
    """Allocation normalization reports clipping and produces unit mass."""
    allocation, clipped = normalized_allocation(np.array([-2.0, 1.0, 3.0]))

    assert clipped == 1
    assert allocation.tolist() == pytest.approx([0.0, 0.25, 0.75])


def test_bootstrap_is_deterministic_for_fixed_seed():
    """A fixed RNG seed reproduces the same interval and p-value."""
    values = np.array([0.1, 0.2, 0.3, 0.4])
    first = bootstrap_mean(values, 1000, np.random.default_rng(7))
    second = bootstrap_mean(values, 1000, np.random.default_rng(7))

    assert first == second
    assert first[0] == pytest.approx(0.25)


def test_lower_distance_is_encoded_as_positive_improvement():
    """Paired contrasts reverse lower-is-better distance differences."""
    rows = []
    for passage_id in range(3):
        rows.extend(
            [
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_raw",
                    "human_spearman": 0.1,
                    "js_divergence": 0.4,
                    "wasserstein": 0.3,
                    "ob1_spearman": 0.2,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_asymmetric",
                    "human_spearman": 0.2,
                    "js_divergence": 0.3,
                    "wasserstein": 0.1,
                    "ob1_spearman": 0.25,
                },
            ]
        )
    contrasts = paired_contrasts(
        pd.DataFrame(rows),
        bootstrap_samples=100,
        seed=1,
    )

    wasserstein = contrasts.query(
        "candidate == 'et1_asymmetric' and metric == 'wasserstein'"
    ).iloc[0]
    assert wasserstein["mean_paired_improvement"] == pytest.approx(0.2)


def test_asymmetric_is_directly_compared_with_symmetric():
    """The asymmetry claim has a paired symmetric-control contrast."""
    rows = []
    for passage_id in range(3):
        rows.extend(
            [
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_raw",
                    "human_spearman": 0.1,
                    "js_divergence": 0.5,
                    "wasserstein": 0.4,
                    "ob1_spearman": 0.1,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_symmetric",
                    "human_spearman": 0.2,
                    "js_divergence": 0.4,
                    "wasserstein": 0.3,
                    "ob1_spearman": 0.2,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_asymmetric",
                    "human_spearman": 0.3,
                    "js_divergence": 0.2,
                    "wasserstein": 0.1,
                    "ob1_spearman": 0.4,
                },
            ]
        )

    contrasts = paired_contrasts(
        pd.DataFrame(rows),
        bootstrap_samples=100,
        seed=1,
    )
    comparison = contrasts.query(
        "candidate == 'et1_asymmetric' "
        "and baseline == 'et1_symmetric' "
        "and metric == 'human_spearman'"
    ).iloc[0]

    assert comparison["mean_paired_improvement"] == pytest.approx(0.1)


def test_checkpoint_summary_keeps_each_rm_seed_separate():
    """Seed robustness rows cannot be collapsed into only a grand mean."""
    rows = []
    for checkpoint_id, value in (("seed41", 0.2), ("seed42", 0.4)):
        for passage_id in range(2):
            rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "passage_id_zero_based": passage_id,
                    "method": "et1_raw",
                    "human_spearman": value,
                    "js_divergence": 0.1,
                    "wasserstein": 0.2,
                    "ob1_spearman": 0.3,
                }
            )
    summary = summarize_methods_by_checkpoint(
        pd.DataFrame(rows),
        bootstrap_samples=100,
        seed=7,
    )

    assert set(summary["checkpoint_id"]) == {"seed41", "seed42"}
    assert summary.set_index("checkpoint_id").loc[
        "seed41",
        "human_spearman",
    ] == pytest.approx(0.2)

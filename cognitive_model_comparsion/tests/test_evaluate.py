"""Hand-calculated tests for cognitive-comparison metrics and bootstrap."""

import numpy as np
import pandas as pd
import pytest

from cognitive_model_comparsion.src.evaluate import (
    attach_checkpoint_metadata,
    build_sigma_sweep_summary,
    bootstrap_mean,
    cognitive_contrast_table,
    cognitive_result_table,
    evaluate_passages,
    matched_asymmetry_contrast_table,
    merge_word_values,
    normalized_allocation,
    paired_contrasts,
    paired_contrasts_by_checkpoint,
    paired_sign_flip_pvalue,
    passage_metric_values,
    select_ob1_clean_passages,
    summarize_methods,
    summarize_methods_by_checkpoint,
    write_evaluation_outputs,
)


def _passage_word_values(
    human_unconditional: list[float],
    human_conditional: list[float],
) -> pd.DataFrame:
    """Build one complete checkpoint-passage grid for evaluation tests."""
    count = len(human_unconditional)
    increasing = np.arange(1, count + 1, dtype=float)
    return pd.DataFrame(
        {
            "checkpoint_id": ["seed"] * count,
            "passage_id_zero_based": [0] * count,
            "word_id_zero_based": np.arange(count),
            "human_trt_unconditional": human_unconditional,
            "human_trt_conditional": human_conditional,
            "et1_raw_word_trt": increasing,
            "et1_symmetric_word_trt": increasing,
            "et1_rms_side_scale_symmetric_word_trt": increasing,
            "et1_asymmetric_word_trt": increasing,
            "ob1_tvt": increasing,
        }
    )


def _complete_merge_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build matching canonical, two-checkpoint ET1, and OB1 word grids."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1, 1],
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [0, 1],
            "word_raw": ["one", "two"],
            "human_trt_unconditional": [1.0, 2.0],
            "human_trt_conditional": [1.0, 2.0],
        }
    )
    et1 = pd.DataFrame(
        {
            "checkpoint_id": [
                "seed-a",
                "seed-a",
                "seed-b",
                "seed-b",
            ],
            "passage_id_zero_based": [0, 0, 0, 0],
            "word_id_zero_based": [0, 1, 0, 1],
            "et1_raw_word_trt": [1.0, 2.0, 1.0, 2.0],
            "et1_symmetric_word_trt": [1.0, 2.0, 1.0, 2.0],
            "et1_rms_side_scale_symmetric_word_trt": [
                1.0,
                2.0,
                1.0,
                2.0,
            ],
            "et1_asymmetric_word_trt": [1.0, 2.0, 1.0, 2.0],
        }
    )
    ob1 = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [0, 1],
            "ob1_tvt": [1.0, 2.0],
        }
    )
    return canonical, et1, ob1


def test_identical_allocations_have_perfect_rank_and_zero_distance():
    """Identical vectors yield rho one and zero JS/word-order distance."""
    values = np.array([1.0, 2.0, 4.0])
    metrics = passage_metric_values(values, values, values)

    assert metrics["human_spearman"] == pytest.approx(1.0)
    assert metrics["ob1_spearman"] == pytest.approx(1.0)
    assert metrics["js_divergence"] == pytest.approx(0.0)
    assert metrics["word_order_wasserstein"] == pytest.approx(0.0)
    assert metrics["ob1_js_divergence"] == pytest.approx(0.0)
    assert metrics["ob1_word_order_wasserstein"] == pytest.approx(0.0)


def test_ob1_distribution_metrics_use_ob1_instead_of_human_reference():
    """OB1 distances are independent of the separate Human target."""
    human = np.array([1.0, 0.0, 0.0])
    ob1 = np.array([0.0, 1.0, 0.0])
    metrics = passage_metric_values(human, ob1, ob1)

    assert metrics["js_divergence"] == pytest.approx(1.0)
    assert metrics["word_order_wasserstein"] == pytest.approx(0.5)
    assert metrics["ob1_js_divergence"] == pytest.approx(0.0)
    assert metrics["ob1_word_order_wasserstein"] == pytest.approx(0.0)


def test_word_order_transport_distinguishes_adjacent_and_far_mass():
    """Normalized word order charges less for adjacent than far transport."""
    human = np.array([1.0, 0.0, 0.0])
    adjacent = passage_metric_values(
        human,
        np.array([0.0, 1.0, 0.0]),
        human,
    )
    far = passage_metric_values(
        human,
        np.array([0.0, 0.0, 1.0]),
        human,
    )

    assert adjacent["word_order_wasserstein"] == pytest.approx(0.5)
    assert far["word_order_wasserstein"] == pytest.approx(1.0)
    assert adjacent["js_divergence"] == pytest.approx(1.0)
    assert far["js_divergence"] == pytest.approx(1.0)


def test_word_order_transport_uses_original_sparse_coordinates():
    """Excluded words do not collapse the remaining word-order distance."""
    human = np.array([1.0, 0.0, 0.0])
    method = np.array([0.0, 1.0, 0.0])
    metrics = passage_metric_values(
        human,
        method,
        human,
        positions=np.array([0.0, 0.2, 1.0]),
    )

    assert metrics["word_order_wasserstein"] == pytest.approx(0.2)


def test_js_output_is_divergence_not_scipy_distance():
    """The squared SciPy distance equals the hand-calculated JS divergence."""
    human = np.array([1.0, 0.0, 0.0])
    method = np.array([0.5, 0.5, 0.0])
    metrics = passage_metric_values(human, method, human)

    assert metrics["js_divergence"] == pytest.approx(0.31127812445913283)


def test_negative_values_are_clipped_only_for_distribution_metrics():
    """Allocation normalization reports clipping and produces unit mass."""
    allocation, clipped = normalized_allocation(np.array([-2.0, 1.0, 3.0]))

    assert clipped == 1
    assert allocation.tolist() == pytest.approx([0.0, 0.25, 0.75])


@pytest.mark.parametrize(
    ("human", "ob1", "message"),
    [
        (
            np.array([-1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            "Human TRT contains negative values",
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([-1.0, 2.0, 3.0]),
            "OB1 TVT contains negative values",
        ),
    ],
)
def test_passage_metrics_reject_negative_human_or_ob1(
    human,
    ob1,
    message,
):
    """Recorded Human TRT and simulated OB1 TVT cannot be negative."""
    with pytest.raises(ValueError, match=message):
        passage_metric_values(
            human,
            np.array([1.0, 2.0, 3.0]),
            ob1,
        )


def test_passage_metrics_allow_negative_et1_regression_output():
    """Only ET1 regression output may be clipped for allocation metrics."""
    metrics = passage_metric_values(
        np.array([1.0, 2.0, 3.0]),
        np.array([-1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert metrics["method_clipped_values"] == 1
    assert metrics["human_clipped_values"] == 0
    assert metrics["ob1_clipped_values"] == 0


def test_bootstrap_is_deterministic_for_fixed_seed():
    """A fixed RNG seed reproduces the same percentile interval."""
    values = np.array([0.1, 0.2, 0.3, 0.4])
    first = bootstrap_mean(values, 1000, np.random.default_rng(7))
    second = bootstrap_mean(values, 1000, np.random.default_rng(7))

    assert first == second
    assert first[0] == pytest.approx(0.25)


def test_cluster_bootstrap_divides_once_by_resampled_observation_count():
    """Unequal clusters retain a passage-weighted mean in every resample."""

    class FixedClusterRng:
        """Return all ordered two-cluster bootstrap samples once."""

        def integers(self, low, high=None, size=None):
            assert low == 0
            assert high == 2
            assert size == (4, 2)
            return np.array(
                [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ]
            )

    values = np.array([2.0, 4.0, 10.0])
    clusters = np.array(["a", "a", "b"])
    replicates = np.array([3.0, 16.0 / 3.0, 16.0 / 3.0, 10.0])
    expected_lower, expected_upper = np.percentile(
        replicates,
        [2.5, 97.5],
    )

    mean, lower, upper = bootstrap_mean(
        values,
        samples=4,
        rng=FixedClusterRng(),
        clusters=clusters,
    )

    assert mean == pytest.approx(16.0 / 3.0)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_paired_sign_flip_has_exact_hand_calculated_p_value():
    """Two equal positive differences have exact two-sided sign-flip p 0.5."""
    p_value = paired_sign_flip_pvalue(
        np.array([1.0, 1.0]),
        samples=4,
        rng=np.random.default_rng(7),
    )

    assert p_value == pytest.approx(0.5)


def test_monte_carlo_sign_flip_is_seeded_and_plus_one_corrected():
    """Monte Carlo sign flipping is reproducible and never returns zero."""
    differences = np.ones(30)
    first = paired_sign_flip_pvalue(
        differences,
        samples=999,
        rng=np.random.default_rng(11),
    )
    second = paired_sign_flip_pvalue(
        differences,
        samples=999,
        rng=np.random.default_rng(11),
    )

    assert first == second
    assert first >= 1 / 1000


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
                    "word_order_wasserstein": 0.3,
                    "ob1_spearman": 0.2,
                    "ob1_js_divergence": 0.4,
                    "ob1_word_order_wasserstein": 0.3,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_asymmetric",
                    "human_spearman": 0.2,
                    "js_divergence": 0.3,
                    "word_order_wasserstein": 0.1,
                    "ob1_spearman": 0.25,
                    "ob1_js_divergence": 0.1,
                    "ob1_word_order_wasserstein": 0.05,
                },
            ]
        )
    contrasts = paired_contrasts(
        pd.DataFrame(rows),
        bootstrap_samples=100,
        seed=1,
    )

    word_order_distance = contrasts.query(
        "candidate == 'et1_asymmetric' "
        "and metric == 'word_order_wasserstein'"
    ).iloc[0]
    ob1_js = contrasts.query(
        "candidate == 'et1_asymmetric' "
        "and metric == 'ob1_js_divergence'"
    ).iloc[0]
    ob1_word_order = contrasts.query(
        "candidate == 'et1_asymmetric' "
        "and metric == 'ob1_word_order_wasserstein'"
    ).iloc[0]
    assert word_order_distance["mean_paired_improvement"] == pytest.approx(0.2)
    assert ob1_js["mean_paired_improvement"] == pytest.approx(0.3)
    assert ob1_word_order["mean_paired_improvement"] == pytest.approx(0.25)
    assert "permutation_p_two_sided" in contrasts
    assert "bootstrap_p_two_sided" not in contrasts


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
                    "word_order_wasserstein": 0.4,
                    "ob1_spearman": 0.1,
                    "ob1_js_divergence": 0.5,
                    "ob1_word_order_wasserstein": 0.4,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_symmetric",
                    "human_spearman": 0.2,
                    "js_divergence": 0.4,
                    "word_order_wasserstein": 0.3,
                    "ob1_spearman": 0.2,
                    "ob1_js_divergence": 0.4,
                    "ob1_word_order_wasserstein": 0.3,
                },
                {
                    "checkpoint_id": "seed",
                    "passage_id_zero_based": passage_id,
                    "method": "et1_asymmetric",
                    "human_spearman": 0.3,
                    "js_divergence": 0.2,
                    "word_order_wasserstein": 0.1,
                    "ob1_spearman": 0.4,
                    "ob1_js_divergence": 0.2,
                    "ob1_word_order_wasserstein": 0.1,
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
                    "word_order_wasserstein": 0.2,
                    "ob1_spearman": 0.3,
                    "ob1_js_divergence": 0.15,
                    "ob1_word_order_wasserstein": 0.25,
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


def test_cluster_aware_summary_resamples_whole_clusters():
    """Optional cluster labels are accepted without changing the point mean."""
    rows = []
    for passage_id, cluster, value in (
        (0, "document-a", 0.1),
        (1, "document-a", 0.3),
        (2, "document-b", 0.7),
        (3, "document-b", 0.9),
    ):
        rows.append(
            {
                "checkpoint_id": "seed",
                "passage_id_zero_based": passage_id,
                "method": "et1_raw",
                "document_id": cluster,
                "human_spearman": value,
                "js_divergence": value,
                "word_order_wasserstein": value,
                "ob1_spearman": value,
                "ob1_js_divergence": value,
                "ob1_word_order_wasserstein": value,
            }
        )

    summary = summarize_methods(
        pd.DataFrame(rows),
        bootstrap_samples=100,
        seed=3,
        cluster_column="document_id",
    )

    assert summary.loc[0, "clusters"] == 2
    assert summary.loc[0, "human_spearman"] == pytest.approx(0.5)


def test_cognitive_tables_keep_only_et1_to_ob1_metrics():
    """Reviewer-facing tables exclude Human metrics and OB1 self-alignment."""
    method_summary = pd.DataFrame(
        {
            "method": [
                "et1_raw",
                "et1_symmetric",
                "et1_asymmetric",
                "ob1",
            ],
            "display_name": [
                "ET1 raw",
                "ET1 + symmetric",
                "ET1 + learned asymmetric",
                "OB1 baseline",
            ],
            "passages": [55, 55, 55, 55],
            "ob1_spearman": [0.6, 0.7, 0.8, 1.0],
            "ob1_spearman_ci_low": [0.5, 0.6, 0.7, 1.0],
            "ob1_spearman_ci_high": [0.7, 0.8, 0.9, 1.0],
            "ob1_js_divergence": [0.3, 0.2, 0.1, 0.0],
            "ob1_js_divergence_ci_low": [0.2, 0.1, 0.05, 0.0],
            "ob1_js_divergence_ci_high": [0.4, 0.3, 0.2, 0.0],
            "ob1_word_order_wasserstein": [0.3, 0.2, 0.1, 0.0],
            "ob1_word_order_wasserstein_ci_low": [
                0.2,
                0.1,
                0.05,
                0.0,
            ],
            "ob1_word_order_wasserstein_ci_high": [
                0.4,
                0.3,
                0.2,
                0.0,
            ],
            "human_spearman": [0.1, 0.2, 0.3, 0.4],
        }
    )
    contrasts = pd.DataFrame(
        {
            "candidate": ["et1_asymmetric"] * 4,
            "baseline": ["et1_raw"] * 4,
            "metric": [
                "human_spearman",
                "ob1_spearman",
                "ob1_js_divergence",
                "ob1_word_order_wasserstein",
            ],
            "mean_paired_improvement": [0.1, 0.2, 0.3, 0.4],
        }
    )

    summary = cognitive_result_table(method_summary)
    cognitive_contrasts = cognitive_contrast_table(contrasts)

    assert summary["method"].tolist() == [
        "et1_raw",
        "et1_symmetric",
        "et1_asymmetric",
    ]
    assert set(summary["reference_model"]) == {"ob1_tvt"}
    assert "human_spearman" not in summary
    assert set(cognitive_contrasts["metric"]) == {
        "ob1_spearman",
        "ob1_js_divergence",
        "ob1_word_order_wasserstein",
    }
    assert set(cognitive_contrasts["reference_model"]) == {"ob1_tvt"}


def test_output_writer_emits_separate_cognitive_csvs(tmp_path):
    """The standard writer always emits OB1-only summary and contrast files."""
    passage_metrics = pd.DataFrame(
        {
            "method": ["et1_raw", "ob1"],
            "passage_id_zero_based": [0, 0],
            "human_spearman": [0.5, 0.6],
        }
    )
    method_summary = pd.DataFrame(
        {
            "method": ["et1_raw", "ob1"],
            "display_name": ["ET1 raw", "OB1 baseline"],
            "passages": [55, 55],
            "ob1_spearman": [0.6, 1.0],
            "ob1_spearman_ci_low": [0.5, 1.0],
            "ob1_spearman_ci_high": [0.7, 1.0],
            "ob1_js_divergence": [0.2, 0.0],
            "ob1_js_divergence_ci_low": [0.1, 0.0],
            "ob1_js_divergence_ci_high": [0.3, 0.0],
            "ob1_word_order_wasserstein": [0.1, 0.0],
            "ob1_word_order_wasserstein_ci_low": [0.05, 0.0],
            "ob1_word_order_wasserstein_ci_high": [0.15, 0.0],
        }
    )
    contrasts = pd.DataFrame(
        [
            {
                "candidate": "et1_asymmetric",
                "baseline": "et1_raw",
                "metric": "human_spearman",
                "mean_paired_improvement": 0.1,
            },
            {
                "candidate": "et1_asymmetric",
                "baseline": "et1_raw",
                "metric": "ob1_spearman",
                "mean_paired_improvement": 0.2,
            },
            {
                "candidate": "et1_asymmetric",
                "baseline": "et1_symmetric",
                "metric": "human_spearman",
                "mean_paired_improvement": 0.3,
            },
            {
                "candidate": "et1_asymmetric",
                "baseline": "et1_symmetric",
                "metric": "ob1_spearman",
                "mean_paired_improvement": 0.4,
            },
        ]
    )

    write_evaluation_outputs(
        tmp_path,
        pd.DataFrame({"word": ["test"]}),
        passage_metrics,
        method_summary,
        pd.DataFrame({"checkpoint_id": ["seed"]}),
        contrasts,
        {"status": "complete"},
    )

    cognitive_summary = pd.read_csv(
        tmp_path / "cognitive_result_table.csv"
    )
    cognitive_contrasts = pd.read_csv(
        tmp_path / "cognitive_bootstrap_summary.csv"
    )
    assert cognitive_summary["method"].tolist() == ["et1_raw"]
    assert cognitive_summary["reference_model"].tolist() == ["ob1_tvt"]
    assert cognitive_contrasts["metric"].tolist() == [
        "ob1_spearman",
        "ob1_spearman",
    ]
    assert cognitive_contrasts["reference_model"].tolist() == [
        "ob1_tvt",
        "ob1_tvt",
    ]
    matched = pd.read_csv(
        tmp_path / "matched_asymmetry_contrasts.csv"
    )
    assert set(matched["baseline"]) == {"et1_symmetric"}
    assert matched["human_reference"].tolist() == [True, False]
    assert matched["ob1_reference"].tolist() == [False, True]


def test_matched_asymmetry_table_selects_only_symmetric_baseline():
    """The reviewer-facing matched table excludes raw-baseline contrasts."""
    contrasts = pd.DataFrame(
        {
            "candidate": ["et1_asymmetric", "et1_asymmetric"],
            "baseline": ["et1_raw", "et1_symmetric"],
            "metric": ["human_spearman", "ob1_spearman"],
            "mean_paired_improvement": [0.1, 0.2],
        }
    )

    matched = matched_asymmetry_contrast_table(contrasts)

    assert len(matched) == 1
    assert matched.iloc[0]["baseline"] == "et1_symmetric"
    assert matched.iloc[0]["ob1_reference"]


def test_matched_asymmetry_table_is_empty_for_raw_only_run():
    """Raw-only diagnostics keep a schema without inventing a contrast."""
    contrasts = pd.DataFrame(
        {
            "candidate": ["et1_asymmetric"],
            "baseline": ["et1_raw"],
            "metric": ["ob1_spearman"],
        }
    )

    matched = matched_asymmetry_contrast_table(contrasts)

    assert matched.empty
    assert "contrast_interpretation" in matched
    assert "human_reference" in matched
    assert "ob1_reference" in matched


def test_conditional_human_missing_words_are_masked_for_every_array():
    """Conditional NaNs remove the same word from Human, ET1, and OB1."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0, 4.0],
        human_conditional=[1.0, np.nan, 3.0, 4.0],
    )
    word_values.loc[1, "et1_raw_word_trt"] = np.nan
    word_values.loc[1, "et1_symmetric_word_trt"] = 999.0
    word_values.loc[1, "et1_asymmetric_word_trt"] = 999.0
    word_values.loc[1, "ob1_tvt"] = 999.0

    metrics = evaluate_passages(
        word_values,
        human_column="human_trt_conditional",
    )

    assert len(metrics) == 5
    assert set(metrics["original_word_count"]) == {4}
    assert set(metrics["word_count"]) == {3}
    assert set(metrics["human_missing_words_excluded"]) == {1}
    assert metrics["human_spearman"].tolist() == pytest.approx([1.0] * 5)
    assert metrics["ob1_spearman"].tolist() == pytest.approx([1.0] * 5)


def test_ob1_incompatible_words_are_common_masked_before_metrics():
    """Punctuation-only OB1 positions are excluded from every comparison."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 999.0, 3.0, 4.0],
        human_conditional=[1.0, 999.0, 3.0, 4.0],
    )
    word_values["ob1_evaluable"] = [True, False, True, True]
    for column in (
        "et1_raw_word_trt",
        "et1_symmetric_word_trt",
        "et1_rms_side_scale_symmetric_word_trt",
        "et1_asymmetric_word_trt",
        "ob1_tvt",
    ):
        word_values.loc[1, column] = -999.0

    metrics = evaluate_passages(word_values)

    assert set(metrics["original_word_count"]) == {4}
    assert set(metrics["ob1_compatible_word_count"]) == {3}
    assert set(metrics["ob1_incompatible_words_excluded"]) == {1}
    assert set(metrics["word_count"]) == {3}
    assert metrics["human_spearman"].tolist() == pytest.approx([1.0] * 5)


def test_ob1_clean_passage_sensitivity_filters_complete_passages():
    """Any passage containing an OB1-incompatible word is fully excluded."""
    passage_metrics = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0, 1, 1, 2, 2],
            "method": [
                "et1_raw",
                "ob1",
                "et1_raw",
                "ob1",
                "et1_raw",
                "ob1",
            ],
            "ob1_incompatible_words_excluded": [0, 0, 2, 2, 0, 0],
        }
    )

    clean, audit = select_ob1_clean_passages(passage_metrics)

    assert set(clean["passage_id_zero_based"]) == {0, 2}
    assert audit["source_passages"] == 3
    assert audit["included_passages"] == 2
    assert audit["excluded_passages"] == 1
    assert audit["excluded_passage_ids"] == [1]
    assert audit["source_ob1_incompatible_words"] == 2
    assert audit["included_passage_metric_rows"] == 4
    assert audit["sensitivity_policy"] == (
        "exclude_passages_with_any_ob1_incompatible_word"
    )


def test_ob1_clean_passage_sensitivity_rejects_inconsistent_counts():
    """One passage must have the same incompatibility count in every row."""
    passage_metrics = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0],
            "ob1_incompatible_words_excluded": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="differ within passages"):
        select_ob1_clean_passages(passage_metrics)


def test_merge_rejects_balanced_missing_and_duplicate_et1_coordinates():
    """A duplicate cannot conceal a missing checkpoint-word prediction."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1, 1],
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [0, 1],
            "word_raw": ["one", "two"],
            "human_trt_unconditional": [1.0, 2.0],
            "human_trt_conditional": [1.0, 2.0],
        }
    )
    et1 = pd.DataFrame(
        {
            "checkpoint_id": ["seed-a", "seed-a", "seed-b", "seed-b"],
            "passage_id_zero_based": [0, 0, 0, 0],
            "word_id_zero_based": [0, 0, 0, 1],
            "et1_raw_word_trt": [1.0, 1.0, 1.0, 2.0],
            "et1_symmetric_word_trt": [1.0, 1.0, 1.0, 2.0],
            "et1_rms_side_scale_symmetric_word_trt": [
                1.0,
                1.0,
                1.0,
                2.0,
            ],
            "et1_asymmetric_word_trt": [1.0, 1.0, 1.0, 2.0],
        }
    )
    ob1 = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [0, 1],
            "ob1_tvt": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate ET1 checkpoint-word coordinates",
    ):
        merge_word_values(canonical, et1, ob1)


def test_merge_rejects_duplicate_ob1_coordinates():
    """OB1 duplicates are rejected before a many-to-one merge."""
    canonical, et1, ob1 = _complete_merge_inputs()
    ob1 = pd.concat([ob1, ob1.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate OB1 word coordinates"):
        merge_word_values(canonical, et1, ob1)


def test_merge_rejects_missing_ob1_coordinates():
    """OB1 must contain every canonical evaluation coordinate."""
    canonical, et1, ob1 = _complete_merge_inputs()
    ob1 = ob1.iloc[:-1].copy()

    with pytest.raises(
        ValueError,
        match=r"OB1 coordinate grid differs.*missing=1, extra=0",
    ):
        merge_word_values(canonical, et1, ob1)


def test_merge_rejects_extra_ob1_coordinates():
    """OB1 coordinates outside the canonical evaluation grid are rejected."""
    canonical, et1, ob1 = _complete_merge_inputs()
    extra = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "word_id_zero_based": [2],
            "ob1_tvt": [3.0],
        }
    )
    ob1 = pd.concat([ob1, extra], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=r"OB1 coordinate grid differs.*missing=0, extra=1",
    ):
        merge_word_values(canonical, et1, ob1)


def test_merge_rejects_missing_et1_checkpoint_coordinates():
    """Every ET1 checkpoint must cover the complete canonical word grid."""
    canonical, et1, ob1 = _complete_merge_inputs()
    et1 = et1.loc[
        ~(
            et1["checkpoint_id"].eq("seed-a")
            & et1["word_id_zero_based"].eq(1)
        )
    ].copy()

    with pytest.raises(
        ValueError,
        match=(
            r"ET1 coordinate grid differs.*checkpoint seed-a: "
            r"missing=1, extra=0"
        ),
    ):
        merge_word_values(canonical, et1, ob1)


def test_merge_rejects_extra_et1_checkpoint_coordinates():
    """No ET1 checkpoint may contain words outside the canonical grid."""
    canonical, et1, ob1 = _complete_merge_inputs()
    extra = et1.iloc[[0]].copy()
    extra["word_id_zero_based"] = 2
    et1 = pd.concat([et1, extra], ignore_index=True)

    with pytest.raises(
        ValueError,
        match=(
            r"ET1 coordinate grid differs.*checkpoint seed-a: "
            r"missing=0, extra=1"
        ),
    ):
        merge_word_values(canonical, et1, ob1)


def test_partial_et1_method_missingness_is_rejected():
    """A retained NaN cannot silently remove one passage from a method."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0],
        human_conditional=[1.0, 2.0, 3.0],
    )
    word_values.loc[1, "et1_asymmetric_word_trt"] = np.nan

    with pytest.raises(ValueError, match="only partially missing"):
        evaluate_passages(word_values)


def test_raw_only_checkpoint_skips_fully_unavailable_redistributions():
    """The documented raw-only ET1 path still evaluates raw ET1 and OB1."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0],
        human_conditional=[1.0, 2.0, 3.0],
    )
    word_values["et1_symmetric_word_trt"] = np.nan
    word_values["et1_rms_side_scale_symmetric_word_trt"] = np.nan
    word_values["et1_asymmetric_word_trt"] = np.nan

    metrics = evaluate_passages(word_values)

    assert set(metrics["method"]) == {"et1_raw", "ob1"}


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_unconditional_human_rejects_missing_or_nonfinite_values(invalid_value):
    """The primary unconditional Human target must be complete and finite."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, invalid_value, 3.0],
        human_conditional=[1.0, 2.0, 3.0],
    )

    with pytest.raises(ValueError, match="missing or non-finite"):
        evaluate_passages(
            word_values,
            human_column="human_trt_unconditional",
        )


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_conditional_human_rejects_infinities(invalid_value):
    """Conditional masking permits NaN only, never positive or negative infinity."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0],
        human_conditional=[1.0, invalid_value, 3.0],
    )

    with pytest.raises(ValueError, match="contains infinity"):
        evaluate_passages(
            word_values,
            human_column="human_trt_conditional",
        )


@pytest.mark.parametrize("column", ["et1_raw_word_trt", "ob1_tvt"])
def test_conditional_evaluated_model_values_reject_infinity(column):
    """Infinity in any retained prediction array remains a hard error."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0],
        human_conditional=[1.0, np.nan, 3.0],
    )
    word_values.loc[2, column] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_passages(
            word_values,
            human_column="human_trt_conditional",
        )


def test_conditional_human_requires_two_evaluable_words():
    """A passage cannot be evaluated after masking down to fewer than two words."""
    word_values = _passage_word_values(
        human_unconditional=[1.0, 2.0, 3.0],
        human_conditional=[np.nan, np.nan, 3.0],
    )

    with pytest.raises(ValueError, match="leaves fewer than two words"):
        evaluate_passages(
            word_values,
            human_column="human_trt_conditional",
        )


def test_checkpoint_contrasts_and_sweep_summary_keep_all_sigma_pairs():
    """A multi-sigma sweep remains separate through bootstrap and export."""
    rows = []
    for checkpoint_id, asymmetric_gain in (
        ("rightward", 0.2),
        ("leftward", -0.1),
    ):
        for passage_id in range(3):
            for method, offset in (
                ("et1_raw", 0.0),
                ("et1_symmetric", 0.1),
                ("et1_asymmetric", 0.1 + asymmetric_gain),
                ("ob1", 0.3),
            ):
                value = 0.2 + 0.01 * passage_id + offset
                rows.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_zero_based": passage_id,
                        "method": method,
                        "human_spearman": value,
                        "js_divergence": 1.0 - value,
                        "word_order_wasserstein": 1.0 - value,
                        "ob1_spearman": value,
                        "ob1_js_divergence": 1.0 - value,
                        "ob1_word_order_wasserstein": 1.0 - value,
                    }
                )
    passage_metrics = pd.DataFrame(rows)
    checkpoint_summary = summarize_methods_by_checkpoint(
        passage_metrics,
        bootstrap_samples=100,
        seed=7,
    )
    checkpoint_contrasts = paired_contrasts_by_checkpoint(
        passage_metrics,
        bootstrap_samples=100,
        seed=7,
    )
    metadata = pd.DataFrame(
        {
            "checkpoint_id": ["rightward", "leftward"],
            "source_accuracy": [0.77, 0.75],
            "sigma_left": [0.4, 3.5],
            "sigma_right": [3.4, 0.7],
            "sigma_symmetric": [2.42, 2.52],
        }
    )
    checkpoint_summary = attach_checkpoint_metadata(
        checkpoint_summary,
        metadata,
    )
    checkpoint_contrasts = attach_checkpoint_metadata(
        checkpoint_contrasts,
        metadata,
    )
    sweep = build_sigma_sweep_summary(
        checkpoint_summary,
        checkpoint_contrasts,
    )

    matched = matched_asymmetry_contrast_table(checkpoint_contrasts)
    human_matched = matched.loc[
        matched["metric"].eq("human_spearman")
    ].set_index("checkpoint_id")
    assert human_matched.loc[
        "rightward",
        "mean_paired_improvement",
    ] == pytest.approx(0.2)
    assert human_matched.loc[
        "leftward",
        "mean_paired_improvement",
    ] == pytest.approx(-0.1)
    assert sweep["checkpoint_id"].tolist() == ["rightward", "leftward"]
    assert "et1_asymmetric__ob1_spearman" in sweep
    assert (
        "asym_minus_symmetric__human_spearman"
        "__mean_paired_improvement"
    ) in sweep

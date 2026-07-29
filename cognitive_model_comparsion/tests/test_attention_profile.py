"""Tests for the OB1-attention to native-T5 projection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cognitive_model_comparsion.src.attention_profile import (
    PROFILE_ANALYSIS_ESTIMAND,
    PROFILE_CONTRAST_CI_SCOPE,
    PROFILE_DISPLAY_NAMES,
    PROFILE_DISTRIBUTION_METRIC_SCOPE,
    PROFILE_MEAN_CI_SCOPE,
    PROFILE_SPEARMAN_SCOPE,
    RIGHTWARD_SHARE_SCOPE,
    build_t5_letter_geometry,
    centered_token_offset_sd,
    compare_attention_profiles,
    effective_ob1_attention_width,
    fit_scale_to_centered_token_offset_sd,
    fit_ob1_gaussian_prior,
    gaussian_weights,
    ob1_reference_display_name,
    ob1_attention_weight,
    profile_metrics,
    project_ob1_attention_to_t5,
    write_attention_profile_outputs,
)
from cognitive_model_comparsion.src.attention_profile_diagnostics import (
    plot_sigma_landscape,
)


def synthetic_profile_inputs():
    """Build two passages with aligned one-token-per-word T5 grids."""
    text = "Alpha beta gamma delta epsilon zeta"
    words = text.split()
    starts = []
    cursor = 0
    for word in words:
        starts.append(cursor)
        cursor += len(word) + 1
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 1],
            "passage_text": [text, text],
        }
    )
    token_rows = []
    fixation_rows = []
    for passage_id in range(2):
        for token_index, (word, start) in enumerate(zip(words, starts)):
            token_rows.append(
                {
                    "checkpoint_id": "learned",
                    "passage_id_zero_based": passage_id,
                    "token_index": token_index,
                    "character_start": start,
                    "character_end": start + len(word),
                    "is_special": 0,
                    "attention_mask": 1,
                    "word_id_zero_based": token_index,
                }
            )
        fixation_rows.extend(
            [
                {
                    "simulation_id": 0,
                    "seed": 0,
                    "text_id": passage_id,
                    "fixation_counter": 0,
                    "word_id": 1,
                    "word": "beta",
                    "fixation_duration": 200.0,
                    "saccade_type": np.nan,
                    "attentional_width": 5.0,
                    "eye_position": 8.0,
                },
                {
                    "simulation_id": 0,
                    "seed": 0,
                    "text_id": passage_id,
                    "fixation_counter": 1,
                    "word_id": 2,
                    "word": "gamma",
                    "fixation_duration": 300.0,
                    "saccade_type": "forward",
                    "attentional_width": 5.0,
                    "eye_position": 7.0,
                },
            ]
        )
    return passages, pd.DataFrame(token_rows), pd.DataFrame(fixation_rows)


def test_effective_width_replays_update_after_saved_value():
    """Saved width is converted to the width used for visual input."""
    assert effective_ob1_attention_width(5.0, np.nan) == 5.0
    assert effective_ob1_attention_width(4.0, "forward") == 4.5
    assert effective_ob1_attention_width(5.0, "regression") == 4.0
    assert effective_ob1_attention_width(3.0, "regression") == 3.0


def test_attention_equation_is_right_asymmetric_and_residual_is_explicit():
    """Focused and full profiles reproduce the vendored OB1 equation."""
    eccentricity = np.array([-2.0, 0.0, 2.0])
    focused = ob1_attention_weight(
        eccentricity,
        attention_width=5.0,
        attention_skew=3.0,
        profile_component="focused",
    )
    full = ob1_attention_weight(
        eccentricity,
        attention_width=5.0,
        attention_skew=3.0,
        profile_component="full",
    )

    assert focused[2] > focused[0]
    assert full == pytest.approx(focused + 0.25)


def test_letter_geometry_uses_real_character_offsets():
    """Punctuation-only T5 pieces remain represented in native coordinates."""
    passage = "Can't stop."
    tokens = pd.DataFrame(
        {
            "token_index": [0, 1, 2, 3],
            "word_id_zero_based": [0, 0, 1, 1],
            "character_start": [0, 3, 6, 10],
            "character_end": [3, 5, 10, 11],
        }
    )
    geometry, clean_words, starts = build_t5_letter_geometry(
        passage,
        tokens,
    )

    assert clean_words == ["cant", "stop"]
    assert starts == [0, 5]
    assert geometry["token_index"].tolist() == [0, 1, 2, 3]
    assert geometry.loc[3, "punctuation_only_token"]


def test_profile_comparison_writes_width_direction_and_fit_diagnostics(
    tmp_path,
):
    """The comparison emits width, ratio, direction, and fitted controls."""
    assert (
        PROFILE_DISPLAY_NAMES["support_rms_displacement_ratio4"]
        == "4:1"
    )
    passages, tokens, fixations = synthetic_profile_inputs()
    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        {
            "checkpoint_id": "learned",
            "sigma_left": 0.4,
            "sigma_right": 3.4,
        },
        attention_skews=(3.0, 4.0),
        bootstrap_samples=100,
        seed=7,
        trajectory_attention_skew=3.0,
    )
    write_attention_profile_outputs(tmp_path, artifacts)

    assert set(artifacts["passage_metrics"]["method"]) == {
        "raw_delta",
        "fixed_symmetric_sigma1",
        "rms_side_scale_symmetric",
        "fixed_ratio4_same_rms",
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
        "mirrored_learned",
        "learned_asymmetric",
    }
    assert not artifacts["audit"][
        "support_centered_sd_controls_enabled"
    ]
    assert artifacts["audit"]["support_centered_sd_matches"] == {}
    assert "fixed_ob1_gaussian" in artifacts[
        "attention_profiles"
    ].columns
    result_table = artifacts["result_table"]
    assert set(result_table["fixed_symmetric_sigma"]) == {1.0}
    rms_scales = result_table[
        "rms_side_scale_symmetric_sigma"
    ].drop_duplicates()
    assert len(rms_scales) == 1
    assert rms_scales.iloc[0] == pytest.approx(
        np.sqrt((0.4**2 + 3.4**2) / 2)
    )
    ratio4_left = result_table[
        "fixed_ratio4_same_rms_sigma_left"
    ].drop_duplicates().iloc[0]
    ratio4_right = result_table[
        "fixed_ratio4_same_rms_sigma_right"
    ].drop_duplicates().iloc[0]
    assert ratio4_right / ratio4_left == pytest.approx(4.0)
    assert np.sqrt((ratio4_left**2 + ratio4_right**2) / 2) == (
        pytest.approx(rms_scales.iloc[0])
    )
    no_redistribution = result_table.loc[
        result_table["method"].eq("raw_delta")
    ].iloc[0]
    assert no_redistribution["display_name"] == (
        "No redistribution "
        "(all allocation weight remains at the source token)"
    )
    assert not result_table["method"].eq("fixed_ob1_gaussian").any()
    assert not (
        artifacts["contrasts"]["candidate"].eq("fixed_ob1_gaussian")
        | artifacts["contrasts"]["baseline"].eq("fixed_ob1_gaussian")
    ).any()
    reference_label = ob1_reference_display_name("focused")
    assert set(result_table["reference_profile"]) == {reference_label}
    assert set(result_table["ci_scope"]) == {PROFILE_MEAN_CI_SCOPE}
    assert set(result_table["analysis_estimand"]) == {
        PROFILE_ANALYSIS_ESTIMAND
    }
    assert "same exact visible relative-token offsets" in (
        PROFILE_ANALYSIS_ESTIMAND
    )
    assert set(result_table["candidate_support_policy"]) == {
        "fixation_matched"
    }
    assert not bool(
        result_table["actual_et1_trt_magnitudes_used"].any()
    )
    audit = artifacts["audit"]
    assert audit["trajectory_attention_skew"] == 3.0
    assert audit["ci_level"] == 0.95
    assert audit["ci_resampling_unit"] == "passage"
    assert audit["ob1_simulations_pooled_before_bootstrap"]
    assert not audit["ob1_simulation_ids_resampled"]
    assert audit["sigma_values_treated_as_fixed"]
    assert audit["rightward_share_scope"] == RIGHTWARD_SHARE_SCOPE
    assert audit["profile_spearman_scope"] == PROFILE_SPEARMAN_SCOPE
    assert (
        audit["distribution_metric_scope"]
        == PROFILE_DISTRIBUTION_METRIC_SCOPE
    )
    assert audit["candidate_support_policy"] == "fixation_matched"
    assert audit["candidate_support_policy_is_primary"]
    assert not audit["candidate_support_policy_legacy_sensitivity"]
    assert audit["fixation_support_patterns_aggregated"]
    assert not audit["actual_et1_trt_magnitudes_used"]
    assert set(artifacts["passage_metrics"]["passage_id_zero_based"]) == {
        0,
        1,
    }
    metric_columns = {
        "profile_spearman",
        "js_divergence",
        "hellinger_distance",
        "total_variation_distance",
        "overlap_coefficient",
        "token_offset_wasserstein",
    }
    assert metric_columns.issubset(artifacts["passage_metrics"].columns)
    for metric in metric_columns:
        assert {
            metric,
            f"{metric}_ci_low",
            f"{metric}_ci_high",
        }.issubset(result_table.columns)
    assert set(artifacts["contrasts"]["metric"]) == metric_columns
    assert result_table["overlap_coefficient"].to_numpy() == pytest.approx(
        1.0 - result_table["total_variation_distance"].to_numpy()
    )
    assert result_table[
        "overlap_coefficient_ci_low"
    ].to_numpy() == pytest.approx(
        1.0 - result_table["total_variation_distance_ci_high"].to_numpy()
    )
    assert result_table[
        "overlap_coefficient_ci_high"
    ].to_numpy() == pytest.approx(
        1.0 - result_table["total_variation_distance_ci_low"].to_numpy()
    )
    contrast_index = [
        "checkpoint_id",
        "ob1_attention_skew",
        "candidate",
        "baseline",
    ]
    total_variation_contrasts = artifacts["contrasts"].loc[
        artifacts["contrasts"]["metric"].eq("total_variation_distance")
    ].set_index(contrast_index)
    overlap_contrasts = artifacts["contrasts"].loc[
        artifacts["contrasts"]["metric"].eq("overlap_coefficient")
    ].set_index(contrast_index)
    pd.testing.assert_frame_equal(
        total_variation_contrasts[
            [
                "mean_paired_improvement",
                "ci_low",
                "ci_high",
                "permutation_p_two_sided",
            ]
        ],
        overlap_contrasts[
            [
                "mean_paired_improvement",
                "ci_low",
                "ci_high",
                "permutation_p_two_sided",
            ]
        ],
    )
    prior = json.loads(
        (tmp_path / "fixed_ob1_priors.json").read_text()
    )[0]
    assert prior["profile_component"] == "focused"
    assert prior["projection_coordinate"] == (
        "relative_native_t5_token_index"
    )
    assert prior["candidate_support_policy"] == "fixation_matched"
    assert prior["sigma_right"] > 0
    assert prior["sigma_left"] > 0
    assert prior["right_left_ratio"] == pytest.approx(
        prior["sigma_right"] / prior["sigma_left"]
    )
    assert (
        tmp_path / "kernel_alignment_contrasts.csv"
    ).is_file()
    contrasts = pd.read_csv(
        tmp_path / "kernel_alignment_contrasts.csv"
    )
    assert set(contrasts["ci_scope"]) == {PROFILE_CONTRAST_CI_SCOPE}
    assert set(contrasts["reference_profile"]) == {reference_label}
    assert set(contrasts["candidate_display_name"]).issubset(
        set(PROFILE_DISPLAY_NAMES.values())
    )
    assert set(contrasts["baseline_display_name"]).issubset(
        set(PROFILE_DISPLAY_NAMES.values())
    )
    directionality = pd.read_csv(
        tmp_path / "kernel_directionality.csv"
    )
    learned = directionality.loc[
        directionality["method"].eq("learned_asymmetric")
    ].iloc[0]
    assert learned["right_mass"] > learned["left_mass"]
    ob1_reference = directionality.loc[
        directionality["method"].eq("ob1_attention_profile")
    ].iloc[0]
    assert ob1_reference["display_name"] == reference_label
    candidate_metadata = [
        "source_accuracy",
        "learned_sigma_left",
        "learned_sigma_right",
        "learned_right_left_ratio",
        "fixed_symmetric_sigma",
    ]
    assert directionality.loc[
        directionality["method"].eq("ob1_attention_profile"),
        candidate_metadata,
    ].isna().all().all()
    reviewer_summary = pd.read_csv(
        tmp_path / "reviewer_kernel_summary.csv"
    )
    assert set(reviewer_summary["method"]) == {
        "raw_delta",
        "fixed_symmetric_sigma1",
        "rms_side_scale_symmetric",
        "fixed_ratio4_same_rms",
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
        "mirrored_learned",
        "learned_asymmetric",
        "ob1_attention_profile",
    }
    assert set(reviewer_summary["reference_profile"]) == {
        reference_label
    }
    assert set(reviewer_summary["rightward_share_scope"]) == {
        RIGHTWARD_SHARE_SCOPE
    }
    assert set(reviewer_summary["profile_spearman_scope"]) == {
        PROFILE_SPEARMAN_SCOPE
    }
    assert set(reviewer_summary["distribution_metric_scope"]) == {
        PROFILE_DISTRIBUTION_METRIC_SCOPE
    }
    candidate_summary = reviewer_summary.loc[
        reviewer_summary["method"].ne("ob1_attention_profile")
    ]
    assert candidate_summary[
        [
            "hellinger_distance",
            "total_variation_distance",
            "overlap_coefficient",
        ]
    ].notna().all().all()
    for metric in (
        "hellinger_distance",
        "total_variation_distance",
        "overlap_coefficient",
    ):
        assert candidate_summary[
            [metric, f"{metric}_ci_low", f"{metric}_ci_high"]
        ].notna().all().all()
    assert set(reviewer_summary["candidate_support_policy"]) == {
        "fixation_matched"
    }
    summary_references = reviewer_summary.loc[
        reviewer_summary["method"].eq("ob1_attention_profile")
    ]
    assert summary_references[candidate_metadata].isna().all().all()
    skew_roles = reviewer_summary[
        [
            "ob1_attention_skew",
            "requested_skew_matches_trajectory",
            "attention_skew_analysis_role",
        ]
    ].drop_duplicates()
    matched = skew_roles.loc[
        skew_roles["ob1_attention_skew"].eq(3.0)
    ].iloc[0]
    sensitivity = skew_roles.loc[
        skew_roles["ob1_attention_skew"].eq(4.0)
    ].iloc[0]
    assert bool(matched["requested_skew_matches_trajectory"])
    assert matched["attention_skew_analysis_role"] == (
        "trajectory-matched attention-function evaluation"
    )
    assert not bool(sensitivity["requested_skew_matches_trajectory"])
    assert sensitivity["attention_skew_analysis_role"].startswith(
        "formula sensitivity:"
    )
    assert (tmp_path / "kernel_profiles.png").is_file()
    assert (tmp_path / "kernel_metric_comparison.png").is_file()
    assert (tmp_path / "kernel_profile_regions.png").is_file()
    assert (tmp_path / "gaussian_parameter_diagnostics.png").is_file()
    regions = pd.read_csv(tmp_path / "kernel_profile_regions.csv")
    assert regions["region_mass_sum"].to_numpy() == pytest.approx(1.0)
    diagnostics = pd.read_csv(
        tmp_path / "gaussian_parameter_diagnostics.csv"
    )
    assert {
        "fixed_ratio4_same_rms",
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
        "mirrored_learned",
        "ob1_fit_symmetric",
        "ob1_fit_ratio4",
        "ob1_fit_unconstrained",
    }.issubset(set(diagnostics["model_id"]))
    diagnostic_index = diagnostics.set_index(
        ["ob1_attention_skew", "model_id"]
    )
    for skew in (3.0, 4.0):
        learned_displacement = diagnostic_index.loc[
            (skew, "learned_fixed"),
            "realized_rms_token_displacement",
        ]
        for model_id in (
            "support_rms_displacement_symmetric",
            "support_rms_displacement_ratio4",
        ):
            assert diagnostic_index.loc[
                (skew, model_id),
                "realized_rms_token_displacement",
            ] == pytest.approx(learned_displacement, rel=1e-7)


def test_profile_comparison_can_skip_support_matched_controls(tmp_path):
    """The sweep option omits support-matched methods and diagnostics."""
    passages, tokens, fixations = synthetic_profile_inputs()
    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        {
            "checkpoint_id": "learned",
            "sigma_left": 0.4,
            "sigma_right": 3.4,
        },
        attention_skews=(3.0,),
        bootstrap_samples=20,
        seed=7,
        skip_support_rms_displacement_controls=True,
    )
    omitted = {
        "support_rms_displacement_symmetric",
        "support_rms_displacement_ratio4",
    }

    assert omitted.isdisjoint(artifacts["attention_profiles"].columns)
    for key in (
        "passage_metrics",
        "result_table",
        "directionality",
        "reviewer_summary",
    ):
        assert omitted.isdisjoint(set(artifacts[key]["method"]))
    assert omitted.isdisjoint(
        set(artifacts["parameter_diagnostics"]["model_id"])
    )
    assert omitted.isdisjoint(set(artifacts["contrasts"]["candidate"]))
    assert omitted.isdisjoint(set(artifacts["contrasts"]["baseline"]))
    assert not artifacts["audit"][
        "support_rms_displacement_controls_enabled"
    ]
    assert artifacts["audit"]["support_rms_displacement_matches"] == {}
    write_attention_profile_outputs(tmp_path, artifacts)
    assert (tmp_path / "kernel_profiles.png").is_file()
    assert (tmp_path / "kernel_metric_comparison.png").is_file()
    assert (tmp_path / "kernel_profile_regions.png").is_file()
    assert (tmp_path / "gaussian_parameter_diagnostics.png").is_file()


def test_centered_sd_controls_match_width_without_using_ob1_weights():
    """Opt-in controls match learned centered SD at ratios one and four."""
    passages, tokens, fixations = synthetic_profile_inputs()
    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        {
            "checkpoint_id": "learned",
            "sigma_left": 0.4,
            "sigma_right": 3.4,
        },
        attention_skews=(3.0,),
        bootstrap_samples=20,
        seed=7,
        include_support_centered_sd_controls=True,
    )
    methods = {
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
    }

    assert methods.issubset(set(artifacts["passage_metrics"]["method"]))
    assert methods.issubset(artifacts["attention_profiles"].columns)
    assert methods.issubset(
        set(artifacts["parameter_diagnostics"]["model_id"])
    )
    audit = artifacts["audit"]
    assert audit["support_centered_sd_controls_enabled"]
    match = audit["support_centered_sd_matches"]["learned"]
    assert not match["target_profile_uses_ob1_attention_weights"]
    for fit_name in ("symmetric_ratio1", "fixed_ratio4"):
        assert match[fit_name]["absolute_match_error"] < 1e-7
    assert (
        match["fixed_ratio4"]["sigma_right"]
        / match["fixed_ratio4"]["sigma_left"]
    ) == pytest.approx(4.0)

    diagnostics = artifacts["parameter_diagnostics"].set_index(
        "model_id"
    )
    learned_sd = diagnostics.loc[
        "learned_fixed",
        "realized_centered_token_offset_sd",
    ]
    for method in methods:
        assert diagnostics.loc[
            method,
            "realized_centered_token_offset_sd",
        ] == pytest.approx(learned_sd, rel=1e-7)
    for row in diagnostics.itertuples():
        assert row.realized_rms_token_displacement**2 == pytest.approx(
            row.realized_centered_token_offset_sd**2
            + row.realized_mean_token_offset**2,
            rel=1e-10,
            abs=1e-10,
        )
    result_table = artifacts["result_table"]
    assert result_table[
        [
            "learned_support_centered_token_sd",
            "support_centered_sd_symmetric_sigma",
            "support_centered_sd_ratio4_sigma_left",
            "support_centered_sd_ratio4_sigma_right",
        ]
    ].notna().all().all()


def test_centered_sd_fitter_matches_probability_weighted_dispersion():
    """The fixed-ratio optimizer matches SD around each profile mean."""
    support = np.arange(-3, 7)

    def builder(sigma_left, sigma_right):
        """Build one normalized Gaussian on the test support."""
        return gaussian_weights(support, sigma_left, sigma_right)

    learned = builder(0.4, 3.4)
    target = centered_token_offset_sd(learned, support)
    fitted = fit_scale_to_centered_token_offset_sd(
        builder,
        support,
        4.0,
        target,
    )
    achieved = centered_token_offset_sd(
        builder(fitted["sigma_left"], fitted["sigma_right"]),
        support,
    )

    assert fitted["sigma_right"] / fitted["sigma_left"] == pytest.approx(4.0)
    assert achieved == pytest.approx(target, rel=1e-7)
    assert fitted["absolute_match_error"] < 1e-7


def test_profile_metrics_exclude_shared_padding_zeros_from_spearman():
    """Absent offsets do not inflate passage-level rank correspondence."""
    reference = np.array([0.0, 0.0, 0.6, 0.3, 0.1, 0.0])
    candidate = np.array([0.0, 0.0, 0.6, 0.1, 0.3, 0.0])
    support = np.arange(-2, 4)

    metrics = profile_metrics(reference, candidate, support)

    assert metrics["profile_spearman"] == pytest.approx(0.5)
    assert metrics["hellinger_distance"] > 0
    assert metrics["total_variation_distance"] == pytest.approx(0.2)
    assert metrics["overlap_coefficient"] == pytest.approx(0.8)


def test_gaussian_mirroring_is_exact_on_balanced_support():
    """Swapping the side scales mirrors an intrinsic balanced kernel."""
    support = np.arange(-6, 7)
    learned = gaussian_weights(support, 0.4, 3.4)
    mirrored = gaussian_weights(support, 3.4, 0.4)

    assert mirrored == pytest.approx(learned[::-1])
    assert np.sqrt((0.4**2 + 3.4**2) / 2.0) == pytest.approx(
        np.sqrt((3.4**2 + 0.4**2) / 2.0)
    )


def test_free_ratio_fit_recovers_leftward_reference_with_bounded_sigmas():
    """The bounded two-scale fit permits and recovers a ratio below one."""
    support = np.arange(-8, 9)
    reference = gaussian_weights(support, 2.5, 0.5)

    fitted = fit_ob1_gaussian_prior(reference, support)

    assert 0.05 <= fitted["sigma_left"] <= 30.0
    assert 0.05 <= fitted["sigma_right"] <= 30.0
    assert fitted["right_left_ratio"] < 1.0
    assert fitted["sigma_left"] == pytest.approx(2.5, rel=1e-4)
    assert fitted["sigma_right"] == pytest.approx(0.5, rel=1e-4)


def test_profile_comparison_does_not_use_et1_trt_magnitudes():
    """Kernel alignment is invariant to unused ET1 prediction columns."""
    passages, tokens, fixations = synthetic_profile_inputs()
    tokens["predicted_trt"] = np.linspace(0.0, 1.0, len(tokens))
    sigma_record = {
        "checkpoint_id": "learned",
        "sigma_left": 0.4,
        "sigma_right": 3.4,
    }
    first = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        sigma_record,
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
    )
    changed = tokens.copy()
    changed["predicted_trt"] = np.linspace(1000.0, -1000.0, len(changed))
    second = compare_attention_profiles(
        passages,
        changed,
        fixations,
        sigma_record,
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
    )

    pd.testing.assert_frame_equal(
        first["passage_metrics"],
        second["passage_metrics"],
    )


def test_fixation_matched_support_differs_from_legacy_global_at_boundary():
    """Candidates share every fixation's truncated support in primary mode."""
    passages, tokens, fixations = synthetic_profile_inputs()
    boundary_rows = fixations["fixation_counter"].eq(0)
    fixations.loc[boundary_rows, "word_id"] = 0
    fixations.loc[boundary_rows, "word"] = "alpha"
    fixations.loc[boundary_rows, "eye_position"] = 2.0
    sigma_record = {
        "checkpoint_id": "learned",
        "sigma_left": 0.4,
        "sigma_right": 3.4,
    }

    matched = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        sigma_record,
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
    )
    legacy = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        sigma_record,
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
        candidate_support_policy="global",
    )

    matched_symmetric = matched["directionality"].query(
        "method == 'fixed_symmetric_sigma1'"
    ).iloc[0]
    legacy_symmetric = legacy["directionality"].query(
        "method == 'fixed_symmetric_sigma1'"
    ).iloc[0]
    assert matched_symmetric["right_share_of_noncenter_mass"] > (
        legacy_symmetric["right_share_of_noncenter_mass"]
    )
    matched_metrics = matched["result_table"].set_index("method")
    legacy_metrics = legacy["result_table"].set_index("method")
    assert matched_metrics.loc[
        "fixed_symmetric_sigma1",
        "js_divergence",
    ] != pytest.approx(
        legacy_metrics.loc[
            "fixed_symmetric_sigma1",
            "js_divergence",
        ]
    )
    assert matched_metrics.loc[
        "raw_delta",
        "js_divergence",
    ] == pytest.approx(
        legacy_metrics.loc["raw_delta", "js_divergence"]
    )
    assert matched["audit"]["candidate_support_policy_is_primary"]
    assert legacy["audit"]["candidate_support_policy_legacy_sensitivity"]
    assert legacy["fixed_priors"][0]["candidate_support_policy"] == "global"

    _, support, passage_patterns, _ = project_ob1_attention_to_t5(
        passages,
        tokens,
        fixations,
        attention_skews=(3.0,),
        fixation_weighting="duration",
        profile_component="focused",
    )

    def matched_rightward_share(
        sigma_left: float,
        sigma_right: float,
    ) -> float:
        """Compute the expected share from exact fixation supports."""
        left = 0.0
        right = 0.0
        for patterns in passage_patterns.values():
            for offsets, pattern_weight in patterns.items():
                local_offsets = np.asarray(offsets, dtype=int)
                local = gaussian_weights(
                    local_offsets,
                    sigma_left,
                    sigma_right,
                )
                left += float(pattern_weight) * float(
                    local[local_offsets < 0].sum()
                )
                right += float(pattern_weight) * float(
                    local[local_offsets > 0].sum()
                )
        return right / (left + right)

    def global_rightward_share(
        sigma_left: float,
        sigma_right: float,
    ) -> float:
        """Compute the expected share from one global normalization."""
        profile = gaussian_weights(support, sigma_left, sigma_right)
        left = float(profile[support < 0].sum())
        right = float(profile[support > 0].sum())
        return right / (left + right)

    symmetric_sigma = 1.0
    matched_directionality = matched["directionality"].set_index("method")
    legacy_directionality = legacy["directionality"].set_index("method")
    matched_prior = matched["fixed_priors"][0]
    legacy_prior = legacy["fixed_priors"][0]
    matched_sigmas = {
        "fixed_symmetric_sigma1": (
            symmetric_sigma,
            symmetric_sigma,
        ),
        "learned_asymmetric": (0.4, 3.4),
        "fixed_ob1_gaussian": (
            matched_prior["sigma_left"],
            matched_prior["sigma_right"],
        ),
    }
    legacy_sigmas = {
        **matched_sigmas,
        "fixed_ob1_gaussian": (
            legacy_prior["sigma_left"],
            legacy_prior["sigma_right"],
        ),
    }
    for method, (sigma_left, sigma_right) in matched_sigmas.items():
        assert matched_directionality.loc[
            method,
            "right_share_of_noncenter_mass",
        ] == pytest.approx(
            matched_rightward_share(sigma_left, sigma_right)
        )
    for method, (sigma_left, sigma_right) in legacy_sigmas.items():
        assert legacy_directionality.loc[
            method,
            "right_share_of_noncenter_mass",
        ] == pytest.approx(
            global_rightward_share(sigma_left, sigma_right)
        )


def test_candidate_support_policy_rejects_unknown_value():
    """The primary-versus-legacy support contract is closed."""
    passages, tokens, fixations = synthetic_profile_inputs()
    with pytest.raises(ValueError, match="candidate support policy"):
        compare_attention_profiles(
            passages,
            tokens,
            fixations,
            {
                "checkpoint_id": "learned",
                "sigma_left": 0.4,
                "sigma_right": 3.4,
            },
            candidate_support_policy="unknown",
        )


def test_profile_comparison_sweeps_multiple_sigmas_on_one_ob1_projection(
    tmp_path,
):
    """Every sigma pair receives separate metrics and a separate plot."""
    passages, tokens, fixations = synthetic_profile_inputs()
    records = [
        {
            "checkpoint_id": "rightward",
            "source_accuracy": 0.77,
            "sigma_left": 0.4,
            "sigma_right": 3.4,
        },
        {
            "checkpoint_id": "leftward",
            "source_accuracy": 0.75,
            "sigma_left": 3.5,
            "sigma_right": 0.7,
        },
    ]

    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        records,
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
    )
    write_attention_profile_outputs(tmp_path, artifacts)

    assert artifacts["audit"]["checkpoint_count"] == 2
    assert set(artifacts["result_table"]["checkpoint_id"]) == {
        "rightward",
        "leftward",
    }
    assert len(artifacts["fixed_priors"]) == 1
    assert len(artifacts["passage_metrics"]) == 32
    assert (
        tmp_path / "kernel_profile_plots/rightward.png"
    ).is_file()
    assert (
        tmp_path / "kernel_profile_plots/leftward.png"
    ).is_file()
    assert (
        tmp_path / "kernel_metric_plots/rightward.png"
    ).is_file()
    assert (
        tmp_path / "kernel_region_plots/leftward.png"
    ).is_file()
    directionality = artifacts["directionality"]
    rightward = directionality.query(
        "checkpoint_id == 'rightward' "
        "and method == 'learned_asymmetric'"
    ).iloc[0]
    leftward = directionality.query(
        "checkpoint_id == 'leftward' "
        "and method == 'learned_asymmetric'"
    ).iloc[0]
    assert rightward["right_mass"] > rightward["left_mass"]
    assert leftward["right_mass"] < leftward["left_mass"]


def test_sigma_landscape_writes_every_metric_and_landmark(tmp_path):
    """The optional landscape is exhaustive, finite, and visibly annotated."""
    passages, tokens, fixations = synthetic_profile_inputs()
    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        {
            "checkpoint_id": "learned",
            "sigma_left": 0.4,
            "sigma_right": 3.4,
        },
        attention_skews=(3.0, 4.0),
        bootstrap_samples=100,
        seed=7,
        landscape_sigma_values=np.geomspace(0.05, 30.0, 3),
    )
    write_attention_profile_outputs(tmp_path, artifacts)

    landscape = artifacts["sigma_landscape"]
    assert len(landscape) == 2 * 3 * 3
    assert artifacts["audit"]["sigma_landscape_min"] == pytest.approx(0.05)
    assert artifacts["audit"]["sigma_landscape_max"] == pytest.approx(30.0)
    assert artifacts["audit"]["sigma_landscape_points"] == 3
    metric_columns = [
        "profile_spearman",
        "js_divergence",
        "hellinger_distance",
        "total_variation_distance",
        "overlap_coefficient",
        "token_offset_wasserstein",
    ]
    assert np.isfinite(landscape[metric_columns].to_numpy()).all()
    assert landscape["overlap_coefficient"].to_numpy() == pytest.approx(
        1.0 - landscape["total_variation_distance"].to_numpy()
    )
    assert (tmp_path / "sigma_landscape.csv").is_file()
    assert (tmp_path / "sigma_landscape_skew_3.png").is_file()
    assert (tmp_path / "sigma_landscape_skew_4.png").is_file()

    outside = artifacts["parameter_diagnostics"].copy()
    marked_index = outside["landscape_marker"].notna().idxmax()
    outside.loc[marked_index, "sigma_right"] = 31.0
    with pytest.raises(
        ValueError,
        match="does not contain every plotted landmark",
    ):
        plot_sigma_landscape(
            landscape,
            outside,
            tmp_path / "invalid_landscape",
            "learned",
        )

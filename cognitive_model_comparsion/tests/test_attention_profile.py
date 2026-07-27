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
    PROFILE_MEAN_CI_SCOPE,
    RIGHTWARD_SHARE_SCOPE,
    build_t5_letter_geometry,
    compare_attention_profiles,
    effective_ob1_attention_width,
    gaussian_weights,
    ob1_reference_display_name,
    ob1_attention_weight,
    project_ob1_attention_to_t5,
    write_attention_profile_outputs,
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


def test_profile_comparison_writes_five_methods_and_checked_prior(tmp_path):
    """The comparison emits both symmetric controls and a checked prior."""
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
        "fixed_ob1_gaussian",
        "learned_asymmetric",
    }
    result_table = artifacts["result_table"]
    assert set(result_table["fixed_symmetric_sigma"]) == {1.0}
    rms_scales = result_table[
        "rms_side_scale_symmetric_sigma"
    ].drop_duplicates()
    assert len(rms_scales) == 1
    assert rms_scales.iloc[0] == pytest.approx(
        np.sqrt((0.4**2 + 3.4**2) / 2)
    )
    no_redistribution = result_table.loc[
        result_table["method"].eq("raw_delta")
    ].iloc[0]
    assert no_redistribution["display_name"] == (
        "No redistribution "
        "(all allocation weight remains at the source token)"
    )
    fitted = result_table.loc[
        result_table["method"].eq("fixed_ob1_gaussian")
    ].iloc[0]
    assert fitted["display_name"] == (
        "Descriptive Gaussian fitted to the same OB1 profile"
    )
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
    assert audit["candidate_support_policy"] == "fixation_matched"
    assert audit["candidate_support_policy_is_primary"]
    assert not audit["candidate_support_policy_legacy_sensitivity"]
    assert audit["fixation_support_patterns_aggregated"]
    assert not audit["actual_et1_trt_magnitudes_used"]
    assert set(artifacts["passage_metrics"]["passage_id_zero_based"]) == {
        0,
        1,
    }
    prior = json.loads(
        (tmp_path / "fixed_ob1_priors.json").read_text()
    )[0]
    assert prior["profile_component"] == "focused"
    assert prior["projection_coordinate"] == (
        "relative_native_t5_token_index"
    )
    assert prior["candidate_support_policy"] == "fixation_matched"
    assert prior["sigma_right"] > prior["sigma_left"] > 0
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
        "learned_asymmetric",
        "ob1_attention_profile",
    }
    assert set(reviewer_summary["reference_profile"]) == {
        reference_label
    }
    assert set(reviewer_summary["rightward_share_scope"]) == {
        RIGHTWARD_SHARE_SCOPE
    }
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
    assert len(artifacts["passage_metrics"]) == 20
    assert (
        tmp_path / "kernel_profile_plots/rightward.png"
    ).is_file()
    assert (
        tmp_path / "kernel_profile_plots/leftward.png"
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

"""Tests for the OB1-attention to native-T5 projection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cognitive_model_comparsion.src.attention_profile import (
    build_t5_letter_geometry,
    compare_attention_profiles,
    effective_ob1_attention_width,
    ob1_attention_weight,
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


def test_profile_comparison_writes_four_methods_and_checked_prior(tmp_path):
    """The comparison emits complete tables and a focused fixed prior."""
    passages, tokens, fixations = synthetic_profile_inputs()
    artifacts = compare_attention_profiles(
        passages,
        tokens,
        fixations,
        {
            "checkpoint_id": "learned",
            "sigma_left": 0.4,
            "sigma_right": 3.4,
            "sigma_symmetric": np.sqrt((0.4**2 + 3.4**2) / 2),
        },
        attention_skews=(3.0,),
        bootstrap_samples=100,
        seed=7,
    )
    write_attention_profile_outputs(tmp_path, artifacts)

    assert set(artifacts["passage_metrics"]["method"]) == {
        "raw_delta",
        "width_matched_symmetric",
        "fixed_ob1_gaussian",
        "learned_asymmetric",
    }
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
    assert prior["sigma_right"] > prior["sigma_left"] > 0
    assert (
        tmp_path / "kernel_alignment_contrasts.csv"
    ).is_file()
    directionality = pd.read_csv(
        tmp_path / "kernel_directionality.csv"
    )
    learned = directionality.loc[
        directionality["method"].eq("learned_asymmetric")
    ].iloc[0]
    assert learned["right_mass"] > learned["left_mass"]
    assert (tmp_path / "kernel_profiles.png").is_file()

"""Test the ET1-free Human Provo fixation-profile analysis."""

import numpy as np
import pandas as pd

from cognitive_model_comparsion.scripts.build_provo_human_fixation_profile import (
    OFFSET_VALUES,
    align_one_passage,
    bootstrap_profile_summary,
    build_unit_profiles,
)
from cognitive_model_comparsion.src.prepare_provo import normalize_ob1_word


def area_frame(labels: list[str]) -> pd.DataFrame:
    """Build one ordered interest-area inventory for alignment tests."""
    return pd.DataFrame(
        {
            "passage_id_raw": 1,
            "interest_area_index": np.arange(1, len(labels) + 1),
            "interest_area_label": labels,
            "normalized_label": [
                normalize_ob1_word(label) for label in labels
            ],
        }
    )


def test_align_one_passage_marks_merged_interest_area() -> None:
    """Merged display regions must be excluded rather than split arbitrarily."""
    result = align_one_passage(
        1,
        "alpha beta gamma delta",
        area_frame(["alpha", "beta--gamma", "delta"]),
    )
    assert result["mapping_status"].tolist() == [
        "one_to_one",
        "merged_multiple_words",
        "one_to_one",
    ]
    assert result["word_id_zero_based"].iloc[0] == 0
    assert np.isnan(result["word_id_zero_based"].iloc[1])
    assert result["word_id_zero_based"].iloc[2] == 3


def test_align_one_passage_marks_extra_interest_area() -> None:
    """A display artifact absent from canonical text must remain unmapped."""
    result = align_one_passage(
        1,
        "alpha beta",
        area_frame(["alpha", "Ñ", "beta"]),
    )
    assert result["mapping_status"].tolist() == [
        "one_to_one",
        "extra_noncanonical_area",
        "one_to_one",
    ]
    assert result["word_id_zero_based"].iloc[-1] == 1


def test_unit_profiles_preserve_all_tail_mass() -> None:
    """Clipped edge bins must retain a normalized probability distribution."""
    transitions = pd.DataFrame(
        {
            "source": ["Human Provo"] * 6,
            "unit_id": ["reader_1"] * 3 + ["reader_2"] * 3,
            "display_offset": [-3, 0, 6, -2, 1, 2],
            "direction": [
                "Regressive",
                "Refixation",
                "Progressive",
                "Regressive",
                "Progressive",
                "Progressive",
            ],
        }
    )
    profiles = build_unit_profiles(transitions)
    offset_columns = [
        f"offset_{int(offset)}" for offset in OFFSET_VALUES
    ]
    assert np.allclose(profiles[offset_columns].sum(axis=1), 1.0)
    offsets, directions = bootstrap_profile_summary(
        profiles,
        bootstrap_samples=100,
        seed=7,
    )
    assert np.isclose(offsets["mean_probability"].sum(), 1.0)
    assert np.isclose(directions["mean_probability"].sum(), 1.0)

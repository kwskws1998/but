"""Tests for the canonical Provo table builder."""

import pytest

from cognitive_model_comparsion.src.prepare_provo import (
    EYE_FILENAME,
    KNOWN_ALIGNMENT_EXCEPTIONS,
    PREDICTABILITY_FILENAME,
    RAW_DIR,
    build_canonical_tables,
    normalize_ob1_word,
)


def test_ob1_word_normalization_matches_upstream_behavior():
    """Punctuation is removed and case is lowered as in OB1."""
    assert normalize_ob1_word("Women's") == "womens"
    assert normalize_ob1_word("90%") == "90"


@pytest.mark.skipif(
    not (RAW_DIR / EYE_FILENAME).is_file()
    or not (RAW_DIR / PREDICTABILITY_FILENAME).is_file(),
    reason="Official Provo assets are not installed",
)
def test_canonical_provo_contract():
    """The official files produce the frozen 55-passage evaluation grid."""
    passages, words, excluded, audit = build_canonical_tables(
        RAW_DIR / EYE_FILENAME,
        RAW_DIR / PREDICTABILITY_FILENAME,
    )

    assert len(passages) == 55
    assert len(words) == 2686
    assert len(excluded) == 59
    assert audit["reader_count_min"] == audit["reader_count_max"] == 84
    assert (
        audit["alignment_status_counts"]["published_known_exception"]
        == len(KNOWN_ALIGNMENT_EXCEPTIONS)
        == 4
    )
    assert not audit["canonical_only_positions"]
    assert not audit["human_only_positions"]

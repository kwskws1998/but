"""Unit tests for published OB1 position corrections."""

from cognitive_model_comparsion.src.audit_provo import (
    corrected_ob1_eye_position,
)


def test_misplaced_evolution_position_is_corrected():
    """The duplicated passage-18 position is separated as in OB1 evaluation."""
    assert corrected_ob1_eye_position(18, 3, "evolution") == (17, 50)
    assert corrected_ob1_eye_position(18, 3, "the") == (17, 2)


def test_passage_three_and_thirteen_shifts_are_corrected():
    """The two published index-shift ranges map to model coordinates."""
    assert corrected_ob1_eye_position(3, 46, "token") == (2, 44)
    assert corrected_ob1_eye_position(13, 20, "token") == (12, 18)


def test_unaffected_position_only_becomes_zero_based():
    """A regular raw position receives only the one-based to zero-based shift."""
    assert corrected_ob1_eye_position(1, 2, "are") == (0, 1)

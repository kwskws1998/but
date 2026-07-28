"""Tests for bounded symmetric allocation-distribution metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cognitive_model_comparsion.src.distribution_metrics import (
    distribution_similarity_metrics,
)


def test_identical_distributions_have_complete_overlap():
    """Identical vectors have zero distances and unit overlap."""
    metrics = distribution_similarity_metrics(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )

    assert metrics["hellinger_distance"] == pytest.approx(0.0)
    assert metrics["total_variation_distance"] == pytest.approx(0.0)
    assert metrics["overlap_coefficient"] == pytest.approx(1.0)


def test_disjoint_distributions_have_no_overlap():
    """Disjoint point masses attain both maximum distances."""
    metrics = distribution_similarity_metrics(
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    )

    assert metrics["hellinger_distance"] == pytest.approx(1.0)
    assert metrics["total_variation_distance"] == pytest.approx(1.0)
    assert metrics["overlap_coefficient"] == pytest.approx(0.0)


def test_partial_overlap_matches_closed_form_values():
    """A half-shared point mass has the expected bounded metric values."""
    metrics = distribution_similarity_metrics(
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
    )

    assert metrics["hellinger_distance"] == pytest.approx(
        np.sqrt(1.0 - np.sqrt(0.5))
    )
    assert metrics["total_variation_distance"] == pytest.approx(0.5)
    assert metrics["overlap_coefficient"] == pytest.approx(0.5)
    assert (
        metrics["overlap_coefficient"]
        + metrics["total_variation_distance"]
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        (
            np.array([1.0, -1.0]),
            np.array([1.0, 0.0]),
            "Reference distribution contains negative values",
        ),
        (
            np.array([1.0, 0.0]),
            np.array([np.nan, 1.0]),
            "Candidate distribution contains non-finite values",
        ),
        (
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            "Reference distribution has zero mass",
        ),
        (
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "same shape",
        ),
    ],
)
def test_invalid_distribution_pairs_are_rejected(
    reference,
    candidate,
    message,
):
    """Invalid inputs fail without hidden clipping, smoothing, or padding."""
    with pytest.raises(ValueError, match=message):
        distribution_similarity_metrics(reference, candidate)

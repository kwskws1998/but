"""Tests for ET1 offset alignment and fixed redistribution."""

import math

import pytest
import torch

from cognitive_model_comparsion.src.et1_inference import (
    aggregate_tokens_to_words,
    assign_token_offsets,
    redistribute_values,
    validate_mass_conservation,
)


def test_token_offsets_map_subtokens_to_whitespace_words():
    """Multiple subtokens can sum into one exact passage word."""
    assignments, spans = assign_token_offsets(
        "One two.",
        [(0, 3), (4, 7), (7, 8), (0, 0)],
        [0, 0, 0, 1],
    )

    assert assignments == [0, 1, 1, None]
    assert len(spans) == 2
    assert aggregate_tokens_to_words(
        torch.tensor([1.0, 2.0, 3.0, 9.0]).numpy(),
        assignments,
        2,
    ) == {0: 1.0, 1: 5.0}


def test_symmetric_and_asymmetric_redistribution_conserve_mass():
    """Both fixed kernels preserve valid-token total TRT."""
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    record = {
        "log_sigma_left": math.log(1.5),
        "log_sigma_right": math.log(2.5),
        "min_sigma": 1e-6,
        "sigma_symmetric": math.sqrt((1.500001**2 + 2.500001**2) / 2),
    }

    symmetric, asymmetric = redistribute_values(values, mask, record)

    assert validate_mass_conservation(values, symmetric, mask) == pytest.approx(
        0.0,
        abs=1e-5,
    )
    assert validate_mass_conservation(
        values,
        asymmetric,
        mask,
    ) == pytest.approx(0.0, abs=1e-5)
    assert not torch.allclose(symmetric, asymmetric)


def test_non_special_empty_offset_is_rejected():
    """Only special tokens may use an empty character offset."""
    with pytest.raises(ValueError, match="empty offset"):
        assign_token_offsets("word", [(0, 0)], [0])

"""Tests for learned sigma extraction."""

import math

import pytest
import torch

from cognitive_model_comparsion.src.sigmas import (
    direct_sigma_record,
    extract_sigma_record,
)


def test_extract_sigma_record_and_fixed_symmetric_control(tmp_path):
    """The fixed SymGaussian remains independent of learned checkpoint widths."""
    checkpoint = tmp_path / "adapter_model.bin"
    prefix = (
        "base_model.model.asym_gaussian_redistributor."
        "modules_to_save.default."
    )
    torch.save(
        {
            f"{prefix}log_sigma_left": torch.tensor(math.log(1.5)),
            f"{prefix}log_sigma_right": torch.tensor(math.log(2.5)),
        },
        checkpoint,
    )

    record = extract_sigma_record(checkpoint)

    assert record["sigma_left"] == pytest.approx(1.500001)
    assert record["sigma_right"] == pytest.approx(2.500001)
    assert record["sigma_symmetric"] == pytest.approx(1.0)
    assert record["sigma_symmetric_fixed"] == pytest.approx(1.0)
    assert record["sigma_symmetric_rms_scale"] == pytest.approx(
        math.sqrt((1.500001**2 + 2.500001**2) / 2)
    )
    assert record["symmetric_sigma_source"] == "fixed_independent_control"
    assert record["selected_sigma_prefix"] == prefix


def test_initial_like_sigma_requires_explicit_confirmation(tmp_path):
    """Unchanged 1.0/1.0 widths cannot silently represent a trained model."""
    checkpoint = tmp_path / "adapter_model.bin"
    torch.save(
        {
            "redistributor.log_sigma_left": torch.tensor(0.0),
            "redistributor.log_sigma_right": torch.tensor(0.0),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="initial-like"):
        extract_sigma_record(checkpoint)

    record = extract_sigma_record(checkpoint, allow_initial_sigmas=True)
    assert record["initial_like"]


def test_ambiguous_sigma_pairs_require_prefix(tmp_path):
    """Multiple model copies require an explicit state-key prefix."""
    checkpoint = tmp_path / "adapter_model.bin"
    torch.save(
        {
            "a.log_sigma_left": torch.tensor(math.log(1.2)),
            "a.log_sigma_right": torch.tensor(math.log(1.3)),
            "b.log_sigma_left": torch.tensor(math.log(1.4)),
            "b.log_sigma_right": torch.tensor(math.log(1.5)),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="unambiguous"):
        extract_sigma_record(checkpoint)

    record = extract_sigma_record(checkpoint, requested_prefix="b.")
    assert record["sigma_left"] == pytest.approx(1.400001)


def test_direct_effective_sigma_record_reconstructs_production_parameters():
    """Confirmed effective widths can run without unrelated RM weights."""
    record = direct_sigma_record(
        checkpoint_id="llama8b_seed42",
        sigma_left=0.41553,
        sigma_right=3.46115,
        source_accuracy=0.76675,
    )

    assert record["sigma_left"] == pytest.approx(0.41553)
    assert record["sigma_right"] == pytest.approx(3.46115)
    assert record["log_sigma_left"] == pytest.approx(
        math.log(0.41553 - 1e-6)
    )
    assert record["sigma_symmetric"] == pytest.approx(1.0)
    assert record["sigma_symmetric_fixed"] == pytest.approx(1.0)
    assert record["sigma_symmetric_rms_scale"] == pytest.approx(
        math.sqrt((0.41553**2 + 3.46115**2) / 2)
    )
    assert record["symmetric_sigma_source"] == "fixed_independent_control"
    assert record["source_accuracy"] == pytest.approx(0.76675)

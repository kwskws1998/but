"""Tests for loading a frozen OB1-fitted RM redistribution prior."""

from __future__ import annotations

import json

import pytest

from models.asym_gaussian_redistributor import AsymGaussianRedistributor
from utils.fixed_ob1_prior import load_fixed_ob1_prior


def write_priors(
    path,
    component="focused",
    candidate_support_policy="fixation_matched",
):
    """Write two skew-specific fixed-prior records."""
    path.write_text(
        json.dumps(
            [
                {
                    "ob1_attention_skew": 3.0,
                    "profile_component": component,
                    "fixation_weighting": "duration",
                    "projection_coordinate": (
                        "relative_native_t5_token_index"
                    ),
                    "candidate_support_policy": candidate_support_policy,
                    "sigma_left": 0.4,
                    "sigma_right": 1.2,
                    "fit_js_divergence": 0.01,
                },
                {
                    "ob1_attention_skew": 4.0,
                    "profile_component": component,
                    "fixation_weighting": "duration",
                    "projection_coordinate": (
                        "relative_native_t5_token_index"
                    ),
                    "candidate_support_policy": candidate_support_policy,
                    "sigma_left": 0.3,
                    "sigma_right": 1.3,
                    "fit_js_divergence": 0.02,
                },
            ]
        ),
        encoding="utf-8",
    )


def test_loader_selects_exact_skew_and_records_hash(tmp_path):
    """A valid focused prior is selected with file provenance."""
    path = tmp_path / "fixed_ob1_priors.json"
    write_priors(path)

    prior = load_fixed_ob1_prior(path, 3.0)

    assert prior["sigma_left"] == pytest.approx(0.4)
    assert prior["sigma_right"] == pytest.approx(1.2)
    assert prior["initializer_sigma_left"] == pytest.approx(0.399999)
    assert prior["initializer_sigma_right"] == pytest.approx(1.199999)
    assert prior["right_left_ratio"] == pytest.approx(3.0)
    assert len(prior["sha256"]) == 64
    assert prior["profile_component"] == "focused"
    assert prior["candidate_support_policy"] == "fixation_matched"

    redistributor = AsymGaussianRedistributor(
        prior["initializer_sigma_left"],
        prior["initializer_sigma_right"],
    )
    assert float(redistributor.sigma_left.detach()) == pytest.approx(0.4)
    assert float(redistributor.sigma_right.detach()) == pytest.approx(1.2)


def test_loader_rejects_full_residual_profile_for_rm(tmp_path):
    """The full residual sensitivity cannot silently become an RM prior."""
    path = tmp_path / "fixed_ob1_priors.json"
    write_priors(path, component="full")

    with pytest.raises(ValueError, match="focused OB1 attention"):
        load_fixed_ob1_prior(path, 3.0)


def test_loader_rejects_global_support_prior_for_rm(tmp_path):
    """The legacy global-support fit cannot silently become an RM prior."""
    path = tmp_path / "fixed_ob1_priors.json"
    write_priors(path, candidate_support_policy="global")

    with pytest.raises(ValueError, match="fixation-matched candidate support"):
        load_fixed_ob1_prior(path, 3.0)


def test_loader_rejects_legacy_prior_without_support_policy(tmp_path):
    """A prior without an explicit support policy is not reproducible."""
    path = tmp_path / "fixed_ob1_priors.json"
    write_priors(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        record.pop("candidate_support_policy")
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="fixation-matched candidate support"):
        load_fixed_ob1_prior(path, 3.0)

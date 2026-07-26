"""Tests for ET1 offset alignment and fixed redistribution."""

import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from cognitive_model_comparsion.src.et1_inference import (
    aggregate_tokens_to_words,
    assign_token_offsets,
    build_redistribution_attention_mask,
    redistribution_mass_audit,
    redistribute_values,
    run_et1_inference,
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


def test_redistribution_mask_defaults_to_production_special_token_inclusion():
    """The default mask matches training while sensitivity can exclude EOS."""
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    special_tokens_mask = [0, 0, 1, 1]

    production_mask = build_redistribution_attention_mask(
        attention_mask,
        special_tokens_mask,
    )
    sensitivity_mask = build_redistribution_attention_mask(
        attention_mask,
        special_tokens_mask,
        include_special_tokens=False,
    )

    assert torch.equal(production_mask, attention_mask)
    assert torch.equal(
        sensitivity_mask,
        torch.tensor([[1, 1, 0, 0]]),
    )


def test_eos_leakage_and_word_mass_retention_are_auditable():
    """Including EOS exposes leakage while excluding it retains word mass."""
    values = torch.tensor([[1.0, 0.0]])
    attention_mask = torch.tensor([[1, 1]])
    special_tokens_mask = [0, 1]
    assignments = [0, None]
    record = {
        "log_sigma_left": math.log(1.5),
        "log_sigma_right": math.log(2.5),
        "min_sigma": 1e-6,
        "sigma_symmetric": math.sqrt((1.500001**2 + 2.500001**2) / 2),
    }

    production_mask = build_redistribution_attention_mask(
        attention_mask,
        special_tokens_mask,
    )
    production_symmetric, _ = redistribute_values(
        values,
        production_mask,
        record,
    )
    production_audit = redistribution_mass_audit(
        values,
        production_symmetric,
        attention_mask,
        production_mask,
        special_tokens_mask,
        assignments,
    )

    sensitivity_mask = build_redistribution_attention_mask(
        attention_mask,
        special_tokens_mask,
        include_special_tokens=False,
    )
    sensitivity_symmetric, _ = redistribute_values(
        values,
        sensitivity_mask,
        record,
    )
    sensitivity_audit = redistribution_mass_audit(
        values,
        sensitivity_symmetric,
        attention_mask,
        sensitivity_mask,
        special_tokens_mask,
        assignments,
    )

    assert production_audit["valid_token_mass_difference"] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert (
        production_audit["redistributed_unassigned_special_mass"] > 0.0
    )
    assert production_audit["word_mass_retention_fraction"] < 1.0
    assert sensitivity_audit["valid_token_mass_difference"] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert sensitivity_audit[
        "redistributed_unassigned_special_mass"
    ] == pytest.approx(0.0)
    assert sensitivity_audit["word_mass_retention_fraction"] == pytest.approx(
        1.0
    )
    assert production_audit["raw_evaluable_word_mass"] == pytest.approx(
        production_audit["raw_word_assigned_mass"]
    )
    assert production_audit[
        "redistributed_evaluable_word_mass"
    ] == pytest.approx(production_audit["redistributed_word_assigned_mass"])


def test_mass_audit_separates_evaluable_grid_from_all_assigned_words():
    """Canonical evaluation mass can exclude otherwise assigned passage words."""
    before = torch.tensor([[2.0, 3.0, 0.0]])
    after = torch.tensor([[1.0, 4.0, 0.0]])
    attention_mask = torch.tensor([[1, 1, 1]])
    assignments = [0, 1, None]

    audit = redistribution_mass_audit(
        before,
        after,
        attention_mask,
        attention_mask,
        [0, 0, 1],
        assignments,
        evaluable_word_ids={1},
    )

    assert audit["raw_word_assigned_mass"] == pytest.approx(5.0)
    assert audit["redistributed_word_assigned_mass"] == pytest.approx(5.0)
    assert audit["raw_evaluable_word_mass"] == pytest.approx(3.0)
    assert audit["redistributed_evaluable_word_mass"] == pytest.approx(4.0)
    assert audit[
        "raw_evaluable_mass_fraction_of_attention"
    ] == pytest.approx(3.0 / 5.0)
    assert audit[
        "redistributed_evaluable_mass_fraction_of_attention"
    ] == pytest.approx(4.0 / 5.0)
    assert audit["evaluable_mass_retention_vs_raw"] == pytest.approx(4.0 / 3.0)


class FakeET1Predictor:
    """Provide one word token and one EOS token for end-to-end auditing."""

    def __init__(self):
        self.predictor = SimpleNamespace(
            device=torch.device("cpu"),
            cache_signature="fake-et1",
        )

    def predict(self, text: str) -> dict:
        """Return deterministic native-token predictions for one passage."""
        assert text == "word"
        return {
            "values": torch.tensor([[1.0, 0.0]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "input_ids": torch.tensor([[10, 1]]),
            "tokens": ["word", "</s>"],
            "offsets": [(0, 4), (0, 0)],
            "special_tokens_mask": [0, 1],
            "word_assignments": [0, None],
            "word_spans": [
                {
                    "word_id_zero_based": 0,
                    "word_raw": "word",
                    "character_start": 0,
                    "character_end": 4,
                }
            ],
            "device": "cpu",
            "cache_signature": "fake-et1",
        }


class FakeET1PredictorWithExcludedFirst:
    """Provide two word tokens when only the second is evaluable."""

    def __init__(self):
        self.predictor = SimpleNamespace(
            device=torch.device("cpu"),
            cache_signature="fake-et1-excluded-first",
        )

    def predict(self, text: str) -> dict:
        """Return deterministic predictions with an excluded first word."""
        assert text == "first word"
        return {
            "values": torch.tensor([[2.0, 3.0, 0.0]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "input_ids": torch.tensor([[10, 11, 1]]),
            "tokens": ["first", "word", "</s>"],
            "offsets": [(0, 5), (6, 10), (0, 0)],
            "special_tokens_mask": [0, 0, 1],
            "word_assignments": [0, 1, None],
            "word_spans": [
                {
                    "word_id_zero_based": 0,
                    "word_raw": "first",
                    "character_start": 0,
                    "character_end": 5,
                },
                {
                    "word_id_zero_based": 1,
                    "word_raw": "word",
                    "character_start": 6,
                    "character_end": 10,
                },
            ],
            "device": "cpu",
            "cache_signature": "fake-et1-excluded-first",
        }


def test_run_inference_uses_canonical_words_as_evaluable_grid():
    """The run-level audit excludes words absent from canonical data."""
    passages = pd.DataFrame(
        [
            {
                "passage_id_zero_based": 0,
                "passage_text": "first word",
            }
        ]
    )
    canonical_words = pd.DataFrame(
        [
            {
                "passage_id_raw": 1,
                "passage_id_zero_based": 0,
                "word_id_zero_based": 1,
                "word_raw": "word",
            }
        ]
    )

    _, _, mass_frame, audit = run_et1_inference(
        passages,
        canonical_words,
        [],
        FakeET1PredictorWithExcludedFirst(),
    )

    raw_audit = mass_frame.iloc[0]
    assert audit["canonical_words"] == 1
    assert audit["metric_evaluable_words"] == 1
    assert raw_audit["condition"] == "et1_raw"
    assert raw_audit["raw_word_assigned_mass"] == pytest.approx(5.0)
    assert raw_audit["raw_evaluable_word_mass"] == pytest.approx(3.0)
    assert raw_audit[
        "raw_evaluable_mass_fraction_of_attention"
    ] == pytest.approx(3.0 / 5.0)


def test_run_inference_excludes_ob1_incompatible_words_only_from_metrics():
    """OB1-incompatible canonical words remain output rows but not metric mass."""
    passages = pd.DataFrame(
        [
            {
                "passage_id_zero_based": 0,
                "passage_text": "first word",
            }
        ]
    )
    canonical_words = pd.DataFrame(
        [
            {
                "passage_id_raw": 1,
                "passage_id_zero_based": 0,
                "word_id_zero_based": 0,
                "word_raw": "first",
                "ob1_evaluable": False,
            },
            {
                "passage_id_raw": 1,
                "passage_id_zero_based": 0,
                "word_id_zero_based": 1,
                "word_raw": "word",
                "ob1_evaluable": True,
            },
        ]
    )

    _, word_frame, mass_frame, audit = run_et1_inference(
        passages,
        canonical_words,
        [],
        FakeET1PredictorWithExcludedFirst(),
    )

    assert len(word_frame) == 2
    assert set(word_frame["word_id_zero_based"]) == {0, 1}
    assert audit["canonical_words"] == 2
    assert audit["metric_evaluable_words"] == 1
    raw_audit = mass_frame.iloc[0]
    assert raw_audit["raw_word_assigned_mass"] == pytest.approx(5.0)
    assert raw_audit["raw_evaluable_word_mass"] == pytest.approx(3.0)


def test_run_inference_records_mass_by_checkpoint_passage_and_condition():
    """Inference writes raw and redistributed mass under either mask policy."""
    passages = pd.DataFrame(
        [
            {
                "passage_id_zero_based": 0,
                "passage_text": "word",
            }
        ]
    )
    canonical_words = pd.DataFrame(
        [
            {
                "passage_id_raw": 1,
                "passage_id_zero_based": 0,
                "word_id_zero_based": 0,
                "word_raw": "word",
            }
        ]
    )
    sigma_records = [
        {
            "checkpoint_id": "checkpoint-1",
            "checkpoint": "checkpoint.pt",
            "log_sigma_left": math.log(1.5),
            "log_sigma_right": math.log(2.5),
            "min_sigma": 1e-6,
            "sigma_symmetric": math.sqrt(
                (1.500001**2 + 2.500001**2) / 2
            ),
        }
    ]

    production = run_et1_inference(
        passages,
        canonical_words,
        sigma_records,
        FakeET1Predictor(),
    )
    sensitivity = run_et1_inference(
        passages,
        canonical_words,
        sigma_records,
        FakeET1Predictor(),
        include_special_tokens_in_redistribution=False,
    )

    production_tokens, _, production_mass, production_audit = production
    sensitivity_tokens, _, sensitivity_mass, sensitivity_audit = sensitivity
    assert set(production_mass["condition"]) == {
        "et1_raw",
        "et1_symmetric",
        "et1_asymmetric",
    }
    assert set(production_mass["checkpoint_id"]) == {"checkpoint-1"}
    assert set(production_mass["passage_id_zero_based"]) == {0}
    assert production_audit["mass_checks"] == 3
    assert production_audit["redistribution_special_token_policy"] == (
        "include"
    )
    assert sensitivity_audit["redistribution_special_token_policy"] == (
        "exclude"
    )
    assert production_tokens.loc[
        production_tokens["is_special"].eq(1),
        "redistribution_mask",
    ].item() == 1
    assert sensitivity_tokens.loc[
        sensitivity_tokens["is_special"].eq(1),
        "redistribution_mask",
    ].item() == 0

    production_symmetric = production_mass.loc[
        production_mass["condition"].eq("et1_symmetric")
    ].iloc[0]
    sensitivity_symmetric = sensitivity_mass.loc[
        sensitivity_mass["condition"].eq("et1_symmetric")
    ].iloc[0]
    assert production_symmetric[
        "redistributed_unassigned_special_mass"
    ] > 0.0
    assert production_symmetric["word_mass_retention_fraction"] < 1.0
    assert production_symmetric[
        "raw_evaluable_word_mass"
    ] == pytest.approx(production_symmetric["raw_word_assigned_mass"])
    assert production_symmetric[
        "redistributed_evaluable_word_mass"
    ] == pytest.approx(
        production_symmetric["redistributed_word_assigned_mass"]
    )
    assert sensitivity_symmetric[
        "redistributed_unassigned_special_mass"
    ] == pytest.approx(0.0)
    assert sensitivity_symmetric[
        "word_mass_retention_fraction"
    ] == pytest.approx(1.0)

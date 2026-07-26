"""Tests for the single cognitive-comparison Python entry point."""

import argparse
import json
from pathlib import Path

import pytest

from cognitive_model_comparsion.main import (
    build_parser,
    parse_seed_specification,
    runtime_manifest,
    validate_predict_sigma_arguments,
    validate_trial_count,
)


def test_seed_parser_supports_range_and_explicit_values():
    """Half-open ranges and explicit lists resolve deterministically."""
    assert parse_seed_specification("0:3") == [0, 1, 2]
    assert parse_seed_specification("41,42,43") == [41, 42, 43]


def test_seed_parser_rejects_duplicates():
    """Virtual readers require unique random seeds."""
    with pytest.raises(ValueError, match="unique"):
        parse_seed_specification("1,1")


def test_parser_exposes_all_required_subcommands():
    """The documented Python-only commands are registered."""
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )
    assert set(subparser_action.choices) == {
        "setup",
        "audit",
        "prepare-provo",
        "extract-sigmas",
        "predict-et1",
        "simulate-ob1",
        "evaluate",
        "run",
    }


def test_trial_count_covers_only_published_provo_passages():
    """OB1 trial prefixes cannot extend beyond the 55 passages."""
    validate_trial_count(1)
    validate_trial_count(55)
    with pytest.raises(ValueError, match="between 1 and 55"):
        validate_trial_count(0)
    with pytest.raises(ValueError, match="between 1 and 55"):
        validate_trial_count(56)


def test_runtime_manifest_serializes_path_lists_and_omits_handler(tmp_path):
    """Full-run provenance must be directly JSON serializable."""
    args = argparse.Namespace(
        command="run",
        checkpoint=[Path("seed41"), Path("seed42")],
        handler=lambda _: None,
    )
    manifest = runtime_manifest(args, tmp_path)
    assert manifest["arguments"]["checkpoint"] == ["seed41", "seed42"]
    assert "handler" not in manifest["arguments"]
    json.dumps(manifest)


def test_predict_et1_rejects_two_sigma_sources():
    """Checkpoint and saved sigma JSON cannot compete silently."""
    args = argparse.Namespace(
        sigma_json=Path("sigmas.json"),
        checkpoint=[Path("checkpoint")],
        checkpoint_id=None,
        sigma_prefix=None,
        allow_initial_sigmas=False,
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_predict_sigma_arguments(args)


def test_parser_accepts_confirmed_direct_sigma_values():
    """A full run can use fixed effective sigmas without an RM checkpoint."""
    args = build_parser().parse_args(
        [
            "run",
            "--sigma-left",
            "0.41553",
            "--sigma-right",
            "3.46115",
            "--sigma-source-accuracy",
            "0.76675",
            "--checkpoint-id",
            "llama8b_seed42",
        ]
    )

    validate_predict_sigma_arguments(args)
    assert args.checkpoint == []
    assert args.sigma_left == [0.41553]
    assert args.sigma_right == [3.46115]

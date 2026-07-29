"""Tests for the single cognitive-comparison Python entry point."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import cognitive_model_comparsion.main as cognitive_main
from cognitive_model_comparsion.main import (
    build_parser,
    command_run,
    ensure_prepared,
    load_sigma_records,
    parse_seed_specification,
    runtime_manifest,
    validate_ob1_manifest_against_fixations,
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
        "prepare-onestop",
        "extract-sigmas",
        "predict-et1",
        "simulate-ob1",
        "evaluate",
        "compare-attention-profile",
        "run",
    }


def test_attention_profile_help_separates_kernel_shape_from_et1_trt():
    """The CLI states the estimand and the conditional bootstrap scope."""
    parser = build_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if action.dest == "command"
    )
    help_text = subparser_action.choices[
        "compare-attention-profile"
    ].format_help()
    normalized_help = " ".join(help_text.split())

    assert "not ET1-predicted TRT values" in normalized_help
    assert "not its predicted TRT magnitudes" in normalized_help
    assert "does not rerun OB1 trajectories" in normalized_help
    assert "simulation IDs are pooled" in normalized_help
    assert "--allow-missing-ob1-manifest" in normalized_help
    assert "--candidate-support-policy" in normalized_help
    assert "--with-support-centered-sd-controls" in normalized_help
    assert "exact visible offsets" in normalized_help

    default_args = parser.parse_args(
        [
            "compare-attention-profile",
            "--sigma-left",
            "0.4",
            "--sigma-right",
            "3.4",
            "--checkpoint-id",
            "selected",
            "--et1-dir",
            "outputs/et1",
            "--ob1-dir",
            "outputs/ob1",
            "--output-dir",
            "outputs/attention",
        ]
    )
    assert default_args.candidate_support_policy == "fixation_matched"
    assert not default_args.skip_support_rms_displacement_controls
    assert not default_args.with_support_centered_sd_controls
    assert not default_args.with_sigma_landscape
    assert default_args.landscape_sigma_min == pytest.approx(0.1)
    assert default_args.landscape_sigma_max == pytest.approx(5.0)
    assert default_args.landscape_points == 41
    legacy_args = parser.parse_args(
        [
            "compare-attention-profile",
            "--sigma-left",
            "0.4",
            "--sigma-right",
            "3.4",
            "--checkpoint-id",
            "selected",
            "--et1-dir",
            "outputs/et1",
            "--ob1-dir",
            "outputs/ob1",
            "--output-dir",
            "outputs/attention",
            "--candidate-support-policy",
            "global",
        ]
    )
    assert legacy_args.candidate_support_policy == "global"

    skipped_args = parser.parse_args(
        [
            "compare-attention-profile",
            "--sigma-left",
            "0.4",
            "--sigma-right",
            "3.4",
            "--checkpoint-id",
            "selected",
            "--et1-dir",
            "outputs/et1",
            "--ob1-dir",
            "outputs/ob1",
            "--output-dir",
            "outputs/attention",
            "--skip-support-rms-displacement-controls",
        ]
    )
    assert skipped_args.skip_support_rms_displacement_controls

    centered_sd_args = parser.parse_args(
        [
            "compare-attention-profile",
            "--sigma-left",
            "0.4",
            "--sigma-right",
            "3.4",
            "--checkpoint-id",
            "selected",
            "--et1-dir",
            "outputs/et1",
            "--ob1-dir",
            "outputs/ob1",
            "--output-dir",
            "outputs/attention",
            "--with-support-centered-sd-controls",
        ]
    )
    assert centered_sd_args.with_support_centered_sd_controls


def valid_ob1_manifest_inputs():
    """Build one exact manifest, fixation table, and canonical passage grid."""
    passages = pd.DataFrame(
        {"passage_id_zero_based": [0, 1]}
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0, 1, 1],
            "seed": [41, 41, 42, 42],
            "text_id": [0, 1, 0, 1],
        }
    )
    manifest = {
        "condition": "baseline_no_predictability",
        "fixation_rows": 4,
        "n_trials": 2,
        "parameters": {"attention_skew": 3},
        "runtimes": [
            {"simulation_id": 0, "seed": 41, "seconds": 1.0},
            {"simulation_id": 1, "seed": 42, "seconds": 1.0},
        ],
        "seeds": [41, 42],
    }
    return manifest, fixations, passages


def test_ob1_manifest_validation_checks_exact_identity_and_trial_count():
    """Provenance must match exact seeds, simulations, and passages."""
    manifest, fixations, passages = valid_ob1_manifest_inputs()
    validation = validate_ob1_manifest_against_fixations(
        manifest,
        fixations,
        passages,
    )
    assert validation["trajectory_attention_skew"] == 3.0
    assert validation["validated_trial_count"] == 2
    assert not validation["legacy_parallel_manifest_missing_n_trials"]
    assert validation["trial_count_validation_source"].startswith(
        "manifest n_trials"
    )

    legacy_parallel = json.loads(json.dumps(manifest))
    legacy_parallel.pop("n_trials")
    legacy_parallel["parallel"] = True
    legacy_validation = validate_ob1_manifest_against_fixations(
        legacy_parallel,
        fixations,
        passages,
    )
    assert legacy_validation["validated_trial_count"] == 2
    assert legacy_validation[
        "legacy_parallel_manifest_missing_n_trials"
    ]
    assert legacy_validation["trial_count_validation_source"].startswith(
        "legacy parallel manifest"
    )

    missing_serial_trials = json.loads(json.dumps(manifest))
    missing_serial_trials.pop("n_trials")
    with pytest.raises(ValueError, match="legacy parallel"):
        validate_ob1_manifest_against_fixations(
            missing_serial_trials,
            fixations,
            passages,
        )

    wrong_seeds = json.loads(json.dumps(manifest))
    wrong_seeds["seeds"] = [41, 99]
    with pytest.raises(ValueError, match="seed set"):
        validate_ob1_manifest_against_fixations(
            wrong_seeds,
            fixations,
            passages,
        )

    wrong_runtime = json.loads(json.dumps(manifest))
    wrong_runtime["runtimes"][1]["simulation_id"] = 9
    with pytest.raises(ValueError, match="simulation-seed pairs disagree"):
        validate_ob1_manifest_against_fixations(
            wrong_runtime,
            fixations,
            passages,
        )

    wrong_trials = json.loads(json.dumps(manifest))
    wrong_trials["n_trials"] = 1
    with pytest.raises(ValueError, match="canonical passage count"):
        validate_ob1_manifest_against_fixations(
            wrong_trials,
            fixations,
            passages,
        )


def test_attention_profile_requires_manifest_or_explicit_override(
    tmp_path,
    monkeypatch,
):
    """Missing trajectory provenance is rejected unless explicitly allowed."""
    manifest, fixations, passages = valid_ob1_manifest_inputs()
    del manifest
    et1_dir = tmp_path / "et1"
    ob1_dir = tmp_path / "ob1"
    et1_dir.mkdir()
    ob1_dir.mkdir()
    pd.DataFrame({"placeholder": [1]}).to_csv(
        et1_dir / "et1_token_values.csv",
        index=False,
    )
    fixations.to_csv(ob1_dir / "ob1_fixations.csv", index=False)
    args = argparse.Namespace(
        allow_initial_sigmas=False,
        allow_missing_ob1_manifest=False,
        bootstrap_samples=10,
        checkpoint=[],
        checkpoint_id=["selected"],
        corpus="provo",
        candidate_support_policy="fixation_matched",
        et1_dir=et1_dir,
        fixation_weighting="duration",
        ob1_attention_skew=[3.0],
        ob1_dir=ob1_dir,
        output_dir=tmp_path / "result",
        processed_dir=tmp_path / "processed",
        profile_component="focused",
        seed=7,
        sigma_json=None,
        sigma_left=[0.4],
        sigma_prefix=None,
        sigma_right=[3.4],
        sigma_source_accuracy=[],
        sigma_value_type="effective",
    )
    monkeypatch.setattr(
        cognitive_main,
        "ensure_prepared",
        lambda *_: (passages, pd.DataFrame()),
    )

    with pytest.raises(FileNotFoundError, match="worker manifest"):
        cognitive_main.command_compare_attention_profile(args)

    captured = {}

    def fake_compare(*_args, **kwargs):
        captured["trajectory_attention_skew"] = kwargs[
            "trajectory_attention_skew"
        ]
        captured["candidate_support_policy"] = kwargs[
            "candidate_support_policy"
        ]
        return {"audit": {}}

    monkeypatch.setattr(
        cognitive_main,
        "compare_attention_profiles",
        fake_compare,
    )
    monkeypatch.setattr(
        cognitive_main,
        "write_attention_profile_outputs",
        lambda *_: None,
    )
    monkeypatch.setattr(
        cognitive_main,
        "write_sigma_records",
        lambda *_: None,
    )
    args.allow_missing_ob1_manifest = True
    cognitive_main.command_compare_attention_profile(args)
    assert captured["trajectory_attention_skew"] is None
    assert captured["candidate_support_policy"] == "fixation_matched"


def test_trial_count_covers_only_published_provo_passages():
    """OB1 trial prefixes cannot extend beyond the 55 passages."""
    validate_trial_count(1)
    validate_trial_count(55)
    with pytest.raises(ValueError, match="between 1 and 55"):
        validate_trial_count(0)
    with pytest.raises(ValueError, match="between 1 and 55"):
        validate_trial_count(56)


def test_trial_count_can_use_the_selected_corpus_size():
    """Generic OB1 validation uses the canonical table's passage count."""
    validate_trial_count(162, 162)
    with pytest.raises(ValueError, match="between 1 and 162"):
        validate_trial_count(163, 162)


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


def test_ensure_prepared_preserves_literal_na_and_validates_onestop(
    tmp_path,
    monkeypatch,
):
    """Standalone loading preserves tokens and invokes strict table checks."""
    pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["NA"],
        }
    ).to_csv(tmp_path / "onestop_passages.csv", index=False)
    pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "word_raw": ["NA"],
            "human_trt_conditional": [float("nan")],
        }
    ).to_csv(tmp_path / "onestop_words.csv", index=False)
    (tmp_path / "onestop_prepare_audit.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    observed = {}

    def validate(passages, words, audit, strict):
        """Capture the exact standalone-load validator inputs."""
        observed["passages"] = passages
        observed["words"] = words
        observed["audit"] = audit
        observed["strict"] = strict

    monkeypatch.setattr(
        cognitive_main,
        "validate_loaded_onestop_model_tables",
        validate,
    )

    passages, words = ensure_prepared(tmp_path, "onestop")

    assert passages.iloc[0]["passage_text"] == "NA"
    assert words.iloc[0]["word_raw"] == "NA"
    assert pd.isna(words.iloc[0]["human_trt_conditional"])
    assert observed["strict"] is True
    assert observed["audit"] == {}


def test_ensure_prepared_retains_provo_validator(tmp_path, monkeypatch):
    """Standalone Provo loading keeps its published-grid validation."""
    pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["one"],
        }
    ).to_csv(tmp_path / "provo_passages.csv", index=False)
    pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "word_raw": ["one"],
            "human_trt_conditional": [100.0],
        }
    ).to_csv(tmp_path / "provo_words.csv", index=False)
    pd.DataFrame({"reason": []}).to_csv(
        tmp_path / "provo_excluded_positions.csv",
        index=False,
    )
    (tmp_path / "provo_prepare_audit.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    observed = {}

    def validate(passages, words, excluded, audit):
        """Capture the existing Provo validation contract."""
        observed["passages"] = passages
        observed["words"] = words
        observed["excluded"] = excluded
        observed["audit"] = audit

    monkeypatch.setattr(
        cognitive_main,
        "validate_canonical_tables",
        validate,
    )

    passages, words = ensure_prepared(tmp_path, "provo")

    assert len(passages) == 1
    assert len(words) == 1
    assert observed["audit"] == {}
    assert observed["excluded"].empty


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


def test_compact_sigma_json_expands_to_runtime_records(tmp_path):
    """A sweep config needs only IDs, accuracies, and effective sigmas."""
    path = tmp_path / "sigmas.json"
    path.write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "first",
                    "source_accuracy": 0.75,
                    "sigma_left": 0.4,
                    "sigma_right": 3.4,
                },
                {
                    "checkpoint_id": "second",
                    "source_accuracy": 0.74,
                    "sigma_left": 3.5,
                    "sigma_right": 0.7,
                },
            ]
        ),
        encoding="utf-8",
    )

    records = load_sigma_records(path)

    assert len(records) == 2
    assert records[0]["checkpoint"] == "direct-sigma:first"
    assert records[0]["sigma_symmetric"] == pytest.approx(1.0)
    assert records[1]["sigma_symmetric"] == pytest.approx(1.0)
    assert records[0]["sigma_symmetric_fixed"] == pytest.approx(1.0)
    assert records[0]["sigma_symmetric_rms_scale"] == pytest.approx(
        (0.4**2 + 3.4**2) ** 0.5 / 2**0.5
    )
    assert records[0]["symmetric_sigma_source"] == (
        "fixed_independent_control"
    )
    assert records[0]["right_left_ratio"] == pytest.approx(8.5)


def test_legacy_rms_symmetric_sigma_is_not_reused(tmp_path):
    """Loading an old complete record resets the independent SymGaussian."""
    path = tmp_path / "legacy_sigmas.json"
    path.write_text(
        json.dumps(
            [
                {
                    "checkpoint_id": "legacy",
                    "checkpoint": "direct-sigma:legacy",
                    "log_sigma_left": -1.0,
                    "log_sigma_right": 1.0,
                    "min_sigma": 1e-6,
                    "sigma_left": 0.4,
                    "sigma_right": 3.4,
                    "sigma_symmetric": 2.420744,
                }
            ]
        ),
        encoding="utf-8",
    )

    record = load_sigma_records(path)[0]

    assert record["sigma_symmetric"] == pytest.approx(1.0)
    assert record["legacy_sigma_symmetric"] == pytest.approx(2.420744)
    assert record["sigma_symmetric_fixed"] == pytest.approx(1.0)
    assert record["sigma_symmetric_rms_scale"] == pytest.approx(
        (0.4**2 + 3.4**2) ** 0.5 / 2**0.5
    )
    assert record["symmetric_sigma_source"] == "fixed_independent_control"


@pytest.mark.parametrize(
    ("command", "required_arguments"),
    [
        (
            "predict-et1",
            [
                "--output-dir",
                "outputs/et1",
                "--sigma-left",
                "0.41553",
                "--sigma-right",
                "3.46115",
                "--checkpoint-id",
                "oasst1_et1_trt",
            ],
        ),
        (
            "simulate-ob1",
            ["--output-dir", "outputs/ob1"],
        ),
        (
            "evaluate",
            [
                "--et1-dir",
                "outputs/et1",
                "--ob1-dir",
                "outputs/ob1",
                "--output-dir",
                "outputs/evaluation",
            ],
        ),
        (
            "compare-attention-profile",
            [
                "--sigma-left",
                "0.41553",
                "--sigma-right",
                "3.46115",
                "--checkpoint-id",
                "oasst1_et1_trt",
                "--et1-dir",
                "outputs/et1",
                "--ob1-dir",
                "outputs/ob1",
                "--output-dir",
                "outputs/attention",
            ],
        ),
        (
            "run",
            [
                "--sigma-left",
                "0.41553",
                "--sigma-right",
                "3.46115",
                "--checkpoint-id",
                "oasst1_et1_trt",
            ],
        ),
    ],
)
def test_corpus_commands_accept_onestop(command, required_arguments):
    """Every model or evaluation command can select the OneStop corpus."""
    args = build_parser().parse_args(
        [command, "--corpus", "onestop", *required_arguments]
    )

    assert args.corpus == "onestop"


@pytest.mark.parametrize(
    ("command", "required_arguments"),
    [
        (
            "predict-et1",
            [
                "--output-dir",
                "outputs/et1",
                "--sigma-left",
                "0.41553",
                "--sigma-right",
                "3.46115",
                "--checkpoint-id",
                "oasst1_et1_trt",
            ],
        ),
        (
            "simulate-ob1",
            ["--output-dir", "outputs/ob1"],
        ),
        (
            "evaluate",
            [
                "--et1-dir",
                "outputs/et1",
                "--ob1-dir",
                "outputs/ob1",
                "--output-dir",
                "outputs/evaluation",
            ],
        ),
        (
            "run",
            [
                "--sigma-left",
                "0.41553",
                "--sigma-right",
                "3.46115",
                "--checkpoint-id",
                "oasst1_et1_trt",
            ],
        ),
    ],
)
def test_corpus_commands_default_to_provo(command, required_arguments):
    """Existing commands retain Provo as their backward-compatible default."""
    args = build_parser().parse_args([command, *required_arguments])

    assert args.corpus == "provo"


def test_run_parser_accepts_special_token_sensitivity():
    """A full run can request the special-token sensitivity."""
    args = build_parser().parse_args(
        [
            "run",
            "--sigma-left",
            "0.41553",
            "--sigma-right",
            "3.46115",
            "--checkpoint-id",
            "oasst1_et1_trt",
            "--with-special-token-sensitivity",
        ]
    )

    assert args.with_special_token_sensitivity is True


@pytest.mark.parametrize("command", ["evaluate", "run"])
def test_evaluation_commands_accept_ob1_clean_passage_sensitivity(command):
    """Standalone and full evaluations expose the clean-passage analysis."""
    if command == "evaluate":
        required = [
            "--et1-dir",
            "outputs/et1",
            "--ob1-dir",
            "outputs/ob1",
            "--output-dir",
            "outputs/evaluation",
        ]
    else:
        required = [
            "--sigma-left",
            "0.41553",
            "--sigma-right",
            "3.46115",
            "--checkpoint-id",
            "oasst1_et1_trt",
        ]
    args = build_parser().parse_args(
        [
            command,
            *required,
            "--with-ob1-clean-passage-sensitivity",
        ]
    )

    assert args.with_ob1_clean_passage_sensitivity is True


def test_predict_parser_accepts_special_token_exclusion():
    """A component ET1 run can exclude special tokens from redistribution."""
    args = build_parser().parse_args(
        [
            "predict-et1",
            "--output-dir",
            "outputs/et1",
            "--sigma-left",
            "0.41553",
            "--sigma-right",
            "3.46115",
            "--checkpoint-id",
            "oasst1_et1_trt",
            "--exclude-special-tokens-from-redistribution",
        ]
    )

    assert args.exclude_special_tokens_from_redistribution is True


def test_prepare_onestop_accepts_explicit_paths():
    """OneStop preparation exposes source and destination path overrides."""
    args = build_parser().parse_args(
        [
            "prepare-onestop",
            "--input-zip",
            "data/raw/onestop/ordinary.zip",
            "--output-dir",
            "data/processed/onestop",
        ]
    )

    assert args.input_zip == Path("data/raw/onestop/ordinary.zip")
    assert args.output_dir == Path("data/processed/onestop")


def test_provo_setup_installs_et2_reference_required_by_audit(
    tmp_path,
    monkeypatch,
):
    """Fresh Provo setup includes the reference tree verified by audit."""
    args = build_parser().parse_args(
        [
            "setup",
            "--corpus",
            "provo",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--skip-et1",
        ]
    )
    comparison_assets = []
    downloads = []
    verifications = []
    prepared = []

    monkeypatch.setattr(
        cognitive_main,
        "download_comparison_assets",
        comparison_assets.append,
    )
    monkeypatch.setattr(
        cognitive_main,
        "download_assets",
        downloads.append,
    )
    monkeypatch.setattr(
        cognitive_main,
        "verify_assets",
        verifications.append,
    )
    monkeypatch.setattr(
        cognitive_main,
        "prepare_corpus",
        lambda corpus, processed_dir, onestop_chunksize: prepared.append(
            (corpus, processed_dir, onestop_chunksize)
        ),
    )

    cognitive_main.command_setup(args)

    assert comparison_assets == ["provo"]
    assert downloads == ["et2-reference"]
    assert verifications == ["et2-reference"]
    assert prepared[0][:2] == ("provo", tmp_path / "processed")


def test_onestop_full_run_reuses_ob1_for_special_token_sensitivity(
    tmp_path,
    monkeypatch,
):
    """OneStop orchestration runs OB1 once and evaluates both ET1 policies."""
    args = build_parser().parse_args(
        [
            "run",
            "--corpus",
            "onestop",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--runtime-dir",
            str(tmp_path / "runtime"),
            "--output-dir",
            str(tmp_path / "output"),
            "--sigma-left",
            "0.41553",
            "--sigma-right",
            "3.46115",
            "--checkpoint-id",
            "oasst1_et1_trt",
            "--with-special-token-sensitivity",
            "--with-ob1-clean-passage-sensitivity",
        ]
    )
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 1],
            "passage_text": ["one two", "three four"],
        }
    )
    words = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0, 1, 1],
            "word_id_zero_based": [0, 1, 0, 1],
        }
    )
    predict_calls = []
    ob1_calls = []
    evaluate_calls = []

    monkeypatch.setattr(
        cognitive_main,
        "download_comparison_assets",
        lambda corpus: None,
    )
    monkeypatch.setattr(
        cognitive_main,
        "prepare_corpus",
        lambda corpus, processed_dir: {},
    )
    monkeypatch.setattr(
        cognitive_main,
        "ensure_prepared",
        lambda processed_dir, corpus: (passages, words),
    )
    monkeypatch.setattr(
        cognitive_main,
        "run_predict_et1",
        lambda *positional, **keyword: predict_calls.append(
            (positional, keyword)
        ),
    )
    monkeypatch.setattr(
        cognitive_main,
        "run_simulate_ob1",
        lambda *positional, **keyword: ob1_calls.append(
            (positional, keyword)
        ),
    )
    monkeypatch.setattr(
        cognitive_main,
        "run_evaluate",
        lambda *positional, **keyword: evaluate_calls.append(
            (positional, keyword)
        ),
    )

    command_run(args)

    assert [
        call[1]["include_special_tokens_in_redistribution"]
        for call in predict_calls
    ] == [True, False]
    assert len(ob1_calls) == 1
    assert ob1_calls[0][0][4] == 2
    assert ob1_calls[0][1]["corpus"] == "onestop"
    assert len(evaluate_calls) == 4
    assert {
        call[0][4] for call in evaluate_calls
    } == {"human_trt_unconditional", "human_trt_conditional"}
    assert all(
        call[1]["corpus"] == "onestop" for call in evaluate_calls
    )
    assert all(
        call[1]["with_ob1_clean_passage_sensitivity"] is True
        for call in evaluate_calls
    )
    manifest = json.loads(
        (tmp_path / "output/run_manifest.json").read_text()
    )
    assert manifest["status"] == "complete"
    assert "completed_at_utc" in manifest


def test_run_evaluate_writes_nested_ob1_clean_sensitivity(
    tmp_path,
    monkeypatch,
):
    """Clean passages produce a nested bundle without replacing primary."""
    processed_dir = tmp_path / "processed"
    et1_dir = tmp_path / "et1"
    ob1_dir = tmp_path / "ob1"
    output_dir = tmp_path / "evaluation"
    et1_dir.mkdir()
    ob1_dir.mkdir()
    pd.DataFrame({"placeholder": [1]}).to_csv(
        et1_dir / "et1_word_values.csv",
        index=False,
    )
    pd.DataFrame({"placeholder": [1]}).to_csv(
        ob1_dir / "ob1_word_values.csv",
        index=False,
    )
    (et1_dir / "et1_inference_audit.json").write_text(
        json.dumps({"corpus": "onestop"}),
        encoding="utf-8",
    )
    (ob1_dir / "ob1_aggregation_audit.json").write_text(
        json.dumps({"corpus": "onestop"}),
        encoding="utf-8",
    )
    word_values = pd.DataFrame(
        {
            "checkpoint_id": ["seed"] * 3,
            "passage_id_zero_based": [0, 1, 2],
            "et1_raw_word_trt": [1.0, 1.0, 1.0],
            "et1_symmetric_word_trt": [1.0, 1.0, 1.0],
            "et1_asymmetric_word_trt": [1.0, 1.0, 1.0],
            "ob1_tvt": [1.0, 1.0, 1.0],
        }
    )
    passage_metrics = pd.DataFrame(
        {
            "checkpoint_id": ["seed"] * 3,
            "passage_id_zero_based": [0, 1, 2],
            "method": ["et1_raw"] * 3,
            "cluster_id": ["article-a", "article-b", "article-c"],
            "original_word_count": [3, 3, 3],
            "ob1_compatible_word_count": [3, 2, 3],
            "ob1_incompatible_words_excluded": [0, 1, 0],
            "word_count": [3, 2, 3],
            "human_missing_words_excluded": [0, 0, 0],
        }
    )
    writes = []
    monkeypatch.setattr(
        cognitive_main,
        "ensure_prepared",
        lambda processed, corpus: (pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        cognitive_main,
        "merge_word_values",
        lambda canonical, et1, ob1: word_values.copy(),
    )
    monkeypatch.setattr(
        cognitive_main,
        "evaluate_passages",
        lambda values, target: passage_metrics.copy(),
    )
    monkeypatch.setattr(
        cognitive_main,
        "summarize_methods",
        lambda *args, **kwargs: pd.DataFrame({"summary": [1]}),
    )
    monkeypatch.setattr(
        cognitive_main,
        "summarize_methods_by_checkpoint",
        lambda *args, **kwargs: pd.DataFrame({"summary": [1]}),
    )
    monkeypatch.setattr(
        cognitive_main,
        "paired_contrasts",
        lambda *args, **kwargs: pd.DataFrame({"summary": [1]}),
    )
    monkeypatch.setattr(
        cognitive_main,
        "paired_contrasts_by_checkpoint",
        lambda *args, **kwargs: pd.DataFrame({"summary": [1]}),
    )
    monkeypatch.setattr(
        cognitive_main,
        "write_evaluation_outputs",
        lambda *args: writes.append(args),
    )

    primary_audit = cognitive_main.run_evaluate(
        processed_dir,
        et1_dir,
        ob1_dir,
        output_dir,
        "human_trt_conditional",
        bootstrap_samples=100,
        seed=7,
        corpus="onestop",
        with_ob1_clean_passage_sensitivity=True,
    )

    assert len(writes) == 2
    assert writes[0][0] == output_dir
    assert set(writes[0][2]["passage_id_zero_based"]) == {0, 1, 2}
    assert writes[1][0] == output_dir / "ob1_clean_passages"
    assert set(writes[1][2]["passage_id_zero_based"]) == {0, 2}
    assert primary_audit["passages"] == 3
    assert "sensitivity_policy" not in primary_audit
    clean_audit = writes[1][6]
    assert clean_audit["passages"] == 2
    assert clean_audit["source_passages"] == 3
    assert clean_audit["excluded_passages"] == 1
    assert clean_audit["excluded_passage_ids"] == [1]
    assert clean_audit["primary_results_unchanged"] is True
    assert clean_audit["resampling_clusters"] == 2

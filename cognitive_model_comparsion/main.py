"""One Python entry point for the reviewer-requested cognitive comparison."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "data/runtime_cache/matplotlib"),
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cognitive_model_comparsion.download_assets import (
    download_assets,
    verify_assets,
)
from cognitive_model_comparsion.src.attention_profile import (
    compare_attention_profiles,
    write_attention_profile_outputs,
)
from cognitive_model_comparsion.src.audit_et2_provo import compare_et2_provo
from cognitive_model_comparsion.src.audit_provo import audit_dataset
from cognitive_model_comparsion.src.et1_inference import (
    ET1NativePredictor,
    run_et1_inference,
    write_et1_outputs,
)
from cognitive_model_comparsion.src.evaluate import (
    evaluate_passages,
    merge_word_values,
    paired_contrasts,
    select_ob1_clean_passages,
    summarize_methods,
    summarize_methods_by_checkpoint,
    write_evaluation_outputs,
)
from cognitive_model_comparsion.src.ob1_runner import (
    DEFAULT_ONESTOP_RUNTIME_DIR,
    DEFAULT_RUNTIME_DIR,
    aggregate_ob1_tvt,
    prepare_ob1_runtime,
    run_ob1_subprocess,
    stimulus_word_coordinates,
    write_ob1_aggregation,
)
from cognitive_model_comparsion.src.prepare_onestop import (
    DEFAULT_CHUNK_SIZE as ONESTOP_DEFAULT_CHUNK_SIZE,
    ONESTOP_ZIP_FILENAME,
    RAW_DIR as ONESTOP_RAW_DIR,
    build_onestop_tables,
    validate_loaded_onestop_model_tables,
    write_onestop_tables,
)
from cognitive_model_comparsion.src.prepare_provo import (
    EYE_FILENAME,
    PREDICTABILITY_FILENAME,
    RAW_DIR as PROVO_RAW_DIR,
    build_canonical_tables,
    validate_canonical_tables,
    write_canonical_tables,
)
from cognitive_model_comparsion.src.sigmas import (
    direct_sigma_record,
    extract_sigma_record,
    sha256_file,
)


DEFAULT_PROCESSED_DIR = ROOT / "data/processed"
DEFAULT_ONESTOP_PROCESSED_DIR = ROOT / "data/processed/onestop"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/full_run"
DEFAULT_ONESTOP_OUTPUT_DIR = ROOT / "outputs/onestop_full_run"
CORPORA = ("provo", "onestop")
CORPUS_FILENAMES = {
    "provo": ("provo_passages.csv", "provo_words.csv"),
    "onestop": ("onestop_passages.csv", "onestop_words.csv"),
}
CORPUS_STIMULUS_NAMES = {
    "provo": "Provo_Corpus",
    "onestop": "OneStop_Ordinary_Advanced",
}
OB1_CLEAN_SENSITIVITY_DIRECTORY = "ob1_clean_passages"


def parse_seed_specification(value: str) -> list[int]:
    """Parse comma-separated seeds or a half-open start:stop range."""
    value = value.strip()
    if ":" in value:
        pieces = value.split(":")
        if len(pieces) != 2:
            raise ValueError(f"Invalid seed range: {value}")
        start, stop = map(int, pieces)
        seeds = list(range(start, stop))
    else:
        seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be a nonempty unique sequence")
    return seeds


def validate_trial_count(n_trials: int, passage_count: int = 55) -> None:
    """Require a nonempty prefix within an explicitly known passage table."""
    if passage_count < 1:
        raise ValueError("passage_count must be positive")
    if not 1 <= n_trials <= passage_count:
        raise ValueError(
            f"--n-trials must be between 1 and {passage_count}"
        )


def default_processed_dir(corpus: str) -> Path:
    """Return the checked corpus-specific canonical-table directory."""
    if corpus == "provo":
        return DEFAULT_PROCESSED_DIR
    if corpus == "onestop":
        return DEFAULT_ONESTOP_PROCESSED_DIR
    raise ValueError(f"Unsupported corpus: {corpus}")


def default_runtime_dir(corpus: str) -> Path:
    """Return the isolated corpus-specific OB1 runtime directory."""
    if corpus == "provo":
        return DEFAULT_RUNTIME_DIR
    if corpus == "onestop":
        return DEFAULT_ONESTOP_RUNTIME_DIR
    raise ValueError(f"Unsupported corpus: {corpus}")


def default_output_dir(corpus: str) -> Path:
    """Return the corpus-specific full-run output directory."""
    if corpus == "provo":
        return DEFAULT_OUTPUT_DIR
    if corpus == "onestop":
        return DEFAULT_ONESTOP_OUTPUT_DIR
    raise ValueError(f"Unsupported corpus: {corpus}")


def resolved_path(value: Path | None, fallback: Path) -> Path:
    """Use a user override or one deterministic corpus-specific default."""
    return Path(value) if value is not None else fallback


def json_safe(value):
    """Convert argparse values into JSON-serializable provenance values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def prepare_corpus(
    corpus: str,
    processed_dir: Path,
    onestop_chunksize: int = ONESTOP_DEFAULT_CHUNK_SIZE,
) -> dict:
    """Build and write one corpus's canonical Human-TRT tables."""
    if corpus == "provo":
        artifacts = build_canonical_tables(
            PROVO_RAW_DIR / EYE_FILENAME,
            PROVO_RAW_DIR / PREDICTABILITY_FILENAME,
        )
        write_canonical_tables(processed_dir, *artifacts)
        return artifacts[-1]
    if corpus == "onestop":
        artifacts = build_onestop_tables(
            ONESTOP_RAW_DIR / ONESTOP_ZIP_FILENAME,
            chunksize=onestop_chunksize,
            strict=True,
        )
        write_onestop_tables(processed_dir, *artifacts)
        return artifacts[-1]
    raise ValueError(f"Unsupported corpus: {corpus}")


def ensure_prepared(
    processed_dir: Path,
    corpus: str = "provo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical corpus tables, building checked tables when absent."""
    if corpus not in CORPUS_FILENAMES:
        raise ValueError(f"Unsupported corpus: {corpus}")
    passage_filename, word_filename = CORPUS_FILENAMES[corpus]
    passages_path = processed_dir / passage_filename
    words_path = processed_dir / word_filename
    required_paths = [passages_path, words_path]
    if corpus == "provo":
        required_paths.extend(
            [
                processed_dir / "provo_excluded_positions.csv",
                processed_dir / "provo_prepare_audit.json",
            ]
        )
    else:
        required_paths.append(
            processed_dir / "onestop_prepare_audit.json"
        )
    if not all(path.is_file() for path in required_paths):
        prepare_corpus(corpus, processed_dir)
    passages = pd.read_csv(passages_path, keep_default_na=False)
    words = pd.read_csv(
        words_path,
        keep_default_na=False,
        na_values={"human_trt_conditional": [""]},
    )
    if passages.empty or words.empty:
        raise ValueError(f"Prepared {corpus} tables must not be empty")
    if corpus == "provo":
        excluded = pd.read_csv(
            processed_dir / "provo_excluded_positions.csv",
            keep_default_na=False,
        )
        with (
            processed_dir / "provo_prepare_audit.json"
        ).open(encoding="utf-8") as handle:
            audit = json.load(handle)
        validate_canonical_tables(
            passages,
            words,
            excluded,
            audit,
        )
    else:
        with (
            processed_dir / "onestop_prepare_audit.json"
        ).open(encoding="utf-8") as handle:
            audit = json.load(handle)
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=True,
        )
    return passages, words


def download_comparison_assets(corpus: str) -> None:
    """Download and verify the minimal assets for one full comparison."""
    if corpus not in CORPORA:
        raise ValueError(f"Unsupported corpus: {corpus}")
    for selected in (corpus, "ob1", "subtlex"):
        download_assets(selected)
        verify_assets(selected)


def checkpoint_ids(
    checkpoints: list[Path],
    requested_ids: list[str] | None,
) -> list[str]:
    """Resolve unique stable checkpoint labels."""
    identifiers = (
        requested_ids
        if requested_ids
        else [path.expanduser().resolve().name for path in checkpoints]
    )
    if len(identifiers) != len(checkpoints):
        raise ValueError("--checkpoint-id count must match --checkpoint count")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Checkpoint IDs must be unique")
    return identifiers


def validate_predict_sigma_arguments(args: argparse.Namespace) -> None:
    """Reject incomplete, ambiguous, or silently ignored sigma sources."""
    sigma_json = getattr(args, "sigma_json", None)
    checkpoints = getattr(args, "checkpoint", None) or []
    sigma_left = getattr(args, "sigma_left", None) or []
    sigma_right = getattr(args, "sigma_right", None) or []
    checkpoint_ids_requested = getattr(args, "checkpoint_id", None) or []
    source_accuracies = getattr(args, "sigma_source_accuracy", None) or []
    has_direct = bool(sigma_left or sigma_right)
    source_count = int(bool(sigma_json)) + int(bool(checkpoints)) + int(has_direct)
    if source_count > 1:
        raise ValueError(
            "Pass exactly one sigma source: --sigma-json, --checkpoint, "
            "or direct --sigma-left/--sigma-right values"
        )
    if bool(sigma_left) != bool(sigma_right) or len(sigma_left) != len(sigma_right):
        raise ValueError(
            "Direct sigma input requires matching --sigma-left and --sigma-right"
        )
    if checkpoint_ids_requested and not (checkpoints or has_direct):
        raise ValueError(
            "--checkpoint-id requires --checkpoint or direct sigma values"
        )
    expected_count = len(checkpoints) if checkpoints else len(sigma_left)
    if checkpoint_ids_requested and len(checkpoint_ids_requested) != expected_count:
        raise ValueError("--checkpoint-id count must match the sigma source count")
    if has_direct and not checkpoint_ids_requested:
        raise ValueError("Direct sigma input requires one --checkpoint-id per pair")
    if source_accuracies and not has_direct:
        raise ValueError("--sigma-source-accuracy requires direct sigma values")
    if source_accuracies and len(source_accuracies) != len(sigma_left):
        raise ValueError(
            "--sigma-source-accuracy count must match the direct sigma pair count"
        )
    if args.sigma_prefix and not checkpoints:
        raise ValueError("--sigma-prefix requires --checkpoint")


def direct_sigma_records_from_args(args: argparse.Namespace) -> list[dict]:
    """Convert repeated direct CLI sigma pairs into fixed records."""
    sigma_left = getattr(args, "sigma_left", None) or []
    sigma_right = getattr(args, "sigma_right", None) or []
    checkpoint_ids_requested = getattr(args, "checkpoint_id", None) or []
    source_accuracies = getattr(args, "sigma_source_accuracy", None) or []
    accuracies = (
        source_accuracies
        if source_accuracies
        else [None for _ in sigma_left]
    )
    return [
        direct_sigma_record(
            checkpoint_id=identifier,
            sigma_left=left,
            sigma_right=right,
            value_type=args.sigma_value_type,
            allow_initial_sigmas=args.allow_initial_sigmas,
            source_accuracy=accuracy,
        )
        for identifier, left, right, accuracy in zip(
            checkpoint_ids_requested,
            sigma_left,
            sigma_right,
            accuracies,
        )
    ]


def extract_sigma_records(
    checkpoints: list[Path],
    requested_ids: list[str] | None,
    sigma_prefix: str | None,
    allow_initial_sigmas: bool,
) -> list[dict]:
    """Extract every requested checkpoint's fixed sigma record."""
    identifiers = checkpoint_ids(checkpoints, requested_ids)
    records = []
    for identifier, checkpoint in zip(identifiers, checkpoints):
        record = extract_sigma_record(
            checkpoint,
            requested_prefix=sigma_prefix,
            allow_initial_sigmas=allow_initial_sigmas,
        )
        record["checkpoint_id"] = identifier
        records.append(record)
    return records


def write_sigma_records(output_dir: Path, records: list[dict]) -> None:
    """Write complete JSON and flat CSV sigma provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "checkpoint_sigmas.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    flat_records = []
    for record in records:
        flat = record.copy()
        flat["available_sigma_prefixes"] = json.dumps(
            flat["available_sigma_prefixes"]
        )
        flat_records.append(flat)
    pd.DataFrame(flat_records).to_csv(
        output_dir / "checkpoint_sigmas.csv",
        index=False,
    )


def load_sigma_records(path: Path) -> list[dict]:
    """Load sigma records written by this entry point."""
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Sigma JSON must contain a nonempty list: {path}")
    return records


def run_predict_et1(
    processed_dir: Path,
    output_dir: Path,
    sigma_records: list[dict],
    et1_checkpoint: Path | None,
    et1_tokenizer: Path | None,
    corpus: str = "provo",
    include_special_tokens_in_redistribution: bool = True,
) -> dict:
    """Run one frozen ET1 pass and all fixed redistribution checkpoints."""
    passages, words = ensure_prepared(processed_dir, corpus)
    predictor = ET1NativePredictor(
        checkpoint_path=et1_checkpoint,
        tokenizer_path=et1_tokenizer,
    )
    artifacts = run_et1_inference(
        passages,
        words,
        sigma_records,
        predictor,
        include_special_tokens_in_redistribution=(
            include_special_tokens_in_redistribution
        ),
    )
    audit = dict(artifacts[-1])
    audit["corpus"] = corpus
    write_et1_outputs(output_dir, *artifacts[:-1], audit)
    return audit


def run_simulate_ob1(
    processed_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int | None,
    python_hash_seed: int,
    workers: int,
    corpus: str = "provo",
) -> dict:
    """Prepare runtime, execute OB1, and aggregate word-level TVT."""
    passages, words = ensure_prepared(processed_dir, corpus)
    resolved_n_trials = len(passages) if n_trials is None else n_trials
    validate_trial_count(resolved_n_trials, len(passages))
    stimulus_name = CORPUS_STIMULUS_NAMES[corpus]
    preparation = prepare_ob1_runtime(
        passages,
        runtime_dir,
        python_hash_seed=python_hash_seed,
        stimulus_name=stimulus_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transformations = pd.read_csv(
        preparation["token_transformations_path"]
    )
    if len(transformations) != int(preparation["token_transformations"]):
        raise ValueError("OB1 token-transformation audit count changed")
    if "ob1_evaluable" in words:
        eligibility = words["ob1_evaluable"]
        if eligibility.dtype != bool:
            normalized = eligibility.astype(str).str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise ValueError("Invalid canonical ob1_evaluable values")
            eligibility = normalized.eq("true")
        coordinate_columns = [
            "passage_id_zero_based",
            "word_id_zero_based",
        ]
        expected_coordinates = {
            tuple(map(int, row))
            for row in words.loc[
                ~eligibility,
                coordinate_columns,
            ].itertuples(index=False, name=None)
        }
        transformed_coordinates = {
            tuple(map(int, row))
            for row in transformations[
                coordinate_columns
            ].itertuples(index=False, name=None)
        }
        if transformed_coordinates != expected_coordinates:
            raise ValueError(
                "OB1 runtime transformations do not match canonical "
                "ob1_evaluable exclusions"
            )
    output_transformations = output_dir / "ob1_token_transformations.csv"
    transformations.to_csv(output_transformations, index=False)
    preparation["output_token_transformations_path"] = str(
        output_transformations.resolve()
    )
    with (output_dir / "ob1_runtime_preparation.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(preparation, handle, indent=2, sort_keys=True)
        handle.write("\n")
    run_ob1_subprocess(
        runtime_dir,
        output_dir,
        seeds=seeds,
        n_trials=resolved_n_trials,
        python_hash_seed=python_hash_seed,
        workers=workers,
        stimulus_name=stimulus_name,
    )
    fixations = pd.read_csv(output_dir / "ob1_fixations.csv")
    eligible_words = words[
        words["passage_id_zero_based"] < resolved_n_trials
    ].copy()
    simulated_passages = passages[
        passages["passage_id_zero_based"] < resolved_n_trials
    ].copy()
    artifacts = aggregate_ob1_tvt(
        fixations,
        eligible_words,
        valid_fixation_coordinates=stimulus_word_coordinates(
            simulated_passages
        ),
        expected_seeds=seeds,
    )
    audit = dict(artifacts[-1])
    audit.update(
        {
            "corpus": corpus,
            "stimulus_name": stimulus_name,
            "n_trials": int(resolved_n_trials),
        }
    )
    write_ob1_aggregation(output_dir, *artifacts[:-1], audit)
    return audit


def evaluation_volume_audit(
    word_values: pd.DataFrame,
    passage_metrics: pd.DataFrame,
) -> dict:
    """Summarize the exact passages and words entering one evaluation."""
    audit_columns = [
        "passage_id_zero_based",
        "original_word_count",
        "ob1_compatible_word_count",
        "ob1_incompatible_words_excluded",
        "word_count",
        "human_missing_words_excluded",
    ]
    passage_word_audit = (
        passage_metrics[audit_columns]
        .drop_duplicates()
        .sort_values("passage_id_zero_based")
    )
    if passage_word_audit["passage_id_zero_based"].duplicated().any():
        raise ValueError("Evaluation word filtering differs across methods")
    return {
        "checkpoints": int(word_values["checkpoint_id"].nunique()),
        "passages": int(
            passage_word_audit["passage_id_zero_based"].nunique()
        ),
        "word_rows": int(len(word_values)),
        "passage_metric_rows": int(len(passage_metrics)),
        "original_evaluation_words": int(
            passage_word_audit["original_word_count"].sum()
        ),
        "ob1_compatible_words": int(
            passage_word_audit["ob1_compatible_word_count"].sum()
        ),
        "ob1_incompatible_words_excluded": int(
            passage_word_audit[
                "ob1_incompatible_words_excluded"
            ].sum()
        ),
        "evaluated_words": int(
            passage_word_audit["word_count"].sum()
        ),
        "human_missing_words_excluded": int(
            passage_word_audit["human_missing_words_excluded"].sum()
        ),
        "negative_value_counts": {
            column: int((word_values[column] < 0).sum())
            for column in (
                "et1_raw_word_trt",
                "et1_symmetric_word_trt",
                "et1_asymmetric_word_trt",
                "ob1_tvt",
            )
        },
    }


def run_evaluate(
    processed_dir: Path,
    et1_dir: Path,
    ob1_dir: Path,
    output_dir: Path,
    human_target: str,
    bootstrap_samples: int,
    seed: int,
    corpus: str = "provo",
    cluster_column: str | None = None,
    with_ob1_clean_passage_sensitivity: bool = False,
) -> dict:
    """Join saved model values and produce all statistical outputs."""
    _, canonical_words = ensure_prepared(processed_dir, corpus)
    if cluster_column is None and corpus == "onestop":
        cluster_column = "cluster_id"
    et1_words = pd.read_csv(et1_dir / "et1_word_values.csv")
    ob1_words = pd.read_csv(ob1_dir / "ob1_word_values.csv")
    et1_audit_path = et1_dir / "et1_inference_audit.json"
    ob1_audit_path = ob1_dir / "ob1_aggregation_audit.json"
    with et1_audit_path.open(encoding="utf-8") as handle:
        et1_audit = json.load(handle)
    with ob1_audit_path.open(encoding="utf-8") as handle:
        ob1_audit = json.load(handle)
    for source_name, source_audit in (
        ("ET1", et1_audit),
        ("OB1", ob1_audit),
    ):
        source_corpus = source_audit.get("corpus")
        if source_corpus is not None and source_corpus != corpus:
            raise ValueError(
                f"{source_name} source corpus {source_corpus!r} "
                f"does not match requested corpus {corpus!r}"
            )
    word_values = merge_word_values(canonical_words, et1_words, ob1_words)
    passage_metrics = evaluate_passages(word_values, human_target)
    method_summary = summarize_methods(
        passage_metrics,
        bootstrap_samples,
        seed,
        cluster_column=cluster_column,
    )
    checkpoint_summary = summarize_methods_by_checkpoint(
        passage_metrics,
        bootstrap_samples,
        seed,
        cluster_column=cluster_column,
    )
    contrasts = paired_contrasts(
        passage_metrics,
        bootstrap_samples,
        seed,
        cluster_column=cluster_column,
    )
    audit = {
        "corpus": corpus,
        "et1_source_dir": str(et1_dir.resolve()),
        "et1_redistribution_special_token_policy": et1_audit.get(
            "redistribution_special_token_policy"
        ),
        "et1_special_tokens_included_in_redistribution": et1_audit.get(
            "special_tokens_included_in_redistribution"
        ),
        "et1_predictor_cache_signature": et1_audit.get(
            "predictor_cache_signature"
        ),
        "ob1_source_dir": str(ob1_dir.resolve()),
        "ob1_virtual_readers": ob1_audit.get("virtual_readers"),
        "ob1_seeds": ob1_audit.get("seeds"),
        "ob1_stimulus_name": ob1_audit.get("stimulus_name"),
        "human_target": human_target,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "resampling_cluster_column": cluster_column,
        "resampling_clusters": (
            int(passage_metrics[cluster_column].nunique())
            if cluster_column is not None
            else None
        ),
        **evaluation_volume_audit(word_values, passage_metrics),
    }
    write_evaluation_outputs(
        output_dir,
        word_values,
        passage_metrics,
        method_summary,
        checkpoint_summary,
        contrasts,
        audit,
    )
    if with_ob1_clean_passage_sensitivity:
        clean_metrics, selection_audit = select_ob1_clean_passages(
            passage_metrics
        )
        clean_passage_ids = set(
            clean_metrics["passage_id_zero_based"].unique().tolist()
        )
        clean_word_values = word_values.loc[
            word_values["passage_id_zero_based"].isin(clean_passage_ids)
        ].copy()
        clean_method_summary = summarize_methods(
            clean_metrics,
            bootstrap_samples,
            seed,
            cluster_column=cluster_column,
        )
        clean_checkpoint_summary = summarize_methods_by_checkpoint(
            clean_metrics,
            bootstrap_samples,
            seed,
            cluster_column=cluster_column,
        )
        clean_contrasts = paired_contrasts(
            clean_metrics,
            bootstrap_samples,
            seed,
            cluster_column=cluster_column,
        )
        clean_output_dir = (
            output_dir / OB1_CLEAN_SENSITIVITY_DIRECTORY
        )
        clean_audit = {
            **audit,
            **evaluation_volume_audit(
                clean_word_values,
                clean_metrics,
            ),
            **selection_audit,
            "analysis_role": "sensitivity_only",
            "primary_results_unchanged": True,
            "primary_evaluation_dir": str(output_dir.resolve()),
            "sensitivity_output_dir": str(clean_output_dir.resolve()),
            "resampling_clusters": (
                int(clean_metrics[cluster_column].nunique())
                if cluster_column is not None
                else None
            ),
        }
        write_evaluation_outputs(
            clean_output_dir,
            clean_word_values,
            clean_metrics,
            clean_method_summary,
            clean_checkpoint_summary,
            clean_contrasts,
            clean_audit,
        )
    return audit


def git_metadata() -> dict:
    """Return best-effort Git commit and dirty-state metadata."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def runtime_manifest(args: argparse.Namespace, output_dir: Path) -> dict:
    """Record reproducibility metadata for a full run."""
    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "transformers",
        "safetensors",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": {
            key: json_safe(value)
            for key, value in vars(args).items()
            if key != "handler"
        },
        "output_dir": str(output_dir.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": git_metadata(),
    }


def command_setup(args: argparse.Namespace) -> None:
    """Download and validate one corpus comparison and optional ET1 assets."""
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    )
    download_comparison_assets(args.corpus)
    if args.corpus == "provo":
        download_assets("et2-reference")
        verify_assets("et2-reference")
    prepare_corpus(
        args.corpus,
        processed_dir,
        onestop_chunksize=args.onestop_chunksize,
    )
    if not args.skip_et1:
        from models.et_checkpoint import ensure_et1_checkpoint
        from models.et1_tokenizer import load_et1_tokenizer

        ensure_et1_checkpoint(args.et1_checkpoint)
        load_et1_tokenizer(tokenizer_path=args.et1_tokenizer)


def command_audit(args: argparse.Namespace) -> None:
    """Print checked raw-data provenance for the selected corpus."""
    if args.corpus == "provo":
        verify_assets("provo")
        verify_assets("et2-reference")
        report = {
            "provo": audit_dataset(PROVO_RAW_DIR),
            "et2_provo": compare_et2_provo(
                PROVO_RAW_DIR / EYE_FILENAME,
                ROOT / "third_party/et2_torontocl_cmcl_2021/data/provo.csv",
            ),
        }
    else:
        verify_assets("onestop")
        processed_dir = resolved_path(
            args.processed_dir,
            default_processed_dir("onestop"),
        )
        ensure_prepared(processed_dir, "onestop")
        audit_path = processed_dir / "onestop_prepare_audit.json"
        with audit_path.open(encoding="utf-8") as handle:
            report = {"onestop": json.load(handle)}
    print(json.dumps(report, indent=2, sort_keys=True))


def command_prepare(args: argparse.Namespace) -> None:
    """Build and print the canonical Human Provo grid."""
    artifacts = build_canonical_tables(
        PROVO_RAW_DIR / EYE_FILENAME,
        PROVO_RAW_DIR / PREDICTABILITY_FILENAME,
    )
    write_canonical_tables(args.output_dir, *artifacts)
    print(json.dumps(artifacts[-1], indent=2, sort_keys=True))


def command_prepare_onestop(args: argparse.Namespace) -> None:
    """Build and print the canonical Human OneStop Advanced grid."""
    artifacts = build_onestop_tables(
        args.input_zip,
        chunksize=args.chunksize,
        strict=args.strict,
    )
    write_onestop_tables(args.output_dir, *artifacts)
    print(json.dumps(artifacts[-1], indent=2, sort_keys=True))


def command_extract_sigmas(args: argparse.Namespace) -> None:
    """Extract fixed sigmas from all requested RM checkpoints."""
    records = extract_sigma_records(
        args.checkpoint,
        args.checkpoint_id,
        args.sigma_prefix,
        args.allow_initial_sigmas,
    )
    write_sigma_records(args.output_dir, records)
    print(json.dumps(records, indent=2, sort_keys=True))


def command_predict_et1(args: argparse.Namespace) -> None:
    """Run frozen ET1 with raw or checkpoint-specific redistribution."""
    validate_predict_sigma_arguments(args)
    records = (
        load_sigma_records(args.sigma_json)
        if args.sigma_json
        else extract_sigma_records(
            args.checkpoint,
            args.checkpoint_id,
            args.sigma_prefix,
            args.allow_initial_sigmas,
        )
        if args.checkpoint
        else direct_sigma_records_from_args(args)
        if args.sigma_left
        else []
    )
    if records:
        write_sigma_records(args.output_dir, records)
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    )
    audit = run_predict_et1(
        processed_dir,
        args.output_dir,
        records,
        args.et1_checkpoint,
        args.et1_tokenizer,
        corpus=args.corpus,
        include_special_tokens_in_redistribution=(
            not args.exclude_special_tokens_from_redistribution
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_simulate_ob1(args: argparse.Namespace) -> None:
    """Run the exact baseline OB1 virtual readers and aggregate TVT."""
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    )
    runtime_dir = resolved_path(
        args.runtime_dir,
        default_runtime_dir(args.corpus),
    )
    audit = run_simulate_ob1(
        processed_dir,
        runtime_dir,
        args.output_dir,
        parse_seed_specification(args.seeds),
        args.n_trials,
        args.python_hash_seed,
        args.workers,
        corpus=args.corpus,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate saved ET1 and OB1 word allocations."""
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    )
    audit = run_evaluate(
        processed_dir,
        args.et1_dir,
        args.ob1_dir,
        args.output_dir,
        args.human_target,
        args.bootstrap_samples,
        args.seed,
        corpus=args.corpus,
        with_ob1_clean_passage_sensitivity=(
            args.with_ob1_clean_passage_sensitivity
        ),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_compare_attention_profile(args: argparse.Namespace) -> None:
    """Compare four token kernels with projected OB1 internal attention."""
    validate_predict_sigma_arguments(args)
    records = (
        load_sigma_records(args.sigma_json)
        if args.sigma_json
        else extract_sigma_records(
            args.checkpoint,
            args.checkpoint_id,
            args.sigma_prefix,
            args.allow_initial_sigmas,
        )
        if args.checkpoint
        else direct_sigma_records_from_args(args)
    )
    if len(records) != 1:
        raise ValueError(
            "compare-attention-profile requires exactly one sigma record"
        )
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    )
    passages, _ = ensure_prepared(processed_dir, args.corpus)
    token_path = args.et1_dir / "et1_token_values.csv"
    fixation_path = args.ob1_dir / "ob1_fixations.csv"
    if not token_path.is_file():
        raise FileNotFoundError(f"Missing ET1 token table: {token_path}")
    if not fixation_path.is_file():
        raise FileNotFoundError(f"Missing OB1 fixation table: {fixation_path}")
    attention_skews = tuple(
        args.ob1_attention_skew
        if args.ob1_attention_skew
        else (3.0, 4.0)
    )
    artifacts = compare_attention_profiles(
        passages,
        pd.read_csv(token_path),
        pd.read_csv(fixation_path),
        records[0],
        attention_skews=attention_skews,
        fixation_weighting=args.fixation_weighting,
        profile_component=args.profile_component,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    artifacts["audit"].update(
        {
            "et1_token_values_path": str(token_path.resolve()),
            "et1_token_values_sha256": sha256_file(token_path),
            "ob1_fixations_path": str(fixation_path.resolve()),
            "ob1_fixations_sha256": sha256_file(fixation_path),
            "sigma_json_path": (
                str(args.sigma_json.resolve())
                if args.sigma_json is not None
                else None
            ),
            "sigma_json_sha256": (
                sha256_file(args.sigma_json)
                if args.sigma_json is not None
                else None
            ),
        }
    )
    write_attention_profile_outputs(args.output_dir, artifacts)
    write_sigma_records(args.output_dir, records)
    print(json.dumps(artifacts["audit"], indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    """Execute setup, inference, OB1, and both Human TRT evaluations."""
    validate_predict_sigma_arguments(args)
    if not args.checkpoint and not args.sigma_left:
        raise ValueError(
            "run requires --checkpoint or direct "
            "--sigma-left/--sigma-right values"
        )
    processed_dir = resolved_path(
        args.processed_dir,
        default_processed_dir(args.corpus),
    ).resolve()
    runtime_dir = resolved_path(
        args.runtime_dir,
        default_runtime_dir(args.corpus),
    ).resolve()
    output_dir = resolved_path(
        args.output_dir,
        default_output_dir(args.corpus),
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(args, output_dir)
    manifest["resolved_paths"] = {
        "processed_dir": str(processed_dir),
        "runtime_dir": str(runtime_dir),
        "output_dir": str(output_dir),
    }
    manifest["status"] = "running"
    manifest_path = output_dir / "run_manifest.json"
    write_json_atomic(manifest_path, manifest)

    download_comparison_assets(args.corpus)
    prepare_corpus(args.corpus, processed_dir)
    passages, _ = ensure_prepared(
        processed_dir,
        args.corpus,
    )
    records = (
        extract_sigma_records(
            args.checkpoint,
            args.checkpoint_id,
            args.sigma_prefix,
            args.allow_initial_sigmas,
        )
        if args.checkpoint
        else direct_sigma_records_from_args(args)
    )
    write_sigma_records(output_dir, records)
    run_predict_et1(
        processed_dir,
        output_dir / "et1",
        records,
        args.et1_checkpoint,
        args.et1_tokenizer,
        corpus=args.corpus,
        include_special_tokens_in_redistribution=True,
    )
    seeds = parse_seed_specification(args.seeds)
    run_simulate_ob1(
        processed_dir,
        runtime_dir,
        output_dir / "ob1",
        seeds,
        len(passages),
        args.python_hash_seed,
        args.workers,
        corpus=args.corpus,
    )
    run_evaluate(
        processed_dir,
        output_dir / "et1",
        output_dir / "ob1",
        output_dir / "evaluation_unconditional",
        "human_trt_unconditional",
        args.bootstrap_samples,
        args.seed,
        corpus=args.corpus,
        with_ob1_clean_passage_sensitivity=(
            args.with_ob1_clean_passage_sensitivity
        ),
    )
    run_evaluate(
        processed_dir,
        output_dir / "et1",
        output_dir / "ob1",
        output_dir / "evaluation_conditional",
        "human_trt_conditional",
        args.bootstrap_samples,
        args.seed,
        corpus=args.corpus,
        with_ob1_clean_passage_sensitivity=(
            args.with_ob1_clean_passage_sensitivity
        ),
    )
    if args.with_special_token_sensitivity:
        sensitivity_et1_dir = output_dir / "et1_special_excluded"
        run_predict_et1(
            processed_dir,
            sensitivity_et1_dir,
            records,
            args.et1_checkpoint,
            args.et1_tokenizer,
            corpus=args.corpus,
            include_special_tokens_in_redistribution=False,
        )
        run_evaluate(
            processed_dir,
            sensitivity_et1_dir,
            output_dir / "ob1",
            output_dir / "evaluation_unconditional_special_excluded",
            "human_trt_unconditional",
            args.bootstrap_samples,
            args.seed,
            corpus=args.corpus,
            with_ob1_clean_passage_sensitivity=(
                args.with_ob1_clean_passage_sensitivity
            ),
        )
        run_evaluate(
            processed_dir,
            sensitivity_et1_dir,
            output_dir / "ob1",
            output_dir / "evaluation_conditional_special_excluded",
            "human_trt_conditional",
            args.bootstrap_samples,
            args.seed,
            corpus=args.corpus,
            with_ob1_clean_passage_sensitivity=(
                args.with_ob1_clean_passage_sensitivity
            ),
        )
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(manifest_path, manifest)


def add_common_processed_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared canonical-table directory argument."""
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
    )


def add_corpus_argument(parser: argparse.ArgumentParser) -> None:
    """Add a backward-compatible corpus selector."""
    parser.add_argument(
        "--corpus",
        choices=CORPORA,
        default="provo",
    )


def add_sigma_arguments(
    parser: argparse.ArgumentParser,
    required: bool,
) -> None:
    """Add checkpoint and sigma-state selection arguments."""
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=required,
        default=[],
    )
    parser.add_argument("--checkpoint-id", action="append")
    parser.add_argument("--sigma-prefix")
    parser.add_argument("--allow-initial-sigmas", action="store_true")


def add_et1_arguments(parser: argparse.ArgumentParser) -> None:
    """Add frozen ET1 local override arguments."""
    parser.add_argument("--et1-checkpoint", type=Path)
    parser.add_argument("--et1-tokenizer", type=Path)


def add_direct_sigma_arguments(parser: argparse.ArgumentParser) -> None:
    """Add repeated direct fixed-sigma values and provenance metadata."""
    parser.add_argument("--sigma-left", action="append", type=float)
    parser.add_argument("--sigma-right", action="append", type=float)
    parser.add_argument(
        "--sigma-value-type",
        choices=("effective", "log"),
        default="effective",
    )
    parser.add_argument("--sigma-source-accuracy", action="append", type=float)


def build_parser() -> argparse.ArgumentParser:
    """Build all cognitive-comparison subcommands."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup")
    add_corpus_argument(setup)
    add_common_processed_argument(setup)
    add_et1_arguments(setup)
    setup.add_argument(
        "--onestop-chunksize",
        type=int,
        default=ONESTOP_DEFAULT_CHUNK_SIZE,
    )
    setup.add_argument("--skip-et1", action="store_true")
    setup.set_defaults(handler=command_setup)

    audit = subparsers.add_parser("audit")
    add_corpus_argument(audit)
    add_common_processed_argument(audit)
    audit.set_defaults(handler=command_audit)

    prepare = subparsers.add_parser("prepare-provo")
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    prepare.set_defaults(handler=command_prepare)

    prepare_onestop = subparsers.add_parser("prepare-onestop")
    prepare_onestop.add_argument(
        "--input-zip",
        type=Path,
        default=ONESTOP_RAW_DIR / ONESTOP_ZIP_FILENAME,
    )
    prepare_onestop.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ONESTOP_PROCESSED_DIR,
    )
    prepare_onestop.add_argument(
        "--chunksize",
        type=int,
        default=ONESTOP_DEFAULT_CHUNK_SIZE,
    )
    prepare_onestop.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    prepare_onestop.set_defaults(handler=command_prepare_onestop)

    sigmas = subparsers.add_parser("extract-sigmas")
    add_sigma_arguments(sigmas, required=True)
    sigmas.add_argument("--output-dir", type=Path, required=True)
    sigmas.set_defaults(handler=command_extract_sigmas)

    et1 = subparsers.add_parser("predict-et1")
    add_corpus_argument(et1)
    add_common_processed_argument(et1)
    add_sigma_arguments(et1, required=False)
    add_direct_sigma_arguments(et1)
    add_et1_arguments(et1)
    et1.add_argument("--sigma-json", type=Path)
    et1.add_argument("--output-dir", type=Path, required=True)
    et1.add_argument(
        "--exclude-special-tokens-from-redistribution",
        action="store_true",
    )
    et1.set_defaults(handler=command_predict_et1)

    ob1 = subparsers.add_parser("simulate-ob1")
    add_corpus_argument(ob1)
    add_common_processed_argument(ob1)
    ob1.add_argument("--runtime-dir", type=Path)
    ob1.add_argument("--output-dir", type=Path, required=True)
    ob1.add_argument("--seeds", default="0:100")
    ob1.add_argument("--n-trials", type=int)
    ob1.add_argument("--python-hash-seed", type=int, default=20260725)
    ob1.add_argument("--workers", type=int, default=1)
    ob1.set_defaults(handler=command_simulate_ob1)

    evaluate = subparsers.add_parser("evaluate")
    add_corpus_argument(evaluate)
    add_common_processed_argument(evaluate)
    evaluate.add_argument("--et1-dir", type=Path, required=True)
    evaluate.add_argument("--ob1-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument(
        "--human-target",
        choices=("human_trt_unconditional", "human_trt_conditional"),
        default="human_trt_unconditional",
    )
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=20260725)
    evaluate.add_argument(
        "--with-ob1-clean-passage-sensitivity",
        action="store_true",
    )
    evaluate.set_defaults(handler=command_evaluate)

    attention = subparsers.add_parser("compare-attention-profile")
    add_corpus_argument(attention)
    add_common_processed_argument(attention)
    add_sigma_arguments(attention, required=False)
    add_direct_sigma_arguments(attention)
    attention.add_argument("--sigma-json", type=Path)
    attention.add_argument("--et1-dir", type=Path, required=True)
    attention.add_argument("--ob1-dir", type=Path, required=True)
    attention.add_argument("--output-dir", type=Path, required=True)
    attention.add_argument(
        "--ob1-attention-skew",
        action="append",
        type=float,
    )
    attention.add_argument(
        "--fixation-weighting",
        choices=("duration", "equal"),
        default="duration",
    )
    attention.add_argument(
        "--profile-component",
        choices=("focused", "full"),
        default="focused",
    )
    attention.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
    )
    attention.add_argument("--seed", type=int, default=20260725)
    attention.set_defaults(handler=command_compare_attention_profile)

    run = subparsers.add_parser("run")
    add_corpus_argument(run)
    add_common_processed_argument(run)
    add_sigma_arguments(run, required=False)
    add_direct_sigma_arguments(run)
    add_et1_arguments(run)
    run.add_argument("--runtime-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--seeds", default="0:100")
    run.add_argument("--python-hash-seed", type=int, default=20260725)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--bootstrap-samples", type=int, default=10000)
    run.add_argument("--seed", type=int, default=20260725)
    run.add_argument(
        "--with-special-token-sensitivity",
        action="store_true",
    )
    run.add_argument(
        "--with-ob1-clean-passage-sensitivity",
        action="store_true",
    )
    run.set_defaults(handler=command_run)
    return parser


def main() -> None:
    """Dispatch one cognitive-comparison subcommand."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

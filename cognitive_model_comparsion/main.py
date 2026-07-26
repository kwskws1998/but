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
    summarize_methods,
    summarize_methods_by_checkpoint,
    write_evaluation_outputs,
)
from cognitive_model_comparsion.src.ob1_runner import (
    DEFAULT_RUNTIME_DIR,
    aggregate_ob1_tvt,
    prepare_ob1_runtime,
    run_ob1_subprocess,
    write_ob1_aggregation,
)
from cognitive_model_comparsion.src.prepare_provo import (
    EYE_FILENAME,
    PREDICTABILITY_FILENAME,
    RAW_DIR,
    build_canonical_tables,
    write_canonical_tables,
)
from cognitive_model_comparsion.src.sigmas import (
    direct_sigma_record,
    extract_sigma_record,
)


DEFAULT_PROCESSED_DIR = ROOT / "data/processed"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/full_run"


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


def validate_trial_count(n_trials: int) -> None:
    """Require a nonempty prefix of the 55 published Provo passages."""
    if not 1 <= n_trials <= 55:
        raise ValueError("--n-trials must be between 1 and 55")


def json_safe(value):
    """Convert argparse values into JSON-serializable provenance values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def ensure_prepared(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load canonical Provo tables, creating them if absent."""
    passages_path = processed_dir / "provo_passages.csv"
    words_path = processed_dir / "provo_words.csv"
    if not passages_path.is_file() or not words_path.is_file():
        artifacts = build_canonical_tables(
            RAW_DIR / EYE_FILENAME,
            RAW_DIR / PREDICTABILITY_FILENAME,
        )
        write_canonical_tables(processed_dir, *artifacts)
    return pd.read_csv(passages_path), pd.read_csv(words_path)


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
) -> dict:
    """Run one frozen ET1 pass and all fixed redistribution checkpoints."""
    passages, words = ensure_prepared(processed_dir)
    predictor = ET1NativePredictor(
        checkpoint_path=et1_checkpoint,
        tokenizer_path=et1_tokenizer,
    )
    artifacts = run_et1_inference(
        passages,
        words,
        sigma_records,
        predictor,
    )
    write_et1_outputs(output_dir, *artifacts)
    return artifacts[-1]


def run_simulate_ob1(
    processed_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int,
    python_hash_seed: int,
    workers: int,
) -> dict:
    """Prepare runtime, execute OB1, and aggregate word-level TVT."""
    validate_trial_count(n_trials)
    passages, words = ensure_prepared(processed_dir)
    preparation = prepare_ob1_runtime(
        passages,
        runtime_dir,
        python_hash_seed=python_hash_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
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
        n_trials=n_trials,
        python_hash_seed=python_hash_seed,
        workers=workers,
    )
    fixations = pd.read_csv(output_dir / "ob1_fixations.csv")
    eligible_words = words[words["passage_id_zero_based"] < n_trials].copy()
    artifacts = aggregate_ob1_tvt(fixations, eligible_words)
    write_ob1_aggregation(output_dir, *artifacts)
    return artifacts[-1]


def run_evaluate(
    processed_dir: Path,
    et1_dir: Path,
    ob1_dir: Path,
    output_dir: Path,
    human_target: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Join saved model values and produce all statistical outputs."""
    _, canonical_words = ensure_prepared(processed_dir)
    et1_words = pd.read_csv(et1_dir / "et1_word_values.csv")
    ob1_words = pd.read_csv(ob1_dir / "ob1_word_values.csv")
    word_values = merge_word_values(canonical_words, et1_words, ob1_words)
    passage_metrics = evaluate_passages(word_values, human_target)
    method_summary = summarize_methods(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    checkpoint_summary = summarize_methods_by_checkpoint(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    contrasts = paired_contrasts(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    audit = {
        "human_target": human_target,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "checkpoints": int(word_values["checkpoint_id"].nunique()),
        "passages": int(word_values["passage_id_zero_based"].nunique()),
        "word_rows": len(word_values),
        "passage_metric_rows": len(passage_metrics),
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
    write_evaluation_outputs(
        output_dir,
        word_values,
        passage_metrics,
        method_summary,
        checkpoint_summary,
        contrasts,
        audit,
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
    """Download and validate all comparison assets and optional ET1 assets."""
    download_assets("all")
    verify_assets("all")
    artifacts = build_canonical_tables(
        RAW_DIR / EYE_FILENAME,
        RAW_DIR / PREDICTABILITY_FILENAME,
    )
    write_canonical_tables(args.processed_dir, *artifacts)
    if not args.skip_et1:
        from models.et_checkpoint import ensure_et1_checkpoint
        from models.et1_tokenizer import load_et1_tokenizer

        ensure_et1_checkpoint(args.et1_checkpoint)
        load_et1_tokenizer(tokenizer_path=args.et1_tokenizer)


def command_audit(args: argparse.Namespace) -> None:
    """Print combined official Provo and ET2 provenance audits."""
    verify_assets("all")
    report = {
        "provo": audit_dataset(RAW_DIR),
        "et2_provo": compare_et2_provo(
            RAW_DIR / EYE_FILENAME,
            ROOT / "third_party/et2_torontocl_cmcl_2021/data/provo.csv",
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def command_prepare(args: argparse.Namespace) -> None:
    """Build and print the canonical Human Provo grid."""
    artifacts = build_canonical_tables(
        RAW_DIR / EYE_FILENAME,
        RAW_DIR / PREDICTABILITY_FILENAME,
    )
    write_canonical_tables(args.output_dir, *artifacts)
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
    audit = run_predict_et1(
        args.processed_dir,
        args.output_dir,
        records,
        args.et1_checkpoint,
        args.et1_tokenizer,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_simulate_ob1(args: argparse.Namespace) -> None:
    """Run the exact baseline OB1 virtual readers and aggregate TVT."""
    audit = run_simulate_ob1(
        args.processed_dir,
        args.runtime_dir,
        args.output_dir,
        parse_seed_specification(args.seeds),
        args.n_trials,
        args.python_hash_seed,
        args.workers,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate saved ET1 and OB1 word allocations."""
    audit = run_evaluate(
        args.processed_dir,
        args.et1_dir,
        args.ob1_dir,
        args.output_dir,
        args.human_target,
        args.bootstrap_samples,
        args.seed,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    """Execute setup, inference, OB1, and both Human TRT evaluations."""
    validate_predict_sigma_arguments(args)
    if not args.checkpoint and not args.sigma_left:
        raise ValueError(
            "run requires --checkpoint or direct --sigma-left/--sigma-right values"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            runtime_manifest(args, output_dir),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    download_assets("all")
    artifacts = build_canonical_tables(
        RAW_DIR / EYE_FILENAME,
        RAW_DIR / PREDICTABILITY_FILENAME,
    )
    write_canonical_tables(args.processed_dir, *artifacts)
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
        args.processed_dir,
        output_dir / "et1",
        records,
        args.et1_checkpoint,
        args.et1_tokenizer,
    )
    seeds = parse_seed_specification(args.seeds)
    run_simulate_ob1(
        args.processed_dir,
        args.runtime_dir,
        output_dir / "ob1",
        seeds,
        55,
        args.python_hash_seed,
        args.workers,
    )
    run_evaluate(
        args.processed_dir,
        output_dir / "et1",
        output_dir / "ob1",
        output_dir / "evaluation_unconditional",
        "human_trt_unconditional",
        args.bootstrap_samples,
        args.seed,
    )
    run_evaluate(
        args.processed_dir,
        output_dir / "et1",
        output_dir / "ob1",
        output_dir / "evaluation_conditional",
        "human_trt_conditional",
        args.bootstrap_samples,
        args.seed,
    )


def add_common_processed_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared canonical-table directory argument."""
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
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
    add_common_processed_argument(setup)
    add_et1_arguments(setup)
    setup.add_argument("--skip-et1", action="store_true")
    setup.set_defaults(handler=command_setup)

    audit = subparsers.add_parser("audit")
    audit.set_defaults(handler=command_audit)

    prepare = subparsers.add_parser("prepare-provo")
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    prepare.set_defaults(handler=command_prepare)

    sigmas = subparsers.add_parser("extract-sigmas")
    add_sigma_arguments(sigmas, required=True)
    sigmas.add_argument("--output-dir", type=Path, required=True)
    sigmas.set_defaults(handler=command_extract_sigmas)

    et1 = subparsers.add_parser("predict-et1")
    add_common_processed_argument(et1)
    add_sigma_arguments(et1, required=False)
    add_direct_sigma_arguments(et1)
    add_et1_arguments(et1)
    et1.add_argument("--sigma-json", type=Path)
    et1.add_argument("--output-dir", type=Path, required=True)
    et1.set_defaults(handler=command_predict_et1)

    ob1 = subparsers.add_parser("simulate-ob1")
    add_common_processed_argument(ob1)
    ob1.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    ob1.add_argument("--output-dir", type=Path, required=True)
    ob1.add_argument("--seeds", default="0:100")
    ob1.add_argument("--n-trials", type=int, default=55)
    ob1.add_argument("--python-hash-seed", type=int, default=20260725)
    ob1.add_argument("--workers", type=int, default=1)
    ob1.set_defaults(handler=command_simulate_ob1)

    evaluate = subparsers.add_parser("evaluate")
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
    evaluate.set_defaults(handler=command_evaluate)

    run = subparsers.add_parser("run")
    add_common_processed_argument(run)
    add_sigma_arguments(run, required=False)
    add_direct_sigma_arguments(run)
    add_et1_arguments(run)
    run.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--seeds", default="0:100")
    run.add_argument("--python-hash-seed", type=int, default=20260725)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--bootstrap-samples", type=int, default=10000)
    run.add_argument("--seed", type=int, default=20260725)
    run.set_defaults(handler=command_run)
    return parser


def main() -> None:
    """Dispatch one cognitive-comparison subcommand."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

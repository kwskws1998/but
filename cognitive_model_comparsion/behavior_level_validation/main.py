"""Run isolated word-level Human-ET1-OB1 behavior validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
COGNITIVE_ROOT = HERE.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cognitive_model_comparsion.behavior_level_validation.analysis import (
    write_behavior_report,
)
from cognitive_model_comparsion.main import (
    ensure_prepared,
    load_sigma_records,
)
from cognitive_model_comparsion.src.evaluate import (
    evaluate_passages,
    merge_word_values,
    paired_contrasts,
    paired_contrasts_by_checkpoint,
    summarize_methods,
    summarize_methods_by_checkpoint,
    write_evaluation_outputs,
)


DEFAULT_PROCESSED_DIR = COGNITIVE_ROOT / "data/processed"


def read_json_if_present(path: Path) -> dict:
    """Read one optional JSON object."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def select_et1_checkpoints(
    et1_words: pd.DataFrame,
    checkpoint_id: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select exactly one checkpoint without averaging learned sigmas."""
    if "checkpoint_id" not in et1_words:
        raise ValueError("ET1 word values are missing checkpoint_id")
    available = sorted(et1_words["checkpoint_id"].astype(str).unique())
    if not available:
        raise ValueError("ET1 word values contain no checkpoints")
    if checkpoint_id is not None:
        if checkpoint_id not in available:
            raise ValueError(
                f"Unknown checkpoint {checkpoint_id!r}; available: {available}"
            )
        selected = et1_words.loc[
            et1_words["checkpoint_id"].astype(str).eq(checkpoint_id)
        ].copy()
        return selected, [checkpoint_id]
    if len(available) > 1:
        raise ValueError(
            "ET1 input contains multiple checkpoints. Pass one "
            f"--checkpoint-id: {available}"
        )
    return et1_words.copy(), available


def load_checkpoint_metadata(
    et1_dir: Path,
    selected_ids: list[str],
) -> pd.DataFrame:
    """Load and filter required sigma provenance for selected checkpoints."""
    path = et1_dir / "checkpoint_sigmas.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Behavior validation requires sigma provenance: "
            f"{path}"
        )
    metadata = pd.DataFrame(load_sigma_records(path))
    selected = metadata.loc[
        metadata["checkpoint_id"].astype(str).isin(selected_ids)
    ].copy()
    missing = sorted(
        set(selected_ids) - set(selected["checkpoint_id"].astype(str))
    )
    if missing:
        raise ValueError(
            f"Sigma metadata is missing selected checkpoints: {missing}"
        )
    return selected


def run_behavior_evaluation(
    processed_dir: Path,
    et1_dir: Path,
    ob1_dir: Path,
    output_dir: Path,
    checkpoint_id: str | None,
    expected_ob1_readers: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Evaluate actual ET1 redistributions against Human TRT and OB1 TVT."""
    if bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    processed_dir = processed_dir.expanduser().resolve()
    et1_dir = et1_dir.expanduser().resolve()
    ob1_dir = ob1_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    _, canonical_words = ensure_prepared(processed_dir, "provo")
    et1_path = et1_dir / "et1_word_values.csv"
    ob1_path = ob1_dir / "ob1_word_values.csv"
    if not et1_path.is_file():
        raise FileNotFoundError(et1_path)
    if not ob1_path.is_file():
        raise FileNotFoundError(ob1_path)

    et1_words, selected_ids = select_et1_checkpoints(
        pd.read_csv(et1_path),
        checkpoint_id,
    )
    ob1_words = pd.read_csv(ob1_path)
    word_values = merge_word_values(
        canonical_words,
        et1_words,
        ob1_words,
    )
    passage_metrics = evaluate_passages(
        word_values,
        "human_trt_conditional",
    )
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
    checkpoint_contrasts = paired_contrasts_by_checkpoint(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    checkpoint_metadata = load_checkpoint_metadata(
        et1_dir,
        selected_ids,
    )

    et1_audit = read_json_if_present(
        et1_dir / "et1_inference_audit.json"
    )
    ob1_audit = read_json_if_present(
        ob1_dir / "ob1_aggregation_audit.json"
    )
    actual_ob1_readers = ob1_audit.get("virtual_readers")
    if expected_ob1_readers < 1:
        raise ValueError("--expected-ob1-readers must be positive")
    if actual_ob1_readers != expected_ob1_readers:
        raise ValueError(
            "OB1 virtual-reader count mismatch: expected "
            f"{expected_ob1_readers}, found {actual_ob1_readers!r}"
        )
    for source_name, source_audit in (
        ("ET1", et1_audit),
        ("OB1", ob1_audit),
    ):
        source_corpus = source_audit.get("corpus")
        if source_corpus is not None and source_corpus != "provo":
            raise ValueError(
                f"{source_name} audit corpus is {source_corpus!r}, "
                "expected 'provo'"
            )
    audit = {
        "analysis": "actual_et1_word_level_behavior_validation",
        "corpus": "provo",
        "human_target": "human_trt_conditional",
        "human_zero_dwell_policy": (
            "exclude zero-dwell reader-word observations from Human TRT mean"
        ),
        "et1_source_dir": str(et1_dir),
        "ob1_source_dir": str(ob1_dir),
        "selected_checkpoint_ids": selected_ids,
        "checkpoint_metadata": checkpoint_metadata.to_dict("records"),
        "actual_passage_specific_et1_values_used": True,
        "unit_impulse_kernel_profiles_used": False,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "passages": int(
            passage_metrics["passage_id_zero_based"].nunique()
        ),
        "word_rows": int(
            word_values[
                ["passage_id_zero_based", "word_id_zero_based"]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "ob1_virtual_readers": actual_ob1_readers,
        "et1_predictor_cache_signature": et1_audit.get(
            "predictor_cache_signature"
        ),
    }
    evaluation_dir = output_dir / "evaluation_conditional"
    write_evaluation_outputs(
        evaluation_dir,
        word_values,
        passage_metrics,
        method_summary,
        checkpoint_summary,
        contrasts,
        audit,
        checkpoint_contrasts,
        checkpoint_metadata,
    )
    report_audit = write_behavior_report(
        evaluation_dir,
        output_dir,
    )
    return {
        "output_dir": str(output_dir),
        "evaluation_dir": str(evaluation_dir),
        "selected_checkpoint_ids": selected_ids,
        "passages": audit["passages"],
        "word_rows": audit["word_rows"],
        "report_outputs": report_audit["outputs"],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the behavior-validation command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare actual ET1 word-level TRT allocation with Human Provo "
            "conditional TRT and OB1 simulated TVT."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run conditional word-level evaluation from saved ET1 and OB1 data.",
    )
    evaluate.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    evaluate.add_argument("--et1-dir", type=Path, required=True)
    evaluate.add_argument("--ob1-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint-id")
    evaluate.add_argument("--expected-ob1-readers", type=int, default=100)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=20260725)

    summarize = subparsers.add_parser(
        "summarize",
        help="Build compact reviewer tables from a completed conditional run.",
    )
    summarize.add_argument("--evaluation-dir", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    """Dispatch behavior-level evaluation and reporting commands."""
    args = build_parser().parse_args()
    if args.command == "evaluate":
        result = run_behavior_evaluation(
            processed_dir=args.processed_dir,
            et1_dir=args.et1_dir,
            ob1_dir=args.ob1_dir,
            output_dir=args.output_dir,
            checkpoint_id=args.checkpoint_id,
            expected_ob1_readers=args.expected_ob1_readers,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    else:
        report_audit = write_behavior_report(
            args.evaluation_dir,
            args.output_dir,
        )
        result = {
            "output_dir": str(args.output_dir.expanduser().resolve()),
            "human_target": report_audit["human_target"],
            "outputs": report_audit["outputs"],
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

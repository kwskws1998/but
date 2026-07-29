"""Combine 12 individual Provo attention runs and compute one aggregate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
REPOSITORY_ROOT = COGNITIVE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cognitive_model_comparsion.scripts.summarize_provo_12_checkpoint_attention import (
    EXPECTED_ET1_SHA256,
    EXPECTED_OB1_MANIFEST_SHA256,
    EXPECTED_OB1_SHA256,
    EXPECTED_SIGMA_SHA256,
    run_summary,
    sha256_file,
)


DEFAULT_INPUT_ROOT = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_rebuttal_experiment"
    / "attention_profile_individual_centered_sd_20260729"
)
DEFAULT_SIGMA_JSON = (
    COGNITIVE_ROOT / "configs" / "reviewer_sigma_sweep_12.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "aggregate_12_checkpoints"
COMBINED_FILES = (
    "kernel_alignment_by_passage.csv",
    "kernel_alignment_result_table.csv",
    "kernel_profiles.csv",
    "gaussian_parameter_diagnostics.csv",
)


def load_sigma_records(path: Path) -> list[dict]:
    """Load and validate the exact ordered 12-checkpoint configuration."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != EXPECTED_SIGMA_SHA256:
        raise ValueError("Sigma JSON SHA-256 does not match the frozen config")
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("Sigma JSON must contain exactly 12 records")
    for index, record in enumerate(records, start=1):
        expected_prefix = f"s{index:02d}_"
        if not str(record["checkpoint_id"]).startswith(expected_prefix):
            raise ValueError(
                f"Checkpoint {index} does not start with {expected_prefix}"
            )
    return records


def require_close(found: object, expected: object, label: str) -> None:
    """Require one scalar metadata value to match the frozen configuration."""
    if expected is None:
        if found is not None:
            raise ValueError(f"{label}: expected None, found {found}")
        return
    if isinstance(expected, (float, int)):
        if not math.isclose(
            float(found),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{label}: expected {expected}, found {found}"
            )
        return
    if found != expected:
        raise ValueError(f"{label}: expected {expected}, found {found}")


def validate_individual_audit(
    audit: dict,
    expected_record: dict,
    run_dir: Path,
) -> dict:
    """Validate one direct single-checkpoint analysis audit."""
    expected_audit = {
        "checkpoint_count": 1,
        "passage_count": 55,
        "ob1_fixations_sha256": EXPECTED_OB1_SHA256,
        "et1_token_values_sha256": EXPECTED_ET1_SHA256,
        "ob1_worker_manifest_sha256": EXPECTED_OB1_MANIFEST_SHA256,
        "candidate_support_policy": "fixation_matched",
        "profile_component": "focused",
        "fixation_weighting": "duration",
        "actual_et1_trt_magnitudes_used": False,
        "trajectory_attention_skew": 3.0,
        "support_rms_displacement_controls_enabled": False,
        "support_centered_sd_controls_enabled": True,
    }
    mismatches = {
        key: {"expected": value, "found": audit.get(key)}
        for key, value in expected_audit.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{run_dir} audit mismatch: {mismatches}")
    learned_records = audit.get("learned_sigma_records")
    if not isinstance(learned_records, list) or len(learned_records) != 1:
        raise ValueError(f"{run_dir} must contain one learned sigma record")
    learned = learned_records[0]
    require_close(
        learned.get("checkpoint_id"),
        expected_record["checkpoint_id"],
        f"{run_dir} checkpoint_id",
    )
    require_close(
        learned.get("source_accuracy"),
        expected_record["source_accuracy"],
        f"{run_dir} source_accuracy",
    )
    require_close(
        learned.get("learned_sigma_left"),
        expected_record["sigma_left"],
        f"{run_dir} sigma_left",
    )
    require_close(
        learned.get("learned_sigma_right"),
        expected_record["sigma_right"],
        f"{run_dir} sigma_right",
    )
    centered_matches = audit.get("support_centered_sd_matches")
    checkpoint_id = str(expected_record["checkpoint_id"])
    if (
        not isinstance(centered_matches, dict)
        or set(centered_matches) != {checkpoint_id}
    ):
        raise ValueError(f"{run_dir} centered-SD audit is incomplete")
    for control in ("symmetric_ratio1", "fixed_ratio4"):
        error = float(
            centered_matches[checkpoint_id][control][
                "absolute_match_error"
            ]
        )
        if not math.isfinite(error) or error > 1e-7:
            raise ValueError(
                f"{run_dir} {control} centered-SD error is {error}"
            )
    return learned


def load_individual_run(
    run_dir: Path,
    expected_record: dict,
) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    """Load one complete direct run and require one matching checkpoint."""
    required_paths = {
        name: run_dir / name
        for name in COMBINED_FILES
    }
    audit_path = run_dir / "attention_profile_audit.json"
    missing = [
        str(path)
        for path in (*required_paths.values(), audit_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing individual-run files: {missing}")
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    learned = validate_individual_audit(
        audit,
        expected_record,
        run_dir,
    )
    checkpoint_id = str(expected_record["checkpoint_id"])
    frames = {}
    for name, path in required_paths.items():
        frame = pd.read_csv(path)
        if "checkpoint_id" not in frame:
            raise ValueError(f"{path} is missing checkpoint_id")
        found_ids = set(frame["checkpoint_id"].astype(str))
        if found_ids != {checkpoint_id}:
            raise ValueError(
                f"{path} checkpoint IDs differ: {sorted(found_ids)}"
            )
        frames[name] = frame
    source_record = {
        "run_dir": str(run_dir),
        "checkpoint_id": checkpoint_id,
        "attention_profile_audit_sha256": sha256_file(audit_path),
        "artifact_sha256": {
            name: sha256_file(path)
            for name, path in required_paths.items()
        },
    }
    return frames, audit, source_record


def build_combined_source(
    input_root: Path,
    combined_dir: Path,
    sigma_records: list[dict],
    sigma_json: Path,
) -> dict:
    """Combine all single-checkpoint artifacts into one validated source."""
    frames_by_name = {
        name: []
        for name in COMBINED_FILES
    }
    learned_records = []
    centered_matches = {}
    source_records = []
    base_audit = None
    for index, expected_record in enumerate(sigma_records, start=1):
        run_dir = input_root / f"s{index:02d}"
        frames, audit, source_record = load_individual_run(
            run_dir,
            expected_record,
        )
        for name, frame in frames.items():
            frames_by_name[name].append(frame)
        learned_record = audit["learned_sigma_records"][0]
        learned_records.append(learned_record)
        centered_matches.update(audit["support_centered_sd_matches"])
        source_records.append(source_record)
        if base_audit is None:
            base_audit = dict(audit)
    if base_audit is None:
        raise ValueError("No individual checkpoint runs were loaded")
    combined_dir.mkdir(parents=True, exist_ok=False)
    for name, frames in frames_by_name.items():
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(combined_dir / name, index=False)
    for key in learned_records[0]:
        base_audit.pop(key, None)
    base_audit.update(
        {
            "checkpoint_count": 12,
            "learned_sigma_records": learned_records,
            "support_centered_sd_matches": centered_matches,
            "sigma_json_path": str(sigma_json.resolve()),
            "sigma_json_sha256": EXPECTED_SIGMA_SHA256,
            "combined_from_individual_runs": True,
            "individual_run_count": 12,
        }
    )
    with (combined_dir / "attention_profile_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(base_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "analysis": "combine_twelve_individual_provo_attention_runs",
        "input_root": str(input_root),
        "combined_dir": str(combined_dir),
        "sigma_json": str(sigma_json),
        "sigma_json_sha256": EXPECTED_SIGMA_SHA256,
        "individual_runs": source_records,
        "ob1_simulation_rerun": False,
        "et1_prediction_rerun": False,
    }
    with (combined_dir / "individual_run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def run(args: argparse.Namespace) -> dict:
    """Combine the 12 run directories and execute the aggregate analysis."""
    input_root = args.input_root.expanduser().resolve()
    sigma_json = args.sigma_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        raise FileExistsError(
            f"Aggregate output already exists: {output_root}. "
            "Pass a new --output-root."
        )
    output_root.mkdir(parents=True)
    combined_dir = output_root / "combined_source"
    summary_dir = output_root / "summary"
    sigma_records = load_sigma_records(sigma_json)
    manifest = build_combined_source(
        input_root,
        combined_dir,
        sigma_records,
        sigma_json,
    )
    result = run_summary(
        Namespace(
            analysis_dir=combined_dir,
            output_dir=summary_dir,
            expected_checkpoints=12,
            expected_passages=55,
            expected_ob1_sha256=EXPECTED_OB1_SHA256,
            expected_et1_sha256=EXPECTED_ET1_SHA256,
            expected_ob1_manifest_sha256=EXPECTED_OB1_MANIFEST_SHA256,
            expected_sigma_sha256=EXPECTED_SIGMA_SHA256,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    )
    return {
        **result,
        "input_root": str(input_root),
        "combined_dir": str(combined_dir),
        "individual_run_count": len(manifest["individual_runs"]),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the individual-run aggregation command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine s01 through s12 direct attention-profile runs and "
            "compute equal-checkpoint aggregate metrics and figures."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--sigma-json",
        type=Path,
        default=DEFAULT_SIGMA_JSON,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main() -> None:
    """Run aggregation and print the output paths as JSON."""
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

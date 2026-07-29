"""Generate skew-4 OB1 trajectories and the final six-checkpoint comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import pandas as pd


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
REPOSITORY_ROOT = COGNITIVE_ROOT.parent
DEFAULT_PYTHON = (
    REPOSITORY_ROOT
    / ".venv-cognitive-model-comparison"
    / "bin"
    / "python"
)
DEFAULT_PROCESSED_DIR = COGNITIVE_ROOT / "data" / "processed"
DEFAULT_EXISTING_EXPERIMENT_DIR = (
    COGNITIVE_ROOT / "outputs" / "provo_rebuttal_experiment"
)
DEFAULT_SIGMA_JSON = (
    COGNITIVE_ROOT / "configs" / "reviewer_sigma_llama3_8b_6.json"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_EXISTING_EXPERIMENT_DIR
    / "trajectory_matched_skew3_skew4_20260729"
)
DEFAULT_RUNTIME_DIR = (
    COGNITIVE_ROOT
    / "data"
    / "ob1_runtime"
    / "provo_rebuttal_skew4"
)
EXPECTED_SEEDS = list(range(100))
EXPECTED_PASSAGES = 55
EXPECTED_CHECKPOINTS = 6


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically write one JSON audit."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def emit(message: str, log_handle: TextIO) -> None:
    """Write one line to the terminal and experiment log."""
    print(message, flush=True)
    log_handle.write(message + "\n")
    log_handle.flush()


def run_logged(
    command: list[str],
    log_handle: TextIO,
    environment: dict[str, str],
) -> None:
    """Run one subprocess while streaming combined output."""
    emit(f"$ {shlex.join(command)}", log_handle)
    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=environment,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess output")
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def validate_sigma_json(path: Path) -> list[dict]:
    """Require the frozen six Llama-3-8B checkpoint records."""
    if not path.is_file():
        raise FileNotFoundError(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != EXPECTED_CHECKPOINTS:
        raise ValueError("Sigma JSON must contain six Llama-3-8B records")
    for index, record in enumerate(records, start=1):
        required = {
            "checkpoint_id",
            "source_accuracy",
            "sigma_left",
            "sigma_right",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(
                f"Sigma record {index} is missing fields: {missing}"
            )
        if not str(record["checkpoint_id"]).startswith(f"s{index:02d}_"):
            raise ValueError(
                f"Sigma record {index} has an unexpected checkpoint ID"
            )
        for key in ("sigma_left", "sigma_right"):
            value = float(record[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Sigma record {index} has invalid {key}")
        accuracy = float(record["source_accuracy"])
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(
                f"Sigma record {index} has invalid source_accuracy"
            )
    return records


def validate_ob1_output(path: Path, expected_skew: float) -> dict:
    """Validate one complete 100-reader trajectory output."""
    fixation_path = path / "ob1_fixations.csv"
    manifest_path = path / "ob1_worker_manifest.json"
    if not fixation_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Incomplete OB1 output under {path}: expected fixation CSV "
            "and worker manifest"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_skew = float(manifest["parameters"]["attention_skew"])
    if not math.isclose(
        actual_skew,
        expected_skew,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{path} used attention_skew={actual_skew:g}, expected "
            f"{expected_skew:g}"
        )
    seeds = [int(value) for value in manifest["seeds"]]
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"{path} does not contain seeds 0 through 99")
    if int(manifest["n_trials"]) != EXPECTED_PASSAGES:
        raise ValueError(f"{path} does not contain all 55 Provo passages")
    fixations = pd.read_csv(
        fixation_path,
        usecols=["simulation_id", "seed", "text_id"],
    )
    if int(manifest["fixation_rows"]) != len(fixations):
        raise ValueError(f"{path} fixation row count disagrees with manifest")
    if sorted(fixations["simulation_id"].unique().tolist()) != EXPECTED_SEEDS:
        raise ValueError(f"{path} does not contain 100 simulation IDs")
    if sorted(fixations["seed"].unique().tolist()) != EXPECTED_SEEDS:
        raise ValueError(f"{path} does not contain 100 unique seeds")
    if sorted(fixations["text_id"].unique().tolist()) != list(
        range(EXPECTED_PASSAGES)
    ):
        raise ValueError(f"{path} does not contain all 55 passage IDs")
    return {
        "path": str(path.resolve()),
        "attention_skew": actual_skew,
        "virtual_readers": len(EXPECTED_SEEDS),
        "passages": EXPECTED_PASSAGES,
        "fixation_rows": len(fixations),
        "ob1_fixations_sha256": sha256_file(fixation_path),
        "ob1_worker_manifest_sha256": sha256_file(manifest_path),
    }


def validate_analysis_output(path: Path, expected_skew: float) -> dict:
    """Validate one six-checkpoint trajectory-matched analysis."""
    required = (
        path / "attention_profile_audit.json",
        path / "kernel_profiles.csv",
        path / "reviewer_kernel_summary.csv",
    )
    for item in required:
        if not item.is_file():
            raise FileNotFoundError(item)
    audit = json.loads(required[0].read_text(encoding="utf-8"))
    expected_values = {
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "passage_count": EXPECTED_PASSAGES,
        "seed_count": len(EXPECTED_SEEDS),
        "candidate_support_policy": "fixation_matched",
        "profile_component": "focused",
        "fixation_weighting": "duration",
        "actual_et1_trt_magnitudes_used": False,
        "support_rms_displacement_controls_enabled": False,
        "support_centered_sd_controls_enabled": True,
    }
    mismatches = {
        key: {"expected": value, "found": audit.get(key)}
        for key, value in expected_values.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path} analysis audit mismatch: {mismatches}")
    trajectory_skew = float(audit["trajectory_attention_skew"])
    if not math.isclose(
        trajectory_skew,
        expected_skew,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{path} trajectory skew is {trajectory_skew:g}, expected "
            f"{expected_skew:g}"
        )
    profiles = pd.read_csv(required[1])
    metrics = pd.read_csv(required[2])
    for frame, label in ((profiles, "profiles"), (metrics, "metrics")):
        if set(frame["ob1_attention_skew"].astype(float)) != {expected_skew}:
            raise ValueError(f"{path} {label} contains another skew")
        matched = frame["requested_skew_matches_trajectory"].astype(
            str
        ).str.lower()
        if not matched.eq("true").all():
            raise ValueError(f"{path} {label} is not trajectory matched")
        checkpoints = frame["checkpoint_id"].astype(str).unique()
        if len(checkpoints) != EXPECTED_CHECKPOINTS:
            raise ValueError(f"{path} {label} does not contain six checkpoints")
    return {
        "path": str(path.resolve()),
        "attention_skew": trajectory_skew,
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "attention_profile_audit_sha256": sha256_file(required[0]),
        "kernel_profiles_sha256": sha256_file(required[1]),
        "reviewer_kernel_summary_sha256": sha256_file(required[2]),
    }


def require_empty_or_complete(
    path: Path,
    validator,
    *validator_args,
) -> dict | None:
    """Reuse a complete step and reject a mixed partial directory."""
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"Expected a directory: {path}")
    if not any(path.iterdir()):
        return None
    try:
        return validator(path, *validator_args)
    except Exception as error:
        raise RuntimeError(
            f"Existing output is incomplete or invalid: {path}. Pass a new "
            "--output-root instead of mixing runs."
        ) from error


def attention_analysis_command(
    python_path: Path,
    processed_dir: Path,
    sigma_json: Path,
    et1_dir: Path,
    ob1_dir: Path,
    skew: int,
    bootstrap_samples: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    """Build one trajectory-matched attention-profile command."""
    return [
        str(python_path),
        "-u",
        "cognitive_model_comparsion/main.py",
        "compare-attention-profile",
        "--corpus",
        "provo",
        "--processed-dir",
        str(processed_dir),
        "--sigma-json",
        str(sigma_json),
        "--et1-dir",
        str(et1_dir),
        "--ob1-dir",
        str(ob1_dir),
        "--ob1-attention-skew",
        str(skew),
        "--fixation-weighting",
        "duration",
        "--profile-component",
        "focused",
        "--candidate-support-policy",
        "fixation_matched",
        "--skip-support-rms-displacement-controls",
        "--with-support-centered-sd-controls",
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]


def run(args: argparse.Namespace) -> dict:
    """Execute simulation, both matched analyses, and final figure generation."""
    python_path = args.python.expanduser().absolute()
    processed_dir = args.processed_dir.expanduser().resolve()
    existing_experiment_dir = (
        args.existing_experiment_dir.expanduser().resolve()
    )
    sigma_json = args.sigma_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python environment is missing: {python_path}"
        )
    if not processed_dir.is_dir():
        raise FileNotFoundError(processed_dir)
    sigma_records = validate_sigma_json(sigma_json)
    et1_dir = existing_experiment_dir / "et1"
    skew3_ob1_dir = existing_experiment_dir / "ob1"
    if not (et1_dir / "et1_token_values.csv").is_file():
        raise FileNotFoundError(et1_dir / "et1_token_values.csv")
    skew3_ob1_audit = validate_ob1_output(skew3_ob1_dir, 3.0)

    output_root.mkdir(parents=True, exist_ok=True)
    skew4_ob1_dir = output_root / "ob1_skew4"
    skew3_analysis_dir = output_root / "attention_skew3_matched"
    skew4_analysis_dir = output_root / "attention_skew4_matched"
    figure_dir = output_root / "final_tables_and_figures"
    log_path = output_root / "experiment.log"
    runner_audit_path = output_root / "runner_audit.json"
    matplotlib_dir = output_root / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(matplotlib_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    runner_audit = {
        "status": "running",
        "started_at_utc": started_at,
        "python": str(python_path),
        "processed_dir": str(processed_dir),
        "existing_experiment_dir": str(existing_experiment_dir),
        "output_root": str(output_root),
        "runtime_dir": str(runtime_dir),
        "workers": int(args.workers),
        "seeds": EXPECTED_SEEDS,
        "passages": EXPECTED_PASSAGES,
        "sigma_json": str(sigma_json),
        "sigma_json_sha256": sha256_file(sigma_json),
        "checkpoint_ids": [
            str(record["checkpoint_id"]) for record in sigma_records
        ],
        "skew3_ob1": skew3_ob1_audit,
    }
    write_json_atomic(runner_audit_path, runner_audit)

    with log_path.open("a", encoding="utf-8") as log_handle:
        emit(f"started_at_utc={started_at}", log_handle)
        emit(f"output_root={output_root}", log_handle)
        try:
            skew4_ob1_audit = require_empty_or_complete(
                skew4_ob1_dir,
                validate_ob1_output,
                4.0,
            )
            if skew4_ob1_audit is None:
                simulation_command = [
                    str(python_path),
                    "-u",
                    "cognitive_model_comparsion/main.py",
                    "simulate-ob1",
                    "--corpus",
                    "provo",
                    "--processed-dir",
                    str(processed_dir),
                    "--runtime-dir",
                    str(runtime_dir),
                    "--output-dir",
                    str(skew4_ob1_dir),
                    "--seeds",
                    "0:100",
                    "--n-trials",
                    str(EXPECTED_PASSAGES),
                    "--python-hash-seed",
                    str(args.python_hash_seed),
                    "--workers",
                    str(args.workers),
                    "--attention-skew",
                    "4",
                ]
                run_logged(simulation_command, log_handle, environment)
                skew4_ob1_audit = validate_ob1_output(
                    skew4_ob1_dir,
                    4.0,
                )
            else:
                emit(
                    "skipping completed skew=4 OB1 simulation",
                    log_handle,
                )

            analysis_audits = {}
            for skew, ob1_dir, analysis_dir in (
                (3, skew3_ob1_dir, skew3_analysis_dir),
                (4, skew4_ob1_dir, skew4_analysis_dir),
            ):
                analysis_audit = require_empty_or_complete(
                    analysis_dir,
                    validate_analysis_output,
                    float(skew),
                )
                if analysis_audit is None:
                    command = attention_analysis_command(
                        python_path,
                        processed_dir,
                        sigma_json,
                        et1_dir,
                        ob1_dir,
                        skew,
                        args.bootstrap_samples,
                        args.seed,
                        analysis_dir,
                    )
                    run_logged(command, log_handle, environment)
                    analysis_audit = validate_analysis_output(
                        analysis_dir,
                        float(skew),
                    )
                else:
                    emit(
                        f"skipping completed skew={skew} attention analysis",
                        log_handle,
                    )
                analysis_audits[str(skew)] = analysis_audit

            figure_audit_path = figure_dir / "figure_audit.json"
            if figure_audit_path.is_file():
                figure_audit = json.loads(
                    figure_audit_path.read_text(encoding="utf-8")
                )
                emit(
                    "skipping completed table and figure generation",
                    log_handle,
                )
            else:
                if figure_dir.exists() and any(figure_dir.iterdir()):
                    raise RuntimeError(
                        f"Partial figure output exists: {figure_dir}. "
                        "Pass a new --output-root."
                    )
                figure_command = [
                    str(python_path),
                    "-u",
                    (
                        "cognitive_model_comparsion/scripts/"
                        "plot_llama3_8b_ob1_mean_profiles.py"
                    ),
                    "--skew3-analysis-dir",
                    str(skew3_analysis_dir),
                    "--skew4-analysis-dir",
                    str(skew4_analysis_dir),
                    "--expected-runs",
                    str(EXPECTED_CHECKPOINTS),
                    "--output-dir",
                    str(figure_dir),
                ]
                run_logged(figure_command, log_handle, environment)
                figure_audit = json.loads(
                    figure_audit_path.read_text(encoding="utf-8")
                )
        except Exception as error:
            runner_audit.update(
                {
                    "status": "failed",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            write_json_atomic(runner_audit_path, runner_audit)
            emit(f"status=failed error={error}", log_handle)
            raise

        runner_audit.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "skew4_ob1": skew4_ob1_audit,
                "attention_analyses": analysis_audits,
                "figure_audit": figure_audit,
                "log_path": str(log_path),
            }
        )
        write_json_atomic(runner_audit_path, runner_audit)
        emit("status=completed", log_handle)
    return runner_audit


def build_parser() -> argparse.ArgumentParser:
    """Build the end-to-end trajectory-matched experiment interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 100 OB1 trajectories with attention_skew=4, rerun "
            "the six Llama-3-8B kernel analyses against matching skew=3 and "
            "skew=4 trajectories, and create the reviewer table and figures."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--existing-experiment-dir",
        type=Path,
        default=DEFAULT_EXISTING_EXPERIMENT_DIR,
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
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
    )
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--python-hash-seed", type=int, default=20260725)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main() -> None:
    """Run the full experiment and print the final artifact paths."""
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

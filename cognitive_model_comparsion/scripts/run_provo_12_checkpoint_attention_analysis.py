"""Run all 12 frozen sigma pairs on saved current Provo/OB1 trajectories."""

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


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
REPOSITORY_ROOT = COGNITIVE_ROOT.parent
DEFAULT_PYTHON = (
    REPOSITORY_ROOT
    / ".venv-cognitive-model-comparison"
    / "bin"
    / "python"
)
DEFAULT_EXPERIMENT_DIR = (
    COGNITIVE_ROOT / "outputs" / "provo_rebuttal_experiment"
)
DEFAULT_PROCESSED_DIR = COGNITIVE_ROOT / "data" / "processed"
DEFAULT_SIGMA_JSON = (
    COGNITIVE_ROOT / "configs" / "reviewer_sigma_sweep_12.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_EXPERIMENT_DIR
    / "attention_profile_sigma_sweep_12_centered_sd_20260729"
)
EXPECTED_OB1_SHA256 = (
    "a8d84e1c264d23e7ab9efac20c1a727e08d16208be8c475ccad9b096fc1fb647"
)
EXPECTED_OB1_MANIFEST_SHA256 = (
    "65f2af0c1a351006b18ebef514611d0685bbdc6bdbd2163108392712927071bb"
)
EXPECTED_ET1_SHA256 = (
    "e2bd70e0d9fee9e956639509642e231c15786ac9b148a6fdcaa2b53e66de8fbd"
)
EXPECTED_SIGMA_SHA256 = (
    "3ced76e695053211c65deb38be04c588452418574ff3b78eaa0227cbc7cc3504"
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> str:
    """Require one exact current-run input digest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        )
    return actual


def validate_sigma_json(path: Path) -> list[dict]:
    """Require the exact ordered 12-record checkpoint configuration."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("Sigma JSON must contain exactly 12 records")
    identifiers = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Sigma record {index} must be an object")
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
        left = float(record["sigma_left"])
        right = float(record["sigma_right"])
        accuracy = float(record["source_accuracy"])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Every sigma pair must be finite")
        if left <= 0 or right <= 0:
            raise ValueError("Every sigma pair must be positive")
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(
                "Every source accuracy must be finite and in [0, 1]"
            )
        identifier = str(record["checkpoint_id"])
        expected_prefix = f"s{index + 1:02d}_"
        if not identifier.startswith(expected_prefix):
            raise ValueError(
                f"Sigma record {index} must start with {expected_prefix}"
            )
        identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Sigma checkpoint IDs must be unique")
    s11 = records[10]
    expected_s11 = {
        "source_accuracy": 0.76942,
        "sigma_left": 0.3738,
        "sigma_right": 3.21289,
    }
    for key, expected in expected_s11.items():
        if not math.isclose(
            float(s11[key]),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"s11 {key} must be {expected}, found {s11[key]}"
            )
    best_accuracy = max(float(record["source_accuracy"]) for record in records)
    if not math.isclose(
        float(s11["source_accuracy"]),
        best_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("s11 must be the highest-accuracy checkpoint")
    return records


def run_logged(
    command: list[str],
    cwd: Path,
    log_path: Path,
    mode: str,
    environment: dict[str, str],
) -> None:
    """Run one subprocess while streaming and saving combined output."""
    with log_path.open(mode, encoding="utf-8") as log:
        command_text = f"$ {shlex.join(command)}"
        print(command_text, flush=True)
        log.write(command_text + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("Subprocess stdout pipe was not created")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def write_runner_audit(path: Path, payload: dict) -> None:
    """Atomically write the runner state as deterministic JSON."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)


def run_experiment(args: argparse.Namespace) -> dict:
    """Validate current inputs, analyze 12 pairs, and aggregate the results."""
    python_path = args.python.expanduser().resolve()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    processed_dir = args.processed_dir.expanduser().resolve()
    sigma_json = args.sigma_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    et1_dir = experiment_dir / "et1"
    ob1_dir = experiment_dir / "ob1"
    et1_path = et1_dir / "et1_token_values.csv"
    ob1_path = ob1_dir / "ob1_fixations.csv"
    manifest_path = ob1_dir / "ob1_worker_manifest.json"

    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python environment is missing: {python_path}"
        )
    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"Processed Provo directory is missing: {processed_dir}"
        )
    sigma_records = validate_sigma_json(sigma_json)
    input_hashes = {
        "sigma_json": require_sha256(
            sigma_json,
            args.expected_sigma_sha256,
        ),
        "et1_token_values": require_sha256(
            et1_path,
            args.expected_et1_sha256,
        ),
        "ob1_fixations": require_sha256(
            ob1_path,
            args.expected_ob1_sha256,
        ),
        "ob1_worker_manifest": require_sha256(
            manifest_path,
            args.expected_ob1_manifest_sha256,
        ),
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass a new "
            "--output-dir so partial and completed runs cannot mix."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "checkpoint_aggregate"
    log_path = output_dir / "experiment.log"
    audit_path = output_dir / "runner_audit.json"
    matplotlib_dir = output_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(matplotlib_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "status": "running",
        "started_at_utc": started_at,
        "repository_root": str(REPOSITORY_ROOT),
        "python": str(python_path),
        "processed_dir": str(processed_dir),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "checkpoint_count": len(sigma_records),
        "checkpoint_ids": [
            str(record["checkpoint_id"]) for record in sigma_records
        ],
        "ob1_simulation_rerun": False,
        "et1_prediction_rerun": False,
        "input_hashes": input_hashes,
    }
    write_runner_audit(audit_path, audit)

    analysis_command = [
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
        "3",
        "--ob1-attention-skew",
        "4",
        "--fixation-weighting",
        "duration",
        "--profile-component",
        "focused",
        "--candidate-support-policy",
        "fixation_matched",
        "--skip-support-rms-displacement-controls",
        "--with-support-centered-sd-controls",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    summary_command = [
        str(python_path),
        "-u",
        (
            "cognitive_model_comparsion/scripts/"
            "summarize_provo_12_checkpoint_attention.py"
        ),
        "--analysis-dir",
        str(output_dir),
        "--output-dir",
        str(summary_dir),
        "--expected-ob1-sha256",
        args.expected_ob1_sha256,
        "--expected-et1-sha256",
        args.expected_et1_sha256,
        "--expected-ob1-manifest-sha256",
        args.expected_ob1_manifest_sha256,
        "--expected-sigma-sha256",
        args.expected_sigma_sha256,
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    try:
        run_logged(
            analysis_command,
            REPOSITORY_ROOT,
            log_path,
            "w",
            environment,
        )
        run_logged(
            summary_command,
            REPOSITORY_ROOT,
            log_path,
            "a",
            environment,
        )
    except Exception as error:
        audit.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_runner_audit(audit_path, audit)
        raise
    audit.update(
        {
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_command": analysis_command,
            "summary_command": summary_command,
            "summary_dir": str(summary_dir),
            "log_path": str(log_path),
        }
    )
    write_runner_audit(audit_path, audit)
    return {
        "status": audit["status"],
        "output_dir": str(output_dir),
        "summary_dir": str(summary_dir),
        "log_path": str(log_path),
        "checkpoint_count": len(sigma_records),
        "ob1_simulation_rerun": False,
        "et1_prediction_rerun": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment-runner command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all 12 frozen learned sigma pairs on the exact current "
            "saved Provo/OB1 trajectories and aggregate paired results."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--sigma-json",
        type=Path,
        default=DEFAULT_SIGMA_JSON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--expected-ob1-sha256",
        default=EXPECTED_OB1_SHA256,
    )
    parser.add_argument(
        "--expected-ob1-manifest-sha256",
        default=EXPECTED_OB1_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-et1-sha256",
        default=EXPECTED_ET1_SHA256,
    )
    parser.add_argument(
        "--expected-sigma-sha256",
        default=EXPECTED_SIGMA_SHA256,
    )
    return parser


def main() -> None:
    """Run the experiment and print a machine-readable completion record."""
    args = build_parser().parse_args()
    print(json.dumps(run_experiment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

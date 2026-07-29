"""Validate and package the trajectory-matched OB1 rebuttal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
REPOSITORY_ROOT = COGNITIVE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cognitive_model_comparsion.scripts.run_provo_skew4_trajectory_analysis import (
    DEFAULT_EXISTING_EXPERIMENT_DIR,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROCESSED_DIR,
    EXPECTED_CHECKPOINTS,
    validate_analysis_output,
    validate_ob1_output,
)


DEFAULT_ARCHIVE_PATH = (
    DEFAULT_OUTPUT_ROOT.parent
    / "trajectory_matched_skew3_skew4_20260729_verification.zip"
)
FINAL_FILENAMES = (
    "figure_audit.json",
    "figure_caption.txt",
    "llama3_8b_ob1_mean_profiles.pdf",
    "llama3_8b_ob1_mean_profiles.png",
    "llama3_8b_ob1_region_mass.pdf",
    "llama3_8b_ob1_region_mass.png",
    "mean_metric_table.csv",
    "mean_offset_profiles.csv",
    "mean_region_mass.csv",
    "region_mass_caption.txt",
    "reviewer_metric_table.csv",
    "reviewer_metric_table.md",
)
ANALYSIS_FILENAMES = (
    "attention_profile_audit.json",
    "checkpoint_sigmas.csv",
    "checkpoint_sigmas.json",
    "fixed_ob1_priors.json",
    "gaussian_parameter_diagnostics.csv",
    "kernel_alignment_by_passage.csv",
    "kernel_alignment_contrasts.csv",
    "kernel_alignment_result_table.csv",
    "kernel_directionality.csv",
    "kernel_profile_regions.csv",
    "kernel_profiles.csv",
    "reviewer_kernel_summary.csv",
)
OB1_FILENAMES = (
    "ob1_aggregation_audit.json",
    "ob1_fixations.csv",
    "ob1_runtime_preparation.json",
    "ob1_token_transformations.csv",
    "ob1_worker_manifest.json",
    "ob1_word_values.csv",
)
PROCESSED_FILENAMES = (
    "provo_excluded_positions.csv",
    "provo_passages.csv",
    "provo_prepare_audit.json",
    "provo_words.csv",
)
CODE_PATHS = (
    "cognitive_model_comparsion/README.md",
    "cognitive_model_comparsion/main.py",
    "cognitive_model_comparsion/requirements.txt",
    "cognitive_model_comparsion/configs/reviewer_sigma_llama3_8b_6.json",
    "cognitive_model_comparsion/scripts/plot_llama3_8b_ob1_mean_profiles.py",
    "cognitive_model_comparsion/scripts/run_provo_skew4_trajectory_analysis.py",
    "cognitive_model_comparsion/scripts/package_provo_skew4_trajectory_results.py",
    "cognitive_model_comparsion/src/attention_profile.py",
    "cognitive_model_comparsion/src/distribution_metrics.py",
    "cognitive_model_comparsion/src/ob1_runner.py",
    "cognitive_model_comparsion/src/ob1_worker.py",
    (
        "cognitive_model_comparsion/third_party/"
        "ob1_reader_provo_2024/src/parameters.py"
    ),
    (
        "cognitive_model_comparsion/third_party/"
        "ob1_reader_provo_2024/src/reading_components.py"
    ),
    (
        "cognitive_model_comparsion/third_party/"
        "ob1_reader_provo_2024/src/reading_helper_functions.py"
    ),
    (
        "cognitive_model_comparsion/third_party/"
        "ob1_reader_provo_2024/src/simulate_experiment.py"
    ),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_entry(
    entries: dict[str, Path],
    source: Path,
    archive_name: str,
) -> None:
    """Add one unique, existing file to the package plan."""
    if not source.is_file():
        raise FileNotFoundError(source)
    normalized_name = Path(archive_name).as_posix()
    if normalized_name.startswith("/") or ".." in Path(normalized_name).parts:
        raise ValueError(f"Unsafe archive path: {normalized_name}")
    if normalized_name in entries:
        raise ValueError(f"Duplicate archive path: {normalized_name}")
    entries[normalized_name] = source.resolve()


def collect_package_entries(
    processed_dir: Path,
    existing_experiment_dir: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Collect the minimal inputs, outputs, and code needed for verification."""
    entries: dict[str, Path] = {}
    for filename in PROCESSED_FILENAMES:
        add_entry(
            entries,
            processed_dir / filename,
            f"inputs/processed/{filename}",
        )
    et1_dir = existing_experiment_dir / "et1"
    for filename in (
        "et1_token_values.csv",
        "et1_inference_audit.json",
    ):
        add_entry(
            entries,
            et1_dir / filename,
            f"inputs/et1/{filename}",
        )
    for label, ob1_dir in (
        ("skew3", existing_experiment_dir / "ob1"),
        ("skew4", output_root / "ob1_skew4"),
    ):
        for filename in OB1_FILENAMES:
            add_entry(
                entries,
                ob1_dir / filename,
                f"inputs/ob1_{label}/{filename}",
            )
    for label, analysis_dir in (
        ("skew3", output_root / "attention_skew3_matched"),
        ("skew4", output_root / "attention_skew4_matched"),
    ):
        for filename in ANALYSIS_FILENAMES:
            add_entry(
                entries,
                analysis_dir / filename,
                f"analysis/{label}/{filename}",
            )
    final_dir = output_root / "final_tables_and_figures"
    for filename in FINAL_FILENAMES:
        add_entry(
            entries,
            final_dir / filename,
            f"final/{filename}",
        )
    for filename in ("experiment.log", "runner_audit.json"):
        add_entry(
            entries,
            output_root / filename,
            f"run/{filename}",
        )
    for relative_path in CODE_PATHS:
        add_entry(
            entries,
            REPOSITORY_ROOT / relative_path,
            f"code/{relative_path}",
        )
    return entries


def validate_completed_experiment(
    processed_dir: Path,
    existing_experiment_dir: Path,
    output_root: Path,
) -> dict:
    """Validate every final stage before packaging any files."""
    runner_audit_path = output_root / "runner_audit.json"
    if not runner_audit_path.is_file():
        raise FileNotFoundError(runner_audit_path)
    runner_audit = json.loads(
        runner_audit_path.read_text(encoding="utf-8")
    )
    if runner_audit.get("status") != "completed":
        raise RuntimeError(
            f"Experiment is not complete: status={runner_audit.get('status')}"
        )
    skew3_ob1 = validate_ob1_output(
        existing_experiment_dir / "ob1",
        3.0,
    )
    skew4_ob1 = validate_ob1_output(
        output_root / "ob1_skew4",
        4.0,
    )
    if (
        skew3_ob1["ob1_fixations_sha256"]
        == skew4_ob1["ob1_fixations_sha256"]
    ):
        raise ValueError("Skew 3 and skew 4 fixation files are identical")
    skew3_analysis = validate_analysis_output(
        output_root / "attention_skew3_matched",
        3.0,
    )
    skew4_analysis = validate_analysis_output(
        output_root / "attention_skew4_matched",
        4.0,
    )
    figure_audit_path = (
        output_root / "final_tables_and_figures" / "figure_audit.json"
    )
    if not figure_audit_path.is_file():
        raise FileNotFoundError(figure_audit_path)
    figure_audit = json.loads(
        figure_audit_path.read_text(encoding="utf-8")
    )
    if figure_audit.get("comparison_design") != (
        "trajectory_matched_skew3_and_skew4"
    ):
        raise ValueError("Final figures are not trajectory matched")
    if int(figure_audit.get("checkpoint_count", -1)) != EXPECTED_CHECKPOINTS:
        raise ValueError("Final figures do not average six checkpoints")
    entries = collect_package_entries(
        processed_dir,
        existing_experiment_dir,
        output_root,
    )
    return {
        "status": "complete",
        "comparison_design": "trajectory_matched_skew3_and_skew4",
        "checkpoint_count": EXPECTED_CHECKPOINTS,
        "skew3_ob1": skew3_ob1,
        "skew4_ob1": skew4_ob1,
        "skew3_analysis": skew3_analysis,
        "skew4_analysis": skew4_analysis,
        "figure_audit": figure_audit,
        "package_file_count": len(entries),
        "package_uncompressed_bytes": sum(
            path.stat().st_size for path in entries.values()
        ),
    }


def write_archive(
    entries: dict[str, Path],
    validation: dict,
    archive_path: Path,
) -> dict:
    """Write and verify one compressed result archive and checksum sidecar."""
    archive_path = archive_path.expanduser().resolve()
    if archive_path.exists():
        raise FileExistsError(
            f"Archive already exists: {archive_path}. Pass a new "
            "--archive-path."
        )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_records = [
        {
            "archive_path": archive_name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for archive_name, source in sorted(entries.items())
    ]
    package_manifest = {
        "package": "trajectory_matched_skew3_skew4_verification",
        "validation": validation,
        "files": file_records,
    }
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for record in file_records:
            archive.write(
                entries[record["archive_path"]],
                arcname=record["archive_path"],
            )
        archive.writestr(
            "PACKAGE_MANIFEST.json",
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        )
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(
                f"ZIP integrity check failed at {corrupt_member}"
            )
        expected_names = {
            *entries,
            "PACKAGE_MANIFEST.json",
        }
        if set(archive.namelist()) != expected_names:
            raise RuntimeError("ZIP member list differs from package plan")
    archive_sha256 = sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(
        archive_path.suffix + ".sha256"
    )
    checksum_path.write_text(
        f"{archive_sha256}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return {
        **validation,
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "checksum_path": str(checksum_path),
        "zip_integrity": "passed",
        "zip_member_count": len(entries) + 1,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the completion-check and package interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the completed skew=3/skew=4 trajectory-matched "
            "experiment and optionally create a minimal verification ZIP."
        )
    )
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
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate completion without writing a ZIP archive.",
    )
    return parser


def main() -> None:
    """Validate the experiment, package it when requested, and print JSON."""
    args = build_parser().parse_args()
    processed_dir = args.processed_dir.expanduser().resolve()
    existing_experiment_dir = (
        args.existing_experiment_dir.expanduser().resolve()
    )
    output_root = args.output_root.expanduser().resolve()
    validation = validate_completed_experiment(
        processed_dir,
        existing_experiment_dir,
        output_root,
    )
    if args.check_only:
        result = validation
    else:
        entries = collect_package_entries(
            processed_dir,
            existing_experiment_dir,
            output_root,
        )
        result = write_archive(
            entries,
            validation,
            args.archive_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

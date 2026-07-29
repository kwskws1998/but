"""Build a report using only the newly supplied Provo simulation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
DEFAULT_ARCHIVE = Path(
    "/Users/wansookim/Downloads/provo_attention_profile_focused.zip"
)
DEFAULT_LOG = Path("/Users/wansookim/Downloads/provo_analysis.log")
DEFAULT_TERMINAL_LOG = Path(
    "/Users/wansookim/.codex/attachments/"
    "7ae92303-7f45-4959-adb1-170f55c236d9/pasted-text.txt"
)
DEFAULT_OUTPUT_DIR = (
    COGNITIVE_ROOT / "outputs" / "provo_current_run_only_20260729"
)
PRIMARY_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "learned_asymmetric",
)
SPREAD_MATCHED_METHODS = (
    "raw_delta",
    "support_rms_displacement_symmetric",
    "learned_asymmetric",
)
INTEGRATED_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "support_rms_displacement_symmetric",
    "learned_asymmetric",
)
METHOD_LABELS = {
    "raw_delta": "Baseline: no redistribution",
    "fixed_symmetric_sigma1": "Fixed symmetric sigma=1",
    "support_rms_displacement_symmetric": (
        "Spread-matched symmetric sigma=3.865763"
    ),
    "learned_asymmetric": "Learned asymmetric",
}
METHOD_COLORS = {
    "raw_delta": "#4C78A8",
    "fixed_symmetric_sigma1": "#59A14F",
    "support_rms_displacement_symmetric": "#F28E2B",
    "learned_asymmetric": "#E45756",
}
METRICS = (
    ("profile_spearman", "Spearman", True),
    ("js_divergence", "JS divergence", False),
    ("hellinger_distance", "Hellinger", False),
    ("total_variation_distance", "Total variation", False),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_objects(text: str) -> list[dict]:
    """Extract unique JSON objects embedded in copied terminal output."""
    decoder = json.JSONDecoder()
    objects = []
    seen = set()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = start + length
        if not isinstance(value, dict):
            continue
        signature = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            objects.append(value)
    return objects


def load_terminal_provenance(path: Path) -> tuple[dict, dict]:
    """Load the OB1 aggregation and attention summaries from terminal output."""
    objects = parse_json_objects(path.read_text(encoding="utf-8"))
    simulation_records = [
        record
        for record in objects
        if "virtual_readers" in record and "per_simulation_word_rows" in record
    ]
    attention_records = [
        record
        for record in objects
        if "actual_et1_trt_magnitudes_used" in record
        and "active_profile_methods" in record
    ]
    if len(simulation_records) != 1 or len(attention_records) != 1:
        raise ValueError(
            "Expected one unique simulation summary and one unique "
            "attention summary in the copied terminal output"
        )
    return simulation_records[0], attention_records[0]


def load_analysis_log(path: Path) -> dict:
    """Load the JSON result from the dedicated attention-analysis log."""
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise ValueError("Dedicated analysis log contains no JSON")
    value = json.loads(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Dedicated analysis log must contain one JSON object")
    return value


def load_archive(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load result, contrast, profile, directionality, and audit records."""
    with ZipFile(path) as archive:
        results = pd.read_csv(
            archive.open("kernel_alignment_result_table.csv")
        )
        contrasts = pd.read_csv(
            archive.open("kernel_alignment_contrasts.csv")
        )
        profiles = pd.read_csv(archive.open("kernel_profiles.csv"))
        directionality = pd.read_csv(
            archive.open("kernel_directionality.csv")
        )
        with archive.open("attention_profile_audit.json") as handle:
            audit = json.load(handle)
    return results, contrasts, profiles, directionality, audit


def validate_provenance(
    simulation: dict,
    terminal_attention: dict,
    dedicated_attention: dict,
    archive_audit: dict,
) -> None:
    """Require all newly supplied provenance records to agree."""
    if terminal_attention != dedicated_attention:
        raise ValueError(
            "Copied terminal attention JSON differs from the dedicated log"
        )
    if dedicated_attention != archive_audit:
        raise ValueError(
            "Dedicated attention log differs from the ZIP audit"
        )
    attention_fields = (
        "checkpoint_id",
        "learned_sigma_left",
        "learned_sigma_right",
        "source_accuracy",
        "simulation_count",
        "seed_count",
        "passage_count",
        "fixation_count",
        "profile_component",
        "actual_et1_trt_magnitudes_used",
        "et1_token_values_sha256",
        "ob1_fixations_sha256",
        "ob1_worker_manifest_sha256",
    )
    mismatches = {}
    for field in attention_fields:
        values = (
            terminal_attention.get(field),
            dedicated_attention.get(field),
            archive_audit.get(field),
        )
        if values[0] != values[1] or values[0] != values[2]:
            mismatches[field] = values
    if mismatches:
        raise ValueError(
            f"New attention artifacts disagree on provenance: {mismatches}"
        )
    simulation_checks = {
        "virtual_readers": (
            simulation.get("virtual_readers"),
            archive_audit.get("simulation_count"),
        ),
        "n_trials": (
            simulation.get("n_trials"),
            archive_audit.get("passage_count"),
        ),
        "fixations": (
            simulation.get("fixations"),
            archive_audit.get("fixation_count"),
        ),
        "seed_count": (
            len(simulation.get("seeds", [])),
            archive_audit.get("seed_count"),
        ),
    }
    failed = {
        name: values
        for name, values in simulation_checks.items()
        if values[0] != values[1]
    }
    if failed:
        raise ValueError(
            f"New simulation and attention summaries disagree: {failed}"
        )


def select_results(
    results: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    """Select and order the requested methods for both attention skews."""
    selected = results.loc[results["method"].isin(methods)].copy()
    selected["_method_order"] = selected["method"].map(
        {method: index for index, method in enumerate(methods)}
    )
    return (
        selected.sort_values(
            ["ob1_attention_skew", "_method_order"]
        )
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def select_contrasts(
    contrasts: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    """Select learned-versus-symmetric paired passage contrasts."""
    return contrasts.loc[
        contrasts["candidate"].eq("learned_asymmetric")
        & contrasts["baseline"].eq(baseline)
        & contrasts["metric"].isin([metric for metric, _, _ in METRICS])
    ].sort_values(["ob1_attention_skew", "metric"]).reset_index(drop=True)


def plot_metrics(
    table: pd.DataFrame,
    methods: tuple[str, ...],
    path: Path,
) -> None:
    """Plot all four metrics for trajectory and formula-sensitivity settings."""
    figure, axes = plt.subplots(
        2,
        len(METRICS),
        figsize=(16, 7),
        constrained_layout=True,
    )
    positions = np.arange(len(methods))
    for row_index, skew in enumerate((3.0, 4.0)):
        rows = (
            table.loc[table["ob1_attention_skew"].eq(skew)]
            .set_index("method")
            .loc[list(methods)]
        )
        for column_index, (metric, label, higher) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            values = rows[metric].to_numpy(dtype=float)
            lower = values - rows[
                f"{metric}_ci_low"
            ].to_numpy(dtype=float)
            upper = rows[
                f"{metric}_ci_high"
            ].to_numpy(dtype=float) - values
            axis.bar(
                positions,
                values,
                color=[METHOD_COLORS[method] for method in methods],
                yerr=np.vstack([lower, upper]),
                capsize=3,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [METHOD_LABELS[method] for method in methods],
                rotation=31,
                ha="right",
            )
            role = "trajectory-matched" if skew == 3.0 else "formula sensitivity"
            axis.set_title(
                f"skew={int(skew)} ({role})\n"
                f"{label} ({'higher' if higher else 'lower'} better)"
            )
            axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_profiles(profiles: pd.DataFrame, path: Path) -> None:
    """Plot the current-run OB1 and three redistribution profiles."""
    columns = (
        ("ob1_attention_profile", "OB1 focused attention", "#222222", 2.8),
        ("raw_delta", "Baseline: no redistribution", "#4C78A8", 1.8),
        (
            "fixed_symmetric_sigma1",
            "Fixed symmetric sigma=1",
            "#59A14F",
            2.0,
        ),
        (
            "support_rms_displacement_symmetric",
            "Spread-matched symmetric",
            "#F28E2B",
            2.0,
        ),
        (
            "learned_asymmetric",
            "Learned asymmetric",
            "#E45756",
            2.3,
        ),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
        sharey=True,
    )
    for axis, skew in zip(axes, (3.0, 4.0)):
        rows = profiles.loc[
            profiles["ob1_attention_skew"].eq(skew)
        ].sort_values("relative_t5_token_offset")
        offsets = rows["relative_t5_token_offset"].to_numpy(dtype=float)
        for column, label, color, width in columns:
            axis.plot(
                offsets,
                rows[column].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=width,
                color=color,
                label=label,
            )
        axis.axvline(0.0, color="#777777", linewidth=1)
        axis.set_xlim(-5, 10)
        axis.set_xlabel("Relative native T5-token offset")
        role = "trajectory-matched" if skew == 3.0 else "formula sensitivity"
        axis.set_title(f"skew={int(skew)} ({role})")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Normalized allocation mass")
    axes[1].legend(frameon=False, fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def format_estimate(row: pd.Series, metric: str) -> str:
    """Format one result estimate and percentile interval."""
    return (
        f"{row[metric]:.4f} "
        f"[{row[f'{metric}_ci_low']:.4f}, "
        f"{row[f'{metric}_ci_high']:.4f}]"
    )


def result_markdown(
    table: pd.DataFrame,
    methods: tuple[str, ...],
    skew: float,
) -> list[str]:
    """Render one integrated condition-comparison table."""
    role = "trajectory-matched" if skew == 3.0 else "formula sensitivity"
    lines = [
        f"### OB1 skew={int(skew)} ({role})",
        "",
        "| Condition | Spearman | JS | Hellinger | TV |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = (
        table.loc[table["ob1_attention_skew"].eq(skew)]
        .set_index("method")
        .loc[list(methods)]
    )
    for method, row in rows.iterrows():
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_estimate(row, 'profile_spearman')} | "
            f"{format_estimate(row, 'js_divergence')} | "
            f"{format_estimate(row, 'hellinger_distance')} | "
            f"{format_estimate(row, 'total_variation_distance')} |"
        )
    return lines


def contrast_markdown(
    contrasts: pd.DataFrame,
) -> list[str]:
    """Render learned-versus-symmetric paired improvements."""
    lines = [
        "Positive values favor learned asymmetric.",
        "",
        "| Setting | Spearman | JS | Hellinger | TV |",
        "|---|---:|---:|---:|---:|",
    ]
    for skew in (3.0, 4.0):
        rows = contrasts.loc[
            contrasts["ob1_attention_skew"].eq(skew)
        ].set_index("metric")
        cells = []
        for metric, _, _ in METRICS:
            row = rows.loc[metric]
            cells.append(
                f"{row['mean_paired_improvement']:.4f} "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
        lines.append(
            f"| skew={int(skew)} | " + " | ".join(cells) + " |"
        )
    return lines


def write_markdown(
    output_dir: Path,
    integrated: pd.DataFrame,
    primary_contrasts: pd.DataFrame,
    spread_contrasts: pd.DataFrame,
    audit: dict,
) -> None:
    """Write the current-run-only tables and interpretation."""
    lines = [
        "# Current Provo simulation: integrated OB1 comparison",
        "",
        "This report uses only the newly supplied terminal log, dedicated "
        "analysis log, and attention-profile ZIP.",
        "",
        (
            f"Run: {audit['simulation_count']} simulated readers, "
            f"{audit['passage_count']} passages, "
            f"{audit['fixation_count']:,} fixations; "
            f"sigma_left={audit['learned_sigma_left']}, "
            f"sigma_right={audit['learned_sigma_right']}."
        ),
        "",
        "All four conditions use the same unit source so that only the "
        "redistribution rule changes.",
        "",
        "Metric columns are Spearman rank correlation, Jensen-Shannon "
        "divergence, Hellinger distance, and total-variation distance. "
        "Spearman is higher-is-better; all three distances are "
        "lower-is-better.",
        "",
        "## Metric mapping",
        "",
        "Each table cell is the mean of 55 passage-level comparisons against "
        "the OB1 fixation-onset focused-attention reference, followed by a "
        "95% percentile passage-bootstrap interval in brackets.",
        "",
        "- Spearman: rank correlation across relative T5-token offsets; "
        "higher is better.",
        "- JS: base-2 Jensen-Shannon divergence between the two normalized "
        "allocation profiles; lower is better.",
        "- Hellinger: bounded distance between the square roots of the two "
        "normalized profiles; lower is better.",
        "- TV: half the summed absolute mass difference between the two "
        "normalized profiles; lower is better. Its complement, 1 - TV, is "
        "the overlap coefficient.",
        "",
        "## Integrated comparison",
        "",
    ]
    for skew in (3.0, 4.0):
        lines.extend(result_markdown(integrated, INTEGRATED_METHODS, skew))
        lines.append("")
    lines.extend(
        [
            "## Paired contrasts",
            "",
            "### Learned asymmetric versus fixed symmetric sigma=1",
            "",
            *contrast_markdown(primary_contrasts),
            "",
            (
                "The learned kernel's realized RMS displacement on the exact "
                "fixation-visible supports is "
                f"{audit['learned_support_rms_token_displacement']:.6f} "
                "tokens. The matched symmetric sigma is "
                f"{audit['support_rms_displacement_symmetric_sigma']:.6f}."
            ),
            "",
            "### Learned asymmetric versus spread-matched symmetric",
            "",
            *contrast_markdown(spread_contrasts),
            "",
            "## Interpretation",
            "",
            "- Both Gaussian rules correspond much more strongly with OB1 "
            "than no redistribution.",
            "- Against fixed symmetric sigma=1, learned asymmetry is not "
            "uniformly better.",
            "- After matching realized spread, learned asymmetry improves "
            "complete allocation-shape distances but not Spearman ranking.",
            "- The split occurs because Spearman evaluates ordering, whereas "
            "JS, Hellinger, and TV evaluate the amount of normalized mass.",
            "- This is a current-run redistribution-kernel comparison with "
            "OB1 focused attention, not a comparison with older behavior "
            "outputs.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_report(
    archive_path: Path,
    log_path: Path,
    terminal_log_path: Path,
    output_dir: Path,
) -> dict:
    """Build the complete current-run-only report and provenance audit."""
    archive_path = archive_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    terminal_log_path = terminal_log_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (archive_path, log_path, terminal_log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    simulation, terminal_attention = load_terminal_provenance(
        terminal_log_path
    )
    dedicated_attention = load_analysis_log(log_path)
    results, contrasts, profiles, directionality, archive_audit = (
        load_archive(archive_path)
    )
    validate_provenance(
        simulation,
        terminal_attention,
        dedicated_attention,
        archive_audit,
    )
    primary = select_results(results, PRIMARY_METHODS)
    integrated = select_results(results, INTEGRATED_METHODS)
    primary_contrasts = select_contrasts(
        contrasts,
        "fixed_symmetric_sigma1",
    )
    spread_matched = select_results(results, SPREAD_MATCHED_METHODS)
    spread_contrasts = select_contrasts(
        contrasts,
        "support_rms_displacement_symmetric",
    )
    selected_directionality = directionality.loc[
        directionality["method"].isin(
            {
                "ob1_attention_profile",
                *PRIMARY_METHODS,
                "support_rms_displacement_symmetric",
            }
        )
    ].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    integrated.to_csv(
        output_dir / "integrated_results.csv",
        index=False,
    )
    primary.to_csv(
        output_dir / "three_condition_results.csv",
        index=False,
    )
    primary_contrasts.to_csv(
        output_dir / "three_condition_learned_vs_symmetric.csv",
        index=False,
    )
    spread_matched.to_csv(
        output_dir / "spread_matched_results.csv",
        index=False,
    )
    spread_contrasts.to_csv(
        output_dir / "spread_matched_learned_vs_symmetric.csv",
        index=False,
    )
    selected_directionality.to_csv(
        output_dir / "directionality.csv",
        index=False,
    )
    profiles.to_csv(output_dir / "pooled_profiles.csv", index=False)
    plot_metrics(
        integrated,
        INTEGRATED_METHODS,
        output_dir / "integrated_metrics.png",
    )
    plot_metrics(
        primary,
        PRIMARY_METHODS,
        output_dir / "three_condition_metrics.png",
    )
    plot_metrics(
        spread_matched,
        SPREAD_MATCHED_METHODS,
        output_dir / "spread_matched_metrics.png",
    )
    plot_profiles(profiles, output_dir / "pooled_profiles.png")
    write_markdown(
        output_dir,
        integrated,
        primary_contrasts,
        spread_contrasts,
        archive_audit,
    )
    audit = {
        "analysis": "current_provo_simulation_only",
        "uses_prior_local_behavior_outputs": False,
        "inputs": {
            "attention_archive": {
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
            },
            "dedicated_analysis_log": {
                "path": str(log_path),
                "sha256": sha256_file(log_path),
            },
            "copied_terminal_log": {
                "path": str(terminal_log_path),
                "sha256": sha256_file(terminal_log_path),
            },
        },
        "provenance_validation": {
            "all_new_artifacts_agree": True,
            "attention_json_objects_exactly_equal": True,
            "simulation_count": archive_audit["simulation_count"],
            "seed_count": archive_audit["seed_count"],
            "passage_count": archive_audit["passage_count"],
            "fixation_count": archive_audit["fixation_count"],
            "checkpoint_id": archive_audit["checkpoint_id"],
            "et1_token_values_sha256": archive_audit[
                "et1_token_values_sha256"
            ],
            "ob1_fixations_sha256": archive_audit[
                "ob1_fixations_sha256"
            ],
            "ob1_worker_manifest_sha256": archive_audit[
                "ob1_worker_manifest_sha256"
            ],
        },
        "estimand": archive_audit["analysis_estimand"],
        "trajectory_attention_skew": archive_audit[
            "trajectory_attention_skew"
        ],
        "formula_sensitivity_skew": 4,
        "bootstrap_samples": archive_audit["bootstrap_samples"],
        "bootstrap_seed": archive_audit["bootstrap_seed"],
        "resampling_unit": archive_audit["ci_resampling_unit"],
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "simulation_count": audit["provenance_validation"][
            "simulation_count"
        ],
        "passage_count": audit["provenance_validation"]["passage_count"],
        "uses_prior_local_behavior_outputs": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an integrated report using only the newly supplied "
            "Provo simulation artifacts."
        )
    )
    parser.add_argument(
        "--attention-archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
    )
    parser.add_argument("--analysis-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--terminal-log",
        type=Path,
        default=DEFAULT_TERMINAL_LOG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    """Run the current-run-only report builder."""
    args = build_parser().parse_args()
    result = run_report(
        archive_path=args.attention_archive,
        log_path=args.analysis_log,
        terminal_log_path=args.terminal_log,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

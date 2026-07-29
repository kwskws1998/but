"""Build a compact three-condition Human Provo and OB1 comparison report."""

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
DEFAULT_BEHAVIOR_DIR = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_three_condition_reanalysis_20260729"
    / "verified_behavior_width_matched"
)
DEFAULT_ARCHIVE = Path(
    "/Users/wansookim/Downloads/provo_attention_profile_focused.zip"
)
DEFAULT_LOG = Path("/Users/wansookim/Downloads/provo_analysis.log")
DEFAULT_OUTPUT_DIR = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_three_condition_reanalysis_20260729"
    / "report"
)
METRICS = (
    ("spearman", "Spearman", True),
    ("js_divergence", "JS divergence", False),
    ("hellinger_distance", "Hellinger", False),
    ("total_variation_distance", "Total variation", False),
)
BEHAVIOR_TARGETS = (
    "human_trt_conditional",
    "human_trt_unconditional",
    "ob1_tvt",
)
TARGET_LABELS = {
    "human_trt_conditional": "Human conditional TRT",
    "human_trt_unconditional": "Human unconditional TRT",
    "ob1_tvt": "OB1 simulated TVT",
}
PRIMARY_BEHAVIOR_METHODS = (
    "et1_raw",
    "fixed_symmetric_sigma1",
    "learned_asymmetric",
)
WIDTH_BEHAVIOR_METHODS = (
    "et1_raw",
    "realized_spread_matched_symmetric",
    "learned_asymmetric",
)
PRIMARY_ATTENTION_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "learned_asymmetric",
)
WIDTH_ATTENTION_METHODS = (
    "raw_delta",
    "support_rms_displacement_symmetric",
    "learned_asymmetric",
)
METHOD_LABELS = {
    "et1_raw": "Raw ET1",
    "fixed_symmetric_sigma1": "Symmetric sigma=1",
    "realized_spread_matched_symmetric": "Spread-matched symmetric",
    "learned_asymmetric": "Learned asymmetric",
    "raw_delta": "Unit-source delta",
    "support_rms_displacement_symmetric": "Spread-matched symmetric",
}
METHOD_COLORS = {
    "et1_raw": "#4C78A8",
    "raw_delta": "#4C78A8",
    "fixed_symmetric_sigma1": "#59A14F",
    "realized_spread_matched_symmetric": "#F28E2B",
    "support_rms_displacement_symmetric": "#F28E2B",
    "learned_asymmetric": "#E45756",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_filter(
    table: pd.DataFrame,
    methods: tuple[str, ...],
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Filter methods and restore an explicit group and method order."""
    selected = table.loc[table["method"].isin(methods)].copy()
    selected["_method_order"] = selected["method"].map(
        {method: index for index, method in enumerate(methods)}
    )
    return (
        selected.sort_values([*group_columns, "_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )


def load_behavior_tables(
    behavior_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load verified behavior results, paired contrasts, and their audit."""
    result_path = behavior_dir / "result_table.csv"
    contrast_path = behavior_dir / "paired_contrasts.csv"
    audit_path = behavior_dir / "analysis_audit.json"
    for path in (result_path, contrast_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    results = pd.read_csv(result_path)
    contrasts = pd.read_csv(contrast_path)
    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if not audit.get("actual_passage_specific_et1_values_used"):
        raise ValueError("Behavior analysis did not use actual ET1 values")
    return results, contrasts, audit


def load_attention_tables(
    archive_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the focused-attention results and audit from the supplied ZIP."""
    with ZipFile(archive_path) as archive:
        results = pd.read_csv(
            archive.open("kernel_alignment_result_table.csv")
        )
        contrasts = pd.read_csv(
            archive.open("kernel_alignment_contrasts.csv")
        )
        profiles = pd.read_csv(archive.open("kernel_profiles.csv"))
        with archive.open("attention_profile_audit.json") as handle:
            audit = json.load(handle)
    if bool(audit.get("actual_et1_trt_magnitudes_used")):
        raise ValueError("Expected the supplied unit-source kernel analysis")
    return results, contrasts, profiles, audit


def load_analysis_log(log_path: Path) -> dict:
    """Parse and validate the JSON result printed in the supplied log."""
    text = log_path.read_text(encoding="utf-8")
    json_start = text.find("{")
    if json_start < 0:
        raise ValueError("Analysis log contains no JSON result")
    record = json.loads(text[json_start:])
    required = {
        "checkpoint_id",
        "learned_sigma_left",
        "learned_sigma_right",
        "source_accuracy",
        "simulation_count",
        "seed_count",
        "passage_count",
        "actual_et1_trt_magnitudes_used",
        "profile_component",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Analysis log is missing fields: {missing}")
    return record


def validate_log_archive_agreement(
    log_record: dict,
    archive_audit: dict,
) -> None:
    """Require key supplied-log settings to agree with the supplied archive."""
    fields = (
        "checkpoint_id",
        "learned_sigma_left",
        "learned_sigma_right",
        "source_accuracy",
        "simulation_count",
        "seed_count",
        "passage_count",
        "actual_et1_trt_magnitudes_used",
        "profile_component",
    )
    mismatches = {
        field: (log_record.get(field), archive_audit.get(field))
        for field in fields
        if log_record.get(field) != archive_audit.get(field)
    }
    if mismatches:
        raise ValueError(
            f"Analysis log and archive audit disagree: {mismatches}"
        )


def select_behavior_contrasts(
    contrasts: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    """Select learned-versus-symmetric paired behavior contrasts."""
    return contrasts.loc[
        contrasts["candidate"].eq("learned_asymmetric")
        & contrasts["baseline"].eq(baseline)
        & contrasts["metric"].isin([metric for metric, _, _ in METRICS])
    ].reset_index(drop=True)


def select_attention_contrasts(
    contrasts: pd.DataFrame,
    baseline: str,
) -> pd.DataFrame:
    """Select learned-versus-symmetric paired attention contrasts."""
    archive_metrics = {
        "profile_spearman",
        "js_divergence",
        "hellinger_distance",
        "total_variation_distance",
    }
    return contrasts.loc[
        contrasts["candidate"].eq("learned_asymmetric")
        & contrasts["baseline"].eq(baseline)
        & contrasts["metric"].isin(archive_metrics)
    ].reset_index(drop=True)


def plot_behavior_panel(
    table: pd.DataFrame,
    methods: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot every requested behavior metric for all three references."""
    figure, axes = plt.subplots(
        len(BEHAVIOR_TARGETS),
        len(METRICS),
        figsize=(16, 10),
        constrained_layout=True,
    )
    positions = np.arange(len(methods))
    for row_index, target in enumerate(BEHAVIOR_TARGETS):
        target_rows = (
            table.loc[table["target"].eq(target)]
            .set_index("method")
            .loc[list(methods)]
        )
        for column_index, (metric, metric_label, higher) in enumerate(
            METRICS
        ):
            axis = axes[row_index, column_index]
            values = target_rows[metric].to_numpy(dtype=float)
            lower = values - target_rows[
                f"{metric}_ci_low"
            ].to_numpy(dtype=float)
            upper = target_rows[
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
                rotation=32,
                ha="right",
            )
            axis.set_title(
                f"{TARGET_LABELS[target]}\n"
                f"{metric_label} ({'higher' if higher else 'lower'} better)"
            )
            axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_attention_panel(
    table: pd.DataFrame,
    methods: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot every focused-attention metric for skew 3 and skew 4."""
    figure, axes = plt.subplots(
        2,
        len(METRICS),
        figsize=(16, 7),
        constrained_layout=True,
    )
    positions = np.arange(len(methods))
    for row_index, skew in enumerate((3.0, 4.0)):
        skew_rows = (
            table.loc[table["ob1_attention_skew"].eq(skew)]
            .set_index("method")
            .loc[list(methods)]
        )
        for column_index, (metric, metric_label, higher) in enumerate(
            METRICS
        ):
            archive_metric = (
                "profile_spearman" if metric == "spearman" else metric
            )
            axis = axes[row_index, column_index]
            values = skew_rows[archive_metric].to_numpy(dtype=float)
            lower = values - skew_rows[
                f"{archive_metric}_ci_low"
            ].to_numpy(dtype=float)
            upper = skew_rows[
                f"{archive_metric}_ci_high"
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
                rotation=32,
                ha="right",
            )
            role = "trajectory-matched" if skew == 3.0 else "formula sensitivity"
            axis.set_title(
                f"OB1 {role}, skew={int(skew)}\n"
                f"{metric_label} ({'higher' if higher else 'lower'} better)"
            )
            axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_attention_profiles(
    profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot pooled OB1 and redistribution profiles on relative token offsets."""
    profile_columns = (
        ("ob1_attention_profile", "OB1 focused attention", "#222222", 2.8),
        ("raw_delta", "Unit-source delta", "#4C78A8", 1.8),
        (
            "fixed_symmetric_sigma1",
            "Symmetric sigma=1",
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
        selected = profiles.loc[
            profiles["ob1_attention_skew"].eq(skew)
        ].sort_values("relative_t5_token_offset")
        offsets = selected[
            "relative_t5_token_offset"
        ].to_numpy(dtype=float)
        for column, label, color, width in profile_columns:
            axis.plot(
                offsets,
                selected[column].to_numpy(dtype=float),
                marker="o",
                markersize=3,
                linewidth=width,
                color=color,
                label=label,
            )
        axis.axvline(0.0, color="#777777", linewidth=1.0)
        axis.set_xlim(-5, 10)
        axis.set_xlabel("Relative native T5-token offset")
        role = "trajectory-matched" if skew == 3.0 else "formula sensitivity"
        axis.set_title(f"OB1 skew={int(skew)} ({role})")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Normalized allocation mass")
    axes[1].legend(frameon=False, fontsize=9)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def format_metric_row(row: pd.Series, metric: str) -> str:
    """Format one estimate and interval for the Markdown report."""
    return (
        f"{row[metric]:.4f} "
        f"[{row[f'{metric}_ci_low']:.4f}, "
        f"{row[f'{metric}_ci_high']:.4f}]"
    )


def behavior_markdown_table(
    table: pd.DataFrame,
    target: str,
    methods: tuple[str, ...],
) -> list[str]:
    """Render one behavior target as a Markdown table."""
    lines = [
        f"### {TARGET_LABELS[target]}",
        "",
        "| Condition | Spearman | JS | Hellinger | TV |",
        "|---|---:|---:|---:|---:|",
    ]
    selected = (
        table.loc[table["target"].eq(target)]
        .set_index("method")
        .loc[list(methods)]
    )
    for method, row in selected.iterrows():
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_metric_row(row, 'spearman')} | "
            f"{format_metric_row(row, 'js_divergence')} | "
            f"{format_metric_row(row, 'hellinger_distance')} | "
            f"{format_metric_row(row, 'total_variation_distance')} |"
        )
    return lines


def attention_markdown_table(
    table: pd.DataFrame,
    skew: float,
    methods: tuple[str, ...],
) -> list[str]:
    """Render one OB1 attention setting as a Markdown table."""
    lines = [
        f"### OB1 attention skew={int(skew)}",
        "",
        "| Condition | Spearman | JS | Hellinger | TV |",
        "|---|---:|---:|---:|---:|",
    ]
    selected = (
        table.loc[table["ob1_attention_skew"].eq(skew)]
        .set_index("method")
        .loc[list(methods)]
    )
    for method, row in selected.iterrows():
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_metric_row(row, 'profile_spearman')} | "
            f"{format_metric_row(row, 'js_divergence')} | "
            f"{format_metric_row(row, 'hellinger_distance')} | "
            f"{format_metric_row(row, 'total_variation_distance')} |"
        )
    return lines


def contrast_markdown_table(
    contrasts: pd.DataFrame,
) -> list[str]:
    """Render paired learned-versus-symmetric behavior improvements."""
    lines = [
        "Positive values favor learned asymmetric for every metric.",
        "",
        "| Reference | Spearman | JS | Hellinger | TV |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in BEHAVIOR_TARGETS:
        selected = contrasts.loc[
            contrasts["target"].eq(target)
        ].set_index("metric")
        cells = []
        for metric, _, _ in METRICS:
            row = selected.loc[metric]
            cells.append(
                f"{row['mean_paired_improvement']:.4f} "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            )
        lines.append(
            f"| {TARGET_LABELS[target]} | " + " | ".join(cells) + " |"
        )
    return lines


def write_report(
    output_dir: Path,
    behavior_primary: pd.DataFrame,
    behavior_width: pd.DataFrame,
    behavior_primary_contrasts: pd.DataFrame,
    behavior_width_contrasts: pd.DataFrame,
    attention_primary: pd.DataFrame,
    attention_width: pd.DataFrame,
    behavior_audit: dict,
) -> None:
    """Write one report with requested and width-controlled panels."""
    matched_sigma = behavior_audit[
        "realized_spread_matched_symmetric_sigma"
    ]
    matched_rms = behavior_audit[
        "learned_et1_weighted_realized_rms_displacement"
    ]
    lines = [
        "# Provo three-condition comparison",
        "",
        "The behavior analysis uses actual passage-specific ET1 token values. "
        "The supplied ZIP is reported separately because its no-redistribution "
        "condition is a unit-source delta, not raw ET1.",
        "",
        "## Requested behavior comparison",
        "",
    ]
    for target in BEHAVIOR_TARGETS:
        lines.extend(
            behavior_markdown_table(
                behavior_primary,
                target,
                PRIMARY_BEHAVIOR_METHODS,
            )
        )
        lines.append("")
    lines.extend(
        [
            "### Learned asymmetric versus symmetric sigma=1",
            "",
            *contrast_markdown_table(behavior_primary_contrasts),
            "",
        ]
    )
    lines.extend(
        [
            "## Behavior spread-control comparison",
            "",
            (
                f"The learned kernel's realized ET1-weighted RMS displacement "
                f"is {matched_rms:.6f} T5 tokens. The symmetric sigma solved "
                f"on the same input geometry is {matched_sigma:.6f}. Human "
                "and OB1 target values were not used to solve this control."
            ),
            "",
        ]
    )
    for target in BEHAVIOR_TARGETS:
        lines.extend(
            behavior_markdown_table(
                behavior_width,
                target,
                WIDTH_BEHAVIOR_METHODS,
            )
        )
        lines.append("")
    lines.extend(
        [
            "### Learned asymmetric versus spread-matched symmetric",
            "",
            *contrast_markdown_table(behavior_width_contrasts),
            "",
        ]
    )
    lines.extend(
        [
            "## Supplied ZIP: internal OB1 attention comparison",
            "",
            "These tables compare unit-source redistribution kernels with the "
            "OB1 fixation-onset focused attention component. They do not use "
            "actual ET1 magnitudes or full OB1 reading behavior.",
            "",
        ]
    )
    for skew in (3.0, 4.0):
        lines.extend(
            attention_markdown_table(
                attention_primary,
                skew,
                PRIMARY_ATTENTION_METHODS,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Supplied ZIP: fixation-support spread control",
            "",
            "This symmetric control matches the learned kernel's realized RMS "
            "displacement on the ZIP's fixation-visible supports. It is "
            "specific to that internal-attention analysis.",
            "",
        ]
    )
    for skew in (3.0, 4.0):
        lines.extend(
            attention_markdown_table(
                attention_width,
                skew,
                WIDTH_ATTENTION_METHODS,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- Raw ET1 retains the strongest word-rank correspondence.",
            "- Gaussian spreading can improve normalized allocation-shape "
            "distance relative to raw ET1.",
            "- Learned asymmetry is not uniformly better than symmetric "
            "redistribution.",
            "- After exact behavior-level spread matching, learned asymmetry "
            "improves rank correspondence but complete-shape evidence remains "
            "mixed.",
            "- These results support limited external consistency, not "
            "recovery of a human perceptual span or overall cognitive validity.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_report(
    behavior_dir: Path,
    archive_path: Path,
    log_path: Path,
    output_dir: Path,
) -> dict:
    """Build all compact tables, plots, and provenance records."""
    behavior_dir = behavior_dir.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    log_path = log_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (archive_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    behavior, behavior_contrasts, behavior_audit = load_behavior_tables(
        behavior_dir
    )
    attention, attention_contrasts, attention_profiles, attention_audit = (
        load_attention_tables(archive_path)
    )
    log_record = load_analysis_log(log_path)
    validate_log_archive_agreement(log_record, attention_audit)
    behavior_primary = ordered_filter(
        behavior,
        PRIMARY_BEHAVIOR_METHODS,
        ("target",),
    )
    behavior_width = ordered_filter(
        behavior,
        WIDTH_BEHAVIOR_METHODS,
        ("target",),
    )
    attention_primary = ordered_filter(
        attention,
        PRIMARY_ATTENTION_METHODS,
        ("ob1_attention_skew",),
    )
    attention_width = ordered_filter(
        attention,
        WIDTH_ATTENTION_METHODS,
        ("ob1_attention_skew",),
    )
    behavior_primary_contrasts = select_behavior_contrasts(
        behavior_contrasts,
        "fixed_symmetric_sigma1",
    )
    behavior_width_contrasts = select_behavior_contrasts(
        behavior_contrasts,
        "realized_spread_matched_symmetric",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "behavior_requested": output_dir
        / "behavior_requested_three_conditions.csv",
        "behavior_requested_contrasts": output_dir
        / "behavior_requested_learned_vs_symmetric.csv",
        "behavior_width": output_dir
        / "behavior_spread_matched_three_conditions.csv",
        "behavior_width_contrasts": output_dir
        / "behavior_spread_matched_learned_vs_symmetric.csv",
        "attention_requested": output_dir
        / "ob1_attention_requested_three_conditions.csv",
        "attention_requested_contrasts": output_dir
        / "ob1_attention_requested_learned_vs_symmetric.csv",
        "attention_width": output_dir
        / "ob1_attention_spread_matched_three_conditions.csv",
        "attention_width_contrasts": output_dir
        / "ob1_attention_spread_matched_learned_vs_symmetric.csv",
    }
    behavior_primary.to_csv(outputs["behavior_requested"], index=False)
    behavior_primary_contrasts.to_csv(
        outputs["behavior_requested_contrasts"],
        index=False,
    )
    behavior_width.to_csv(outputs["behavior_width"], index=False)
    behavior_width_contrasts.to_csv(
        outputs["behavior_width_contrasts"],
        index=False,
    )
    attention_primary.to_csv(outputs["attention_requested"], index=False)
    select_attention_contrasts(
        attention_contrasts,
        "fixed_symmetric_sigma1",
    ).to_csv(outputs["attention_requested_contrasts"], index=False)
    attention_width.to_csv(outputs["attention_width"], index=False)
    select_attention_contrasts(
        attention_contrasts,
        "support_rms_displacement_symmetric",
    ).to_csv(outputs["attention_width_contrasts"], index=False)
    plot_behavior_panel(
        behavior_primary,
        PRIMARY_BEHAVIOR_METHODS,
        output_dir / "behavior_requested_metrics.png",
    )
    plot_behavior_panel(
        behavior_width,
        WIDTH_BEHAVIOR_METHODS,
        output_dir / "behavior_spread_matched_metrics.png",
    )
    plot_attention_panel(
        attention_primary,
        PRIMARY_ATTENTION_METHODS,
        output_dir / "ob1_attention_requested_metrics.png",
    )
    plot_attention_panel(
        attention_width,
        WIDTH_ATTENTION_METHODS,
        output_dir / "ob1_attention_spread_matched_metrics.png",
    )
    plot_attention_profiles(
        attention_profiles,
        output_dir / "ob1_attention_profiles.png",
    )
    write_report(
        output_dir,
        behavior_primary,
        behavior_width,
        behavior_primary_contrasts,
        behavior_width_contrasts,
        attention_primary,
        attention_width,
        behavior_audit,
    )
    audit = {
        "analysis": "provo_three_condition_behavior_and_attention_report",
        "behavior_uses_actual_et1": True,
        "attention_archive_uses_actual_et1": False,
        "behavior_references": [
            "human_trt_conditional",
            "human_trt_unconditional",
            "ob1_tvt",
        ],
        "behavior_passages": behavior_audit["passages"],
        "behavior_words": behavior_audit["word_rows"],
        "behavior_spread_match": {
            "learned_realized_rms": behavior_audit[
                "learned_et1_weighted_realized_rms_displacement"
            ],
            "symmetric_sigma": behavior_audit[
                "realized_spread_matched_symmetric_sigma"
            ],
            "uses_human_or_ob1_target": behavior_audit[
                "realized_spread_match_uses_human_or_ob1_target"
            ],
            "scope": behavior_audit["realized_spread_match_scope"],
        },
        "attention_archive_estimand": attention_audit.get(
            "analysis_estimand"
        ),
        "supplied_log_validation": {
            "checkpoint_id": log_record["checkpoint_id"],
            "simulation_count": log_record["simulation_count"],
            "seed_count": log_record["seed_count"],
            "passage_count": log_record["passage_count"],
            "profile_component": log_record["profile_component"],
            "actual_et1_trt_magnitudes_used": log_record[
                "actual_et1_trt_magnitudes_used"
            ],
            "log_and_archive_key_settings_match": True,
        },
        "inputs": {
            "behavior_result_table": {
                "path": str(behavior_dir / "result_table.csv"),
                "sha256": sha256_file(behavior_dir / "result_table.csv"),
            },
            "attention_archive": {
                "path": str(archive_path),
                "sha256": sha256_file(archive_path),
            },
            "analysis_log": {
                "path": str(log_path),
                "sha256": sha256_file(log_path),
            },
        },
        "outputs": {
            name: str(path) for name, path in outputs.items()
        },
    }
    audit_path = output_dir / "report_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "behavior_rows_requested": int(len(behavior_primary)),
        "behavior_rows_width_matched": int(len(behavior_width)),
        "attention_rows_requested": int(len(attention_primary)),
        "attention_rows_width_matched": int(len(attention_width)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Build compact raw, symmetric, and asymmetric comparisons "
            "against Human Provo, OB1 TVT, and OB1 focused attention."
        )
    )
    parser.add_argument(
        "--behavior-dir",
        type=Path,
        default=DEFAULT_BEHAVIOR_DIR,
    )
    parser.add_argument(
        "--attention-archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
    )
    parser.add_argument("--analysis-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    """Run the report builder."""
    args = build_parser().parse_args()
    result = run_report(
        behavior_dir=args.behavior_dir,
        archive_path=args.attention_archive,
        log_path=args.analysis_log,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

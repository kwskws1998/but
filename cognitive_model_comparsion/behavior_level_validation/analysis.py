"""Build reviewer-facing tables from behavior-level Provo evaluation outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = (
    "et1_raw",
    "et1_symmetric",
    "et1_asymmetric",
    "ob1",
)
METHOD_LABELS = {
    "et1_raw": "ET1 raw",
    "et1_symmetric": "ET1 + fixed SymGaussian (sigma=1.0)",
    "et1_asymmetric": "ET1 + learned asymmetric",
    "ob1": "OB1 simulated TVT",
}
SUMMARY_METRICS = {
    "human_spearman": "human_spearman",
    "js_divergence": "human_js_divergence",
    "ob1_spearman": "ob1_spearman",
    "ob1_js_divergence": "ob1_js_divergence",
}
CONTRAST_PAIRS = (
    ("et1_symmetric", "et1_raw"),
    ("et1_asymmetric", "et1_raw"),
    ("et1_asymmetric", "et1_symmetric"),
)
REFERENCE_LABELS = {
    "human_spearman": "Human Provo conditional TRT",
    "human_js_divergence": "Human Provo conditional TRT",
    "ob1_spearman": "OB1 simulated TVT",
    "ob1_js_divergence": "OB1 simulated TVT",
}
METRIC_LABELS = {
    "human_spearman": "Human Spearman",
    "human_js_divergence": "Human JS divergence",
    "ob1_spearman": "OB1 Spearman",
    "ob1_js_divergence": "OB1 JS divergence",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_outputs(
    evaluation_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and validate one completed conditional behavior evaluation."""
    evaluation_dir = evaluation_dir.expanduser().resolve()
    paths = {
        "summary": evaluation_dir / "result_table.csv",
        "contrasts": evaluation_dir / "bootstrap_summary.csv",
        "audit": evaluation_dir / "evaluation_audit.json",
        "words": evaluation_dir / "word_level_values.csv",
        "passages": evaluation_dir / "passage_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Behavior evaluation is incomplete; missing files: "
            f"{missing}"
        )

    summary = pd.read_csv(paths["summary"])
    contrasts = pd.read_csv(paths["contrasts"])
    with paths["audit"].open(encoding="utf-8") as handle:
        audit = json.load(handle)

    if audit.get("human_target") != "human_trt_conditional":
        raise ValueError(
            "Paper-aligned behavior validation requires "
            "human_trt_conditional"
        )
    required_methods = set(METHOD_ORDER)
    available_methods = set(summary.get("method", pd.Series(dtype=str)))
    missing_methods = sorted(required_methods - available_methods)
    if missing_methods:
        raise ValueError(
            f"Behavior result table is missing methods: {missing_methods}"
        )
    return summary, contrasts, audit


def build_behavior_result_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Select Human- and OB1-referenced scale-free behavior metrics."""
    required = {"method", "passages"}
    for source in SUMMARY_METRICS:
        required.update(
            {
                source,
                f"{source}_ci_low",
                f"{source}_ci_high",
            }
        )
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(
            f"Behavior summary is missing columns: {missing}"
        )

    selected = summary.loc[
        summary["method"].isin(METHOD_ORDER)
    ].copy()
    if selected["method"].duplicated().any():
        duplicates = selected.loc[
            selected["method"].duplicated(keep=False),
            "method",
        ].tolist()
        raise ValueError(
            f"Behavior summary contains duplicate methods: {duplicates}"
        )
    selected = selected.set_index("method").loc[list(METHOD_ORDER)].reset_index()
    selected["display_name"] = selected["method"].map(METHOD_LABELS)

    output_columns = ["method", "display_name", "passages"]
    rename_map = {}
    for source, target in SUMMARY_METRICS.items():
        output_columns.extend(
            [
                source,
                f"{source}_ci_low",
                f"{source}_ci_high",
            ]
        )
        rename_map.update(
            {
                source: target,
                f"{source}_ci_low": f"{target}_ci_low",
                f"{source}_ci_high": f"{target}_ci_high",
            }
        )
    return selected[output_columns].rename(columns=rename_map)


def build_behavior_contrast_table(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Select paired method contrasts without reviewer-facing p-values."""
    required = {
        "candidate",
        "baseline",
        "metric",
        "passages",
        "mean_paired_improvement",
        "ci_low",
        "ci_high",
    }
    missing = sorted(required - set(contrasts.columns))
    if missing:
        raise ValueError(
            f"Behavior contrasts are missing columns: {missing}"
        )

    metric_renames = {
        "human_spearman": "human_spearman",
        "js_divergence": "human_js_divergence",
        "ob1_spearman": "ob1_spearman",
        "ob1_js_divergence": "ob1_js_divergence",
    }
    pair_index = {
        pair: index for index, pair in enumerate(CONTRAST_PAIRS)
    }
    selected = contrasts.loc[
        contrasts["metric"].isin(metric_renames)
        & contrasts.apply(
            lambda row: (row["candidate"], row["baseline"]) in pair_index,
            axis=1,
        )
    ].copy()
    if selected.empty:
        raise ValueError("No requested behavior contrasts were found")
    selected["metric"] = selected["metric"].map(metric_renames)
    selected["candidate_name"] = selected["candidate"].map(METHOD_LABELS)
    selected["baseline_name"] = selected["baseline"].map(METHOD_LABELS)
    selected["reference"] = selected["metric"].map(REFERENCE_LABELS)
    selected["metric_label"] = selected["metric"].map(METRIC_LABELS)
    selected["positive_means_improvement"] = True
    selected["_pair_order"] = selected.apply(
        lambda row: pair_index[(row["candidate"], row["baseline"])],
        axis=1,
    )
    metric_order = {
        metric: index for index, metric in enumerate(SUMMARY_METRICS.values())
    }
    selected["_metric_order"] = selected["metric"].map(metric_order)
    selected = selected.sort_values(
        ["_pair_order", "_metric_order"]
    ).reset_index(drop=True)
    return selected[
        [
            "candidate",
            "candidate_name",
            "baseline",
            "baseline_name",
            "reference",
            "metric",
            "metric_label",
            "positive_means_improvement",
            "passages",
            "mean_paired_improvement",
            "ci_low",
            "ci_high",
        ]
    ]


def format_estimate(
    row: pd.Series,
    metric: str,
    digits: int = 4,
) -> str:
    """Format one point estimate and percentile confidence interval."""
    value = float(row[metric])
    lower = float(row[f"{metric}_ci_low"])
    upper = float(row[f"{metric}_ci_high"])
    return (
        f"{value:.{digits}f} "
        f"[{lower:.{digits}f}, {upper:.{digits}f}]"
    )


def markdown_result_report(
    results: pd.DataFrame,
    audit: dict,
) -> str:
    """Render compact Human and OB1 behavior tables with scope warnings."""
    selected_ids = audit.get("selected_checkpoint_ids", [])
    lines = [
        "# Behavior-level Provo validation",
        "",
        (
            "Human target: participant-averaged conditional word TRT "
            "(zero-dwell reader-word observations excluded)."
        ),
    ]
    if len(selected_ids) == 1:
        lines.extend(["", f"ET1 checkpoint: `{selected_ids[0]}`."])
    metadata = audit.get("checkpoint_metadata", [])
    if len(metadata) == 1:
        record = metadata[0]
        if {"sigma_left", "sigma_right"}.issubset(record):
            lines.extend(
                [
                    "",
                    (
                        "Learned redistribution: "
                        f"sigma_left={float(record['sigma_left']):.5g}, "
                        f"sigma_right={float(record['sigma_right']):.5g}."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "## Human Provo correspondence",
            "",
            "| Method | Spearman (higher) | JS divergence (lower) |",
            "|---|---:|---:|",
        ]
    )
    for row in results.itertuples(index=False):
        series = pd.Series(row._asdict())
        lines.append(
            f"| {series['display_name']} | "
            f"{format_estimate(series, 'human_spearman')} | "
            f"{format_estimate(series, 'human_js_divergence')} |"
        )

    lines.extend(
        [
            "",
            "## OB1 simulated-TVT correspondence",
            "",
            "| Method | Spearman (higher) | JS divergence (lower) |",
            "|---|---:|---:|",
        ]
    )
    for row in results.itertuples(index=False):
        series = pd.Series(row._asdict())
        if series["method"] == "ob1":
            spearman = "-"
            js_divergence = "-"
        else:
            spearman = format_estimate(series, "ob1_spearman")
            js_divergence = format_estimate(
                series,
                "ob1_js_divergence",
            )
        lines.append(
            f"| {series['display_name']} | {spearman} | "
            f"{js_divergence} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "- This analysis uses actual passage-specific ET1 predictions "
                "and word-level OB1 TVT, not unit-impulse kernel profiles."
            ),
            (
                "- Spearman compares word-rank correspondence. JS divergence "
                "compares passage-normalized allocation shapes."
            ),
            (
                "- Raw millisecond RMSE is not reported because ET1 predicts "
                "normalized, discretized TRT targets rather than milliseconds."
            ),
            (
                "- The intervals resample Provo passages after OB1 simulations "
                "have been pooled."
            ),
            "",
            (
                f"Passages: {audit.get('passages', 'unknown')}; "
                f"bootstrap samples: "
                f"{audit.get('bootstrap_samples', 'unknown')}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_behavior_report(
    evaluation_dir: Path,
    output_dir: Path,
) -> dict:
    """Write compact behavior tables, report text, and an audit manifest."""
    evaluation_dir = evaluation_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    summary, contrasts, evaluation_audit = load_evaluation_outputs(
        evaluation_dir
    )
    results = build_behavior_result_table(summary)
    contrast_table = build_behavior_contrast_table(contrasts)
    selected_ids = evaluation_audit.get("selected_checkpoint_ids", [])
    if len(selected_ids) == 1:
        results.insert(0, "checkpoint_id", selected_ids[0])
        contrast_table.insert(0, "checkpoint_id", selected_ids[0])
    metadata = evaluation_audit.get("checkpoint_metadata", [])
    if len(metadata) == 1:
        record = metadata[0]
        insert_at = 1 if "checkpoint_id" in results else 0
        for column in ("source_accuracy", "sigma_left", "sigma_right"):
            if column in record:
                results.insert(insert_at, column, record[column])
                contrast_table.insert(insert_at, column, record[column])
                insert_at += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "behavior_result_table.csv"
    contrast_path = output_dir / "behavior_paired_contrasts.csv"
    report_path = output_dir / "RESULTS.md"
    audit_path = output_dir / "behavior_analysis_audit.json"
    results.to_csv(result_path, index=False)
    contrast_table.to_csv(contrast_path, index=False)
    report_path.write_text(
        markdown_result_report(results, evaluation_audit),
        encoding="utf-8",
    )

    source_paths = {
        name: evaluation_dir / filename
        for name, filename in {
            "result_table": "result_table.csv",
            "bootstrap_summary": "bootstrap_summary.csv",
            "word_level_values": "word_level_values.csv",
            "passage_metrics": "passage_metrics.csv",
            "evaluation_audit": "evaluation_audit.json",
        }.items()
    }
    report_audit = {
        "analysis": "behavior_level_provo_validation",
        "human_target": "human_trt_conditional",
        "actual_passage_specific_et1_values_used": True,
        "unit_impulse_kernel_profiles_used": False,
        "ob1_target": "word_level_total_viewing_time",
        "raw_millisecond_rmse_computed": False,
        "raw_millisecond_rmse_exclusion_reason": (
            "ET1 predicts corpus-normalized, quantile-discretized TRT "
            "targets rather than millisecond TRT."
        ),
        "primary_metrics": [
            "passage_level_spearman",
            "passage_normalized_jensen_shannon_divergence",
        ],
        "resampling_unit": "passage",
        "source_evaluation_dir": str(evaluation_dir),
        "source_sha256": {
            name: sha256_file(path)
            for name, path in source_paths.items()
        },
        "evaluation_audit": evaluation_audit,
        "outputs": {
            "result_table": str(result_path),
            "paired_contrasts": str(contrast_path),
            "markdown_report": str(report_path),
        },
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(report_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report_audit

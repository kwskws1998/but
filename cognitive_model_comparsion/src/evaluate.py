"""Evaluate Human, ET1 redistribution, and OB1 word-level allocation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance


METHOD_COLUMNS = {
    "et1_raw": "et1_raw_word_trt",
    "et1_symmetric": "et1_symmetric_word_trt",
    "et1_asymmetric": "et1_asymmetric_word_trt",
    "ob1": "ob1_tvt",
}
DISPLAY_NAMES = {
    "et1_raw": "ET1 raw",
    "et1_symmetric": "ET1 + symmetric",
    "et1_asymmetric": "ET1 + learned asymmetric",
    "ob1": "OB1 baseline",
}


def normalized_allocation(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip negative values and normalize a passage to unit allocation mass."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Allocation contains non-finite values")
    clipped_count = int((values < 0).sum())
    nonnegative = np.clip(values, 0.0, None)
    mass = float(nonnegative.sum())
    if mass <= 0:
        raise ValueError("Allocation has zero nonnegative passage mass")
    return nonnegative / mass, clipped_count


def passage_metric_values(
    human: np.ndarray,
    method: np.ndarray,
    ob1: np.ndarray,
) -> dict:
    """Compute rank and distribution metrics for one passage."""
    human = np.asarray(human, dtype=float)
    method = np.asarray(method, dtype=float)
    ob1 = np.asarray(ob1, dtype=float)
    if not (len(human) == len(method) == len(ob1)):
        raise ValueError("Human, method, and OB1 passage lengths differ")
    if len(human) < 2:
        raise ValueError("At least two passage words are required")
    if not (
        np.isfinite(human).all()
        and np.isfinite(method).all()
        and np.isfinite(ob1).all()
    ):
        raise ValueError("Passage metric input contains non-finite values")

    human_distribution, human_clipped = normalized_allocation(human)
    method_distribution, method_clipped = normalized_allocation(method)
    ob1_distribution, ob1_clipped = normalized_allocation(ob1)
    positions = np.linspace(0.0, 1.0, len(human))
    human_spearman = float(spearmanr(human, method).statistic)
    ob1_spearman = float(spearmanr(ob1, method).statistic)
    if not np.isfinite(human_spearman) or not np.isfinite(ob1_spearman):
        raise ValueError("Spearman correlation is undefined for a passage")
    return {
        "human_spearman": human_spearman,
        "js_divergence": float(
            jensenshannon(
                human_distribution,
                method_distribution,
                base=2.0,
            )
            ** 2
        ),
        "wasserstein": float(
            wasserstein_distance(
                positions,
                positions,
                u_weights=human_distribution,
                v_weights=method_distribution,
            )
        ),
        "ob1_spearman": ob1_spearman,
        "human_clipped_values": human_clipped,
        "method_clipped_values": method_clipped,
        "ob1_clipped_values": ob1_clipped,
    }


def merge_word_values(
    canonical_words: pd.DataFrame,
    et1_words: pd.DataFrame,
    ob1_words: pd.DataFrame,
) -> pd.DataFrame:
    """Join canonical Human, checkpoint-specific ET1, and OB1 word grids."""
    keys = ["passage_id_zero_based", "word_id_zero_based"]
    canonical_columns = keys + [
        "passage_id_raw",
        "word_raw",
        "human_trt_unconditional",
        "human_trt_conditional",
    ]
    et1_columns = ["checkpoint_id"] + keys + [
        "et1_raw_word_trt",
        "et1_symmetric_word_trt",
        "et1_asymmetric_word_trt",
    ]
    ob1_columns = keys + ["ob1_tvt"]
    merged = canonical_words[canonical_columns].merge(
        et1_words[et1_columns],
        on=keys,
        how="inner",
        validate="one_to_many",
    )
    merged = merged.merge(
        ob1_words[ob1_columns],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    checkpoints = merged["checkpoint_id"].nunique()
    expected = len(canonical_words) * checkpoints
    if len(merged) != expected:
        raise ValueError(
            f"Expected {expected} joined word rows, found {len(merged)}"
        )
    raw_variation = (
        merged.groupby(keys)["et1_raw_word_trt"].nunique(dropna=False).max()
    )
    if raw_variation != 1:
        raise ValueError("Raw ET1 values differ across RM checkpoints")
    return merged.sort_values(
        ["checkpoint_id", "passage_id_zero_based", "word_id_zero_based"]
    ).reset_index(drop=True)


def evaluate_passages(
    word_values: pd.DataFrame,
    human_column: str = "human_trt_unconditional",
) -> pd.DataFrame:
    """Compute every method metric per checkpoint and passage."""
    if human_column not in word_values:
        raise ValueError(f"Unknown Human TRT target: {human_column}")
    records = []
    for checkpoint_id, checkpoint_frame in word_values.groupby(
        "checkpoint_id",
        sort=True,
    ):
        for passage_id, passage in checkpoint_frame.groupby(
            "passage_id_zero_based",
            sort=True,
        ):
            passage = passage.sort_values("word_id_zero_based")
            human = passage[human_column].to_numpy(dtype=float)
            ob1 = passage["ob1_tvt"].to_numpy(dtype=float)
            for method, column in METHOD_COLUMNS.items():
                if method != "ob1" and passage[column].isna().any():
                    continue
                method_values = (
                    ob1 if method == "ob1" else passage[column].to_numpy(dtype=float)
                )
                metrics = passage_metric_values(human, method_values, ob1)
                records.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "passage_id_zero_based": int(passage_id),
                        "method": method,
                        "human_target": human_column,
                        "word_count": len(passage),
                        **metrics,
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No passage metrics were computed")
    return frame


def bootstrap_mean(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Bootstrap a passage-level mean and percentile confidence interval."""
    values = np.asarray(values, dtype=float)
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite with length at least two")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap_values = values[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_values, [2.5, 97.5])
    probability_nonpositive = float((bootstrap_values <= 0).mean())
    probability_nonnegative = float((bootstrap_values >= 0).mean())
    p_value = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )
    return float(values.mean()), float(lower), float(upper), p_value


def summarize_methods(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Average RM seeds within passage and bootstrap method-level means."""
    metric_columns = [
        "human_spearman",
        "js_divergence",
        "wasserstein",
        "ob1_spearman",
    ]
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    records = []
    for method, group in per_passage.groupby("method", sort=True):
        record = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "passages": len(group),
        }
        for metric in metric_columns:
            mean, lower, upper, _ = bootstrap_mean(
                group[metric].to_numpy(),
                bootstrap_samples,
                rng,
            )
            record[metric] = mean
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
        records.append(record)
    return pd.DataFrame(records)


def summarize_methods_by_checkpoint(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Produce a separate method summary for every RM checkpoint."""
    frames = []
    for offset, (checkpoint_id, checkpoint_metrics) in enumerate(
        passage_metrics.groupby("checkpoint_id", sort=True)
    ):
        summary = summarize_methods(
            checkpoint_metrics,
            bootstrap_samples,
            seed + offset,
        )
        summary.insert(0, "checkpoint_id", checkpoint_id)
        frames.append(summary)
    if not frames:
        raise ValueError("No checkpoint-level method summaries were computed")
    return pd.concat(frames, ignore_index=True)


def paired_contrasts(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired raw and asymmetry-specific improvements by passage."""
    metric_directions = {
        "human_spearman": "higher",
        "js_divergence": "lower",
        "wasserstein": "lower",
        "ob1_spearman": "higher",
    }
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )[list(metric_directions)]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    records = []
    comparisons = (
        ("et1_symmetric", "et1_raw"),
        ("et1_asymmetric", "et1_raw"),
        ("et1_asymmetric", "et1_symmetric"),
    )
    for method, baseline_method in comparisons:
        candidate = per_passage[per_passage["method"] == method].set_index(
            "passage_id_zero_based"
        )
        baseline = per_passage[
            per_passage["method"] == baseline_method
        ].set_index("passage_id_zero_based")
        common = baseline.index.intersection(candidate.index)
        if len(common) == 0:
            continue
        for metric, direction in metric_directions.items():
            if direction == "higher":
                differences = (
                    candidate.loc[common, metric] - baseline.loc[common, metric]
                ).to_numpy()
            else:
                differences = (
                    baseline.loc[common, metric] - candidate.loc[common, metric]
                ).to_numpy()
            mean, lower, upper, p_value = bootstrap_mean(
                differences,
                bootstrap_samples,
                rng,
            )
            records.append(
                {
                    "candidate": method,
                    "baseline": baseline_method,
                    "metric": metric,
                    "positive_means_improvement": True,
                    "passages": len(common),
                    "mean_paired_improvement": mean,
                    "ci_low": lower,
                    "ci_high": upper,
                    "bootstrap_p_two_sided": p_value,
                }
            )
    return pd.DataFrame(records)


def plot_human_spearman(
    passage_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot passage-level Human Spearman distributions with visible points."""
    per_passage = (
        passage_metrics.groupby(
            ["method", "passage_id_zero_based"],
            sort=True,
        )["human_spearman"]
        .mean()
        .reset_index()
    )
    methods = [
        method for method in METHOD_COLUMNS if method in per_passage["method"].unique()
    ]
    values = [
        per_passage.loc[
            per_passage["method"] == method,
            "human_spearman",
        ].to_numpy()
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.boxplot(values, labels=[DISPLAY_NAMES[item] for item in methods])
    rng = np.random.default_rng(0)
    for index, method_values in enumerate(values, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(method_values))
        axis.scatter(
            np.full(len(method_values), index) + jitter,
            method_values,
            s=14,
            alpha=0.55,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel("Human word-level TRT Spearman")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def write_evaluation_outputs(
    output_dir: Path,
    word_values: pd.DataFrame,
    passage_metrics: pd.DataFrame,
    method_summary: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    audit: dict,
) -> None:
    """Write machine-readable metrics, rebuttal table, plot, and audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    word_values.to_csv(output_dir / "word_level_values.csv", index=False)
    passage_metrics.to_csv(output_dir / "passage_metrics.csv", index=False)
    method_summary.to_csv(output_dir / "result_table.csv", index=False)
    checkpoint_summary.to_csv(
        output_dir / "seed_result_table.csv",
        index=False,
    )
    contrasts.to_csv(output_dir / "bootstrap_summary.csv", index=False)
    plot_human_spearman(
        passage_metrics,
        output_dir / "human_spearman_by_passage.png",
    )
    with (output_dir / "evaluation_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

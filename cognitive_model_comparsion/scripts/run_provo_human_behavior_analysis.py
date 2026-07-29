"""Evaluate frozen ET1 redistributions against Human Provo TRT and OB1 TVT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
DEFAULT_TOKEN_PATH = (
    COGNITIVE_ROOT
    / "outputs"
    / "et1_raw_55_passages"
    / "et1_token_values.csv"
)
DEFAULT_RAW_WORD_PATH = (
    COGNITIVE_ROOT
    / "outputs"
    / "et1_raw_55_passages"
    / "et1_word_values.csv"
)
DEFAULT_HUMAN_PATH = (
    COGNITIVE_ROOT / "data" / "processed" / "provo" / "provo_words.csv"
)
DEFAULT_OB1_PATH = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_ob1_100_rerun_20260728"
    / "ob1"
    / "ob1_word_values.csv"
)
DEFAULT_OUTPUT_DIR = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_human_behavior_best_seed_20260729"
)

SIGMA_LEFT = 0.3738
SIGMA_RIGHT = 3.21289
SOURCE_ACCURACY = 0.76942
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 10000
KEYS = ["passage_id_zero_based", "word_id_zero_based"]
TARGETS = (
    "human_trt_conditional",
    "human_trt_unconditional",
    "ob1_tvt",
)
TARGET_LABELS = {
    "human_trt_conditional": "Human conditional TRT",
    "human_trt_unconditional": "Human unconditional TRT",
    "ob1_tvt": "OB1 simulated TVT",
}
METHOD_LABELS = {
    "et1_raw": "ET1 raw",
    "fixed_symmetric_sigma1": "Fixed symmetric sigma=1",
    "same_rms_symmetric": "Symmetric, same parameter RMS",
    "realized_spread_matched_symmetric": (
        "Symmetric, ET1-weighted realized-spread match"
    ),
    "fixed_ratio4_same_rms": "4:1, same parameter RMS",
    "learned_asymmetric": "Learned asymmetric",
    "mirrored_learned": "Mirrored learned",
}
CONTRASTS = (
    ("fixed_symmetric_sigma1", "et1_raw"),
    ("learned_asymmetric", "et1_raw"),
    ("learned_asymmetric", "fixed_symmetric_sigma1"),
    ("learned_asymmetric", "realized_spread_matched_symmetric"),
    ("fixed_ratio4_same_rms", "same_rms_symmetric"),
    ("learned_asymmetric", "mirrored_learned"),
)
METRICS = (
    "spearman",
    "js_divergence",
    "hellinger_distance",
    "total_variation_distance",
    "overlap_coefficient",
)
HIGHER_IS_BETTER = {
    "spearman": True,
    "js_divergence": False,
    "hellinger_distance": False,
    "total_variation_distance": False,
    "overlap_coefficient": True,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_sigmas(
    realized_spread_sigma: float,
) -> dict[str, Optional[Tuple[float, float]]]:
    """Build fixed, width-control, learned, and mirrored sigma conditions."""
    parameter_rms = math.sqrt(
        (SIGMA_LEFT**2 + SIGMA_RIGHT**2) / 2.0
    )
    ratio4_left = parameter_rms * math.sqrt(2.0 / 17.0)
    return {
        "et1_raw": None,
        "fixed_symmetric_sigma1": (1.0, 1.0),
        "same_rms_symmetric": (parameter_rms, parameter_rms),
        "realized_spread_matched_symmetric": (
            realized_spread_sigma,
            realized_spread_sigma,
        ),
        "fixed_ratio4_same_rms": (ratio4_left, 4.0 * ratio4_left),
        "learned_asymmetric": (SIGMA_LEFT, SIGMA_RIGHT),
        "mirrored_learned": (SIGMA_RIGHT, SIGMA_LEFT),
    }


def redistribute(
    values: np.ndarray,
    attention_mask: np.ndarray,
    sigma_left: float,
    sigma_right: float,
) -> np.ndarray:
    """Apply the production source-normalized asymmetric Gaussian formula."""
    source_values = np.asarray(values, dtype=float)
    mask = np.asarray(attention_mask, dtype=float)
    if (
        source_values.ndim != 1
        or mask.ndim != 1
        or source_values.shape != mask.shape
    ):
        raise ValueError("Values and attention mask must be aligned vectors")
    if not np.isfinite(source_values).all() or np.any(source_values < 0):
        raise ValueError("ET1 values must be finite and nonnegative")
    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError("Attention mask must contain only zero and one")
    positions = np.arange(source_values.size, dtype=float)
    difference = positions[:, None] - positions[None, :]
    sigma = np.where(difference < 0, sigma_left, sigma_right)
    weights = np.exp(
        -0.5 * (np.abs(difference) / sigma) ** 2
    )
    weights *= mask[:, None]
    weights *= mask[None, :]
    denominator = weights.sum(axis=0, keepdims=True)
    weights = np.divide(
        weights,
        denominator,
        out=np.zeros_like(weights),
        where=denominator > 0,
    )
    return (
        np.sum(weights * source_values[None, :], axis=1) * mask
    )


def et1_weighted_displacement_second_moment(
    tokens: pd.DataFrame,
    sigma_left: float,
    sigma_right: float,
) -> float:
    """Compute the realized kernel second moment on the actual ET1 inputs."""
    weighted_moment = 0.0
    source_mass = 0.0
    for _, passage in tokens.groupby(
        "passage_id_zero_based",
        sort=True,
    ):
        passage = passage.sort_values("token_index")
        values = passage["et1_raw_token_trt"].to_numpy(dtype=float)
        mask = passage["attention_mask"].to_numpy(dtype=float)
        positions = np.arange(len(passage), dtype=float)
        difference = positions[:, None] - positions[None, :]
        sigma = np.where(
            difference < 0,
            sigma_left,
            sigma_right,
        )
        weights = np.exp(
            -0.5 * (np.abs(difference) / sigma) ** 2
        )
        weights *= mask[:, None]
        weights *= mask[None, :]
        denominator = weights.sum(axis=0, keepdims=True)
        weights = np.divide(
            weights,
            denominator,
            out=np.zeros_like(weights),
            where=denominator > 0,
        )
        source_weights = values * mask
        source_second_moment = (
            weights * np.square(difference)
        ).sum(axis=0)
        weighted_moment += float(
            np.sum(source_weights * source_second_moment)
        )
        source_mass += float(source_weights.sum())
    if source_mass <= 0:
        raise ValueError("ET1 source mass must be positive")
    return weighted_moment / source_mass


def solve_realized_spread_matched_symmetric_sigma(
    tokens: pd.DataFrame,
) -> tuple[float, float]:
    """Match the learned kernel's realized ET1-weighted RMS displacement."""
    target_second_moment = et1_weighted_displacement_second_moment(
        tokens,
        SIGMA_LEFT,
        SIGMA_RIGHT,
    )
    lower = 1e-4
    upper = max(10.0, SIGMA_RIGHT * 4.0)
    while (
        et1_weighted_displacement_second_moment(
            tokens,
            upper,
            upper,
        )
        < target_second_moment
    ):
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        midpoint_moment = et1_weighted_displacement_second_moment(
            tokens,
            midpoint,
            midpoint,
        )
        if midpoint_moment < target_second_moment:
            lower = midpoint
        else:
            upper = midpoint
    matched_sigma = 0.5 * (lower + upper)
    matched_second_moment = et1_weighted_displacement_second_moment(
        tokens,
        matched_sigma,
        matched_sigma,
    )
    if abs(matched_second_moment - target_second_moment) > 1e-10:
        raise ValueError("Realized-spread matching did not converge")
    return matched_sigma, target_second_moment


def validate_input_tables(
    tokens: pd.DataFrame,
    saved_raw_words: pd.DataFrame,
    human: pd.DataFrame,
    ob1: pd.DataFrame,
) -> None:
    """Validate required schemas and unique coordinate grids."""
    token_columns = {
        "passage_id_zero_based",
        "token_index",
        "attention_mask",
        "is_special",
        "word_id_zero_based",
        "et1_raw_token_trt",
    }
    raw_word_columns = set(KEYS + ["et1_raw_word_trt"])
    human_columns = set(
        KEYS
        + [
            "passage_id_raw",
            "word_raw",
            "human_reader_count",
            "human_trt_conditional",
            "human_trt_unconditional",
        ]
    )
    ob1_columns = set(KEYS + ["ob1_tvt"])
    requirements = (
        ("token", tokens, token_columns),
        ("saved raw word", saved_raw_words, raw_word_columns),
        ("Human", human, human_columns),
        ("OB1", ob1, ob1_columns),
    )
    for label, table, required in requirements:
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{label} table is missing columns: {missing}")
    for label, table in (
        ("saved raw word", saved_raw_words),
        ("Human", human),
        ("OB1", ob1),
    ):
        if table.duplicated(KEYS).any():
            raise ValueError(f"{label} table has duplicate word coordinates")
    human_grid = set(human[KEYS].itertuples(index=False, name=None))
    ob1_grid = set(ob1[KEYS].itertuples(index=False, name=None))
    raw_grid = set(
        saved_raw_words[KEYS].itertuples(index=False, name=None)
    )
    if human_grid != ob1_grid or human_grid != raw_grid:
        raise ValueError("Human, OB1, and saved raw ET1 grids differ")
    if tokens["passage_id_zero_based"].nunique() != 55:
        raise ValueError("Expected 55 Provo passages in the ET1 token table")
    for passage_id, passage in tokens.groupby(
        "passage_id_zero_based",
        sort=True,
    ):
        indices = (
            passage.sort_values("token_index")["token_index"]
            .to_numpy(dtype=int)
        )
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError(
                f"Noncontiguous token positions in passage {passage_id}"
            )


def build_redistributions(
    tokens: pd.DataFrame,
    evaluation_grid: pd.DataFrame,
    sigmas: dict[str, Optional[Tuple[float, float]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Redistribute full passage sequences and aggregate them to words."""
    token_groups = []
    word_groups = []
    mass_rows = []
    evaluable_coordinates = set(
        evaluation_grid[KEYS].itertuples(index=False, name=None)
    )
    for passage_id, passage in tokens.groupby(
        "passage_id_zero_based",
        sort=True,
    ):
        passage = passage.sort_values("token_index").copy()
        raw = passage["et1_raw_token_trt"].to_numpy(dtype=float)
        mask = passage["attention_mask"].to_numpy(dtype=float)
        passage_output = passage.copy()
        method_values = {}
        for method, sigma_pair in sigmas.items():
            if sigma_pair is None:
                redistributed = raw * mask
            else:
                redistributed = redistribute(
                    raw,
                    mask,
                    sigma_pair[0],
                    sigma_pair[1],
                )
            method_values[method] = redistributed
            passage_output[method] = redistributed
            input_mass = float((raw * mask).sum())
            output_mass = float(redistributed.sum())
            tolerance = 1e-8 * max(1.0, input_mass)
            if abs(input_mass - output_mass) > tolerance:
                raise ValueError(
                    f"Mass changed for {method}, passage {passage_id}: "
                    f"{input_mass} -> {output_mass}"
                )
        token_groups.append(passage_output)
        assigned = passage["word_id_zero_based"].notna().to_numpy()
        assigned_coordinates = passage.loc[
            assigned,
            ["passage_id_zero_based", "word_id_zero_based"],
        ].copy()
        word_output = None
        for method, redistributed in method_values.items():
            method_tokens = assigned_coordinates.copy()
            method_tokens[method] = redistributed[assigned]
            method_words = (
                method_tokens.groupby(KEYS, as_index=False)[method]
                .sum()
            )
            if word_output is None:
                word_output = method_words
            else:
                word_output = word_output.merge(
                    method_words,
                    on=KEYS,
                    how="outer",
                    validate="one_to_one",
                )
        if word_output is None:
            raise ValueError(f"No word tokens in passage {passage_id}")
        word_groups.append(word_output)
        for method, redistributed in method_values.items():
            full_mass = float(redistributed.sum())
            assigned_mass = float(redistributed[assigned].sum())
            evaluated_mask = np.asarray(
                [
                    (
                        int(passage_id),
                        int(word_id),
                    )
                    in evaluable_coordinates
                    if not pd.isna(word_id)
                    else False
                    for word_id in passage["word_id_zero_based"]
                ],
                dtype=bool,
            )
            evaluated_mass = float(redistributed[evaluated_mask].sum())
            mass_rows.append(
                {
                    "passage_id_zero_based": int(passage_id),
                    "method": method,
                    "input_valid_token_mass": float((raw * mask).sum()),
                    "output_valid_token_mass": full_mass,
                    "absolute_mass_error": abs(
                        float((raw * mask).sum()) - full_mass
                    ),
                    "word_assigned_mass": assigned_mass,
                    "evaluated_word_mass": evaluated_mass,
                    "evaluated_mass_retention": (
                        evaluated_mass / full_mass
                        if full_mass > 0
                        else np.nan
                    ),
                    "unassigned_special_mass": full_mass - assigned_mass,
                }
            )
    token_output = pd.concat(token_groups, ignore_index=True)
    all_words = pd.concat(word_groups, ignore_index=True)
    mass_audit = pd.DataFrame(mass_rows)
    return token_output, all_words, mass_audit


def join_evaluation_grid(
    all_words: pd.DataFrame,
    human: pd.DataFrame,
    ob1: pd.DataFrame,
) -> pd.DataFrame:
    """Join redistributed ET1 word values to Human and OB1 references."""
    human_columns = KEYS + [
        "passage_id_raw",
        "word_raw",
        "human_reader_count",
        "human_trt_conditional",
        "human_trt_unconditional",
    ]
    ob1_columns = KEYS + ["ob1_tvt"]
    merged = human[human_columns].merge(
        ob1[ob1_columns],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        all_words,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    method_columns = list(METHOD_LABELS)
    if merged[method_columns].isna().any().any():
        raise ValueError("Redistributed ET1 values are missing on the grid")
    if len(merged) != 2686 or merged["passage_id_zero_based"].nunique() != 55:
        raise ValueError("Expected 2,686 words across 55 passages")
    return merged.sort_values(KEYS).reset_index(drop=True)


def normalized_allocation(values: np.ndarray) -> np.ndarray:
    """Normalize one finite nonnegative passage allocation to unit mass."""
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Allocation must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Allocation must have positive mass")
    return array / total


def spearman_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute Spearman correlation using average ranks."""
    left = pd.Series(np.asarray(reference, dtype=float)).rank(
        method="average"
    )
    right = pd.Series(np.asarray(candidate, dtype=float)).rank(
        method="average"
    )
    value = float(left.corr(right, method="pearson"))
    if not math.isfinite(value):
        raise ValueError("Spearman correlation is undefined")
    return value


def distribution_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    """Compute rank and normalized-allocation similarity metrics."""
    p = normalized_allocation(reference)
    q = normalized_allocation(candidate)
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    js_divergence = 0.5 * (
        float(np.sum(p[p_mask] * np.log2(p[p_mask] / midpoint[p_mask])))
        + float(
            np.sum(q[q_mask] * np.log2(q[q_mask] / midpoint[q_mask]))
        )
    )
    hellinger = float(
        np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
    )
    total_variation = float(0.5 * np.abs(p - q).sum())
    return {
        "spearman": spearman_correlation(reference, candidate),
        "js_divergence": js_divergence,
        "hellinger_distance": hellinger,
        "total_variation_distance": total_variation,
        "overlap_coefficient": 1.0 - total_variation,
    }


def evaluate_passages(word_values: pd.DataFrame) -> pd.DataFrame:
    """Compute every method-target metric independently within passage."""
    rows = []
    for passage_id, passage in word_values.groupby(
        "passage_id_zero_based",
        sort=True,
    ):
        passage = passage.sort_values("word_id_zero_based")
        for target in TARGETS:
            reference = passage[target].to_numpy(dtype=float)
            for method in METHOD_LABELS:
                metrics = distribution_metrics(
                    reference,
                    passage[method].to_numpy(dtype=float),
                )
                rows.append(
                    {
                        "passage_id_zero_based": int(passage_id),
                        "target": target,
                        "method": method,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_mean(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return a mean and 95 percent percentile passage-bootstrap interval."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all() or array.size < 2:
        raise ValueError("Bootstrap requires a finite one-dimensional vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        array.size,
        size=(samples, array.size),
    )
    replicates = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(replicates, 0.025)),
        float(np.quantile(replicates, 0.975)),
    )


def summarize_methods(
    passage_metrics: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize passage-level metrics with bootstrap confidence intervals."""
    rows = []
    for (target, method), group in passage_metrics.groupby(
        ["target", "method"],
        sort=False,
    ):
        row = {
            "target": target,
            "target_label": TARGET_LABELS[target],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "passages": int(group["passage_id_zero_based"].nunique()),
        }
        for metric in METRICS:
            mean, lower, upper = bootstrap_mean(
                group[metric].to_numpy(dtype=float),
                samples,
                seed,
            )
            row[metric] = mean
            row[f"{metric}_ci_low"] = lower
            row[f"{metric}_ci_high"] = upper
        rows.append(row)
    summary = pd.DataFrame(rows)
    target_order = {value: index for index, value in enumerate(TARGETS)}
    method_order = {
        value: index for index, value in enumerate(METHOD_LABELS)
    }
    summary["_target_order"] = summary["target"].map(target_order)
    summary["_method_order"] = summary["method"].map(method_order)
    return (
        summary.sort_values(["_target_order", "_method_order"])
        .drop(columns=["_target_order", "_method_order"])
        .reset_index(drop=True)
    )


def paired_contrasts(
    passage_metrics: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired passage-bootstrap method improvements."""
    rows = []
    for target in TARGETS:
        target_rows = passage_metrics.loc[
            passage_metrics["target"].eq(target)
        ]
        for candidate, baseline in CONTRASTS:
            for metric in METRICS:
                pivot = target_rows.pivot(
                    index="passage_id_zero_based",
                    columns="method",
                    values=metric,
                )
                if HIGHER_IS_BETTER[metric]:
                    improvement = (
                        pivot[candidate] - pivot[baseline]
                    ).to_numpy(dtype=float)
                else:
                    improvement = (
                        pivot[baseline] - pivot[candidate]
                    ).to_numpy(dtype=float)
                mean, lower, upper = bootstrap_mean(
                    improvement,
                    samples,
                    seed,
                )
                rows.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "candidate": candidate,
                        "candidate_label": METHOD_LABELS[candidate],
                        "baseline": baseline,
                        "baseline_label": METHOD_LABELS[baseline],
                        "metric": metric,
                        "positive_means_improvement": True,
                        "passages": int(improvement.size),
                        "mean_paired_improvement": mean,
                        "ci_low": lower,
                        "ci_high": upper,
                    }
                )
    return pd.DataFrame(rows)


def validate_raw_word_reconstruction(
    word_values: pd.DataFrame,
    saved_raw_words: pd.DataFrame,
) -> float:
    """Check reconstructed raw word TRT against the saved full-run table."""
    comparison = saved_raw_words[
        KEYS + ["et1_raw_word_trt"]
    ].merge(
        word_values[KEYS + ["et1_raw"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(comparison) != len(saved_raw_words):
        raise ValueError("Raw word reconstruction grid is incomplete")
    maximum_error = float(
        np.abs(
            comparison["et1_raw_word_trt"].to_numpy(dtype=float)
            - comparison["et1_raw"].to_numpy(dtype=float)
        ).max()
    )
    if maximum_error > 1e-5:
        raise ValueError(
            f"Raw word reconstruction differs by {maximum_error}"
        )
    return maximum_error


def format_interval(row: pd.Series, metric: str) -> str:
    """Format one estimate and confidence interval for Markdown."""
    return (
        f"{row[metric]:.4f} "
        f"[{row[f'{metric}_ci_low']:.4f}, "
        f"{row[f'{metric}_ci_high']:.4f}]"
    )


def write_results_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    sigmas: dict[str, Optional[Tuple[float, float]]],
) -> None:
    """Write compact result tables and interpretation boundaries."""
    lines = [
        "# Frozen ET1 redistribution versus Human Provo and OB1",
        "",
        "All redistributions use actual passage-specific frozen ET1 outputs. "
        "They are applied in native T5-token coordinates before summing "
        "subtokens into Provo words.",
        "",
        "## Sigma conditions",
        "",
        "| Method | sigma_left | sigma_right |",
        "|---|---:|---:|",
    ]
    for method, sigma_pair in sigmas.items():
        if sigma_pair is None:
            left = right = "not applicable"
        else:
            left = f"{sigma_pair[0]:.6f}"
            right = f"{sigma_pair[1]:.6f}"
        lines.append(
            f"| {METHOD_LABELS[method]} | {left} | {right} |"
        )
    for target in TARGETS:
        selected = summary.loc[summary["target"].eq(target)]
        lines.extend(
            [
                "",
                f"## {TARGET_LABELS[target]}",
                "",
                "| Method | Spearman | JS divergence | Hellinger | TV |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in selected.itertuples(index=False):
            series = pd.Series(row._asdict())
            lines.append(
                f"| {row.method_label} | "
                f"{format_interval(series, 'spearman')} | "
                f"{format_interval(series, 'js_divergence')} | "
                f"{format_interval(series, 'hellinger_distance')} | "
                f"{format_interval(series, 'total_variation_distance')} |"
            )
    requested_pairs = {
        (
            "human_trt_conditional",
            "learned_asymmetric",
            "fixed_symmetric_sigma1",
        ),
        (
            "human_trt_conditional",
            "learned_asymmetric",
            "realized_spread_matched_symmetric",
        ),
        (
            "human_trt_conditional",
            "fixed_ratio4_same_rms",
            "same_rms_symmetric",
        ),
        (
            "human_trt_conditional",
            "learned_asymmetric",
            "mirrored_learned",
        ),
        (
            "human_trt_unconditional",
            "fixed_ratio4_same_rms",
            "same_rms_symmetric",
        ),
        (
            "human_trt_unconditional",
            "learned_asymmetric",
            "mirrored_learned",
        ),
    }
    selected_contrasts = contrasts.loc[
        contrasts.apply(
            lambda row: (
                row["target"],
                row["candidate"],
                row["baseline"],
            )
            in requested_pairs,
            axis=1,
        )
        & contrasts["metric"].isin(
            [
                "spearman",
                "js_divergence",
                "hellinger_distance",
                "total_variation_distance",
            ]
        )
    ]
    lines.extend(
        [
            "",
            "## Direction and control contrasts",
            "",
            "Positive values favor the candidate for every metric.",
            "",
            "| Target | Candidate vs baseline | Metric | Improvement [95% CI] |",
            "|---|---|---|---:|",
        ]
    )
    for row in selected_contrasts.itertuples(index=False):
        lines.append(
            f"| {row.target_label} | {row.candidate_label} vs "
            f"{row.baseline_label} | {row.metric} | "
            f"{row.mean_paired_improvement:.4f} "
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Human TRT is observed word-level reading behavior, not a direct "
            "measurement of a covert attention-kernel width.",
            "- Parameter RMS preserves the Euclidean magnitude of the two "
            "side-scale parameters; it does not guarantee identical realized "
            "variance after masking, truncation, and normalization.",
            "- The mirrored control isolates direction while preserving the "
            "same two side scales and parameter count.",
            "- These analyses can support external directional consistency, "
            "but not recovery of a human perceptual span.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_plots(
    output_dir: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> list[str]:
    """Write metric and paired-direction-control figures when available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    method_colors = {
        "et1_raw": "#4C78A8",
        "fixed_symmetric_sigma1": "#59A14F",
        "same_rms_symmetric": "#9C755F",
        "realized_spread_matched_symmetric": "#F28E2B",
        "fixed_ratio4_same_rms": "#B279A2",
        "learned_asymmetric": "#E45756",
        "mirrored_learned": "#79706E",
    }
    metrics = (
        ("spearman", "Spearman correlation", True),
        ("js_divergence", "Jensen-Shannon divergence", False),
        ("hellinger_distance", "Hellinger distance", False),
        ("total_variation_distance", "Total variation distance", False),
    )
    figure, axes = plt.subplots(
        len(TARGETS),
        len(metrics),
        figsize=(19, 12),
        constrained_layout=True,
    )
    for target_index, target in enumerate(TARGETS):
        target_summary = (
            summary.loc[summary["target"].eq(target)]
            .set_index("method")
            .loc[list(METHOD_LABELS)]
        )
        positions = np.arange(len(target_summary))
        for metric_index, (metric, label, higher) in enumerate(metrics):
            axis = axes[target_index, metric_index]
            values = target_summary[metric].to_numpy(dtype=float)
            lower = values - target_summary[
                f"{metric}_ci_low"
            ].to_numpy(dtype=float)
            upper = target_summary[
                f"{metric}_ci_high"
            ].to_numpy(dtype=float) - values
            axis.bar(
                positions,
                values,
                color=[
                    method_colors[method]
                    for method in target_summary.index
                ],
                yerr=np.vstack([lower, upper]),
                capsize=3,
            )
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [
                    METHOD_LABELS[method]
                    for method in target_summary.index
                ],
                rotation=40,
                ha="right",
            )
            axis.set_title(
                f"{TARGET_LABELS[target]}\n{label} "
                f"({'higher' if higher else 'lower'} is better)"
            )
            axis.grid(axis="y", alpha=0.25)
    metric_path = output_dir / "human_ob1_metric_comparison.png"
    figure.savefig(metric_path, dpi=180)
    plt.close(figure)

    controls = contrasts.loc[
        contrasts["metric"].isin(
            [
                "spearman",
                "js_divergence",
                "hellinger_distance",
                "total_variation_distance",
            ]
        )
        & contrasts.apply(
            lambda row: (
                (row["candidate"], row["baseline"])
                in {
                    (
                        "fixed_ratio4_same_rms",
                        "same_rms_symmetric",
                    ),
                    ("learned_asymmetric", "mirrored_learned"),
                }
            ),
            axis=1,
        )
    ].copy()
    controls["contrast"] = (
        controls["candidate_label"] + " vs " + controls["baseline_label"]
    )
    figure, axes = plt.subplots(
        1,
        len(TARGETS),
        figsize=(18, 5.5),
        constrained_layout=True,
    )
    for target_index, target in enumerate(TARGETS):
        axis = axes[target_index]
        target_controls = controls.loc[
            controls["target"].eq(target)
        ].reset_index(drop=True)
        positions = np.arange(len(target_controls))
        values = target_controls[
            "mean_paired_improvement"
        ].to_numpy(dtype=float)
        lower = values - target_controls["ci_low"].to_numpy(dtype=float)
        upper = target_controls["ci_high"].to_numpy(dtype=float) - values
        colors = [
            "#B279A2"
            if candidate == "fixed_ratio4_same_rms"
            else "#E45756"
            for candidate in target_controls["candidate"]
        ]
        axis.errorbar(
            positions,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#555555",
            capsize=4,
            linewidth=2,
        )
        axis.scatter(positions, values, c=colors, s=55, zorder=3)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [
                f"{row.metric}\n{row.contrast}"
                for row in target_controls.itertuples(index=False)
            ],
            rotation=45,
            ha="right",
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_ylabel("Paired improvement")
        axis.grid(axis="y", alpha=0.25)
    contrast_path = output_dir / "direction_control_contrasts.png"
    figure.savefig(contrast_path, dpi=180)
    plt.close(figure)
    return [str(metric_path), str(contrast_path)]


def run_analysis(
    token_path: Path,
    raw_word_path: Path,
    human_path: Path,
    ob1_path: Path,
    output_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Run the complete frozen-checkpoint Human and OB1 analysis."""
    paths = {
        "et1_tokens": token_path.expanduser().resolve(),
        "saved_et1_raw_words": raw_word_path.expanduser().resolve(),
        "human_provo_words": human_path.expanduser().resolve(),
        "ob1_words": ob1_path.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens = pd.read_csv(paths["et1_tokens"])
    saved_raw_words = pd.read_csv(paths["saved_et1_raw_words"])
    human = pd.read_csv(paths["human_provo_words"])
    ob1 = pd.read_csv(paths["ob1_words"])
    validate_input_tables(tokens, saved_raw_words, human, ob1)
    realized_spread_sigma, learned_second_moment = (
        solve_realized_spread_matched_symmetric_sigma(tokens)
    )
    sigmas = method_sigmas(realized_spread_sigma)
    token_values, all_words, mass_audit = build_redistributions(
        tokens,
        human[KEYS],
        sigmas,
    )
    word_values = join_evaluation_grid(all_words, human, ob1)
    raw_reconstruction_error = validate_raw_word_reconstruction(
        word_values,
        saved_raw_words,
    )
    passage_metrics = evaluate_passages(word_values)
    summary = summarize_methods(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    contrasts = paired_contrasts(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    token_values.to_csv(
        output_dir / "token_redistributions.csv",
        index=False,
    )
    word_values.to_csv(
        output_dir / "word_level_values.csv",
        index=False,
    )
    mass_audit.to_csv(
        output_dir / "mass_audit.csv",
        index=False,
    )
    passage_metrics.to_csv(
        output_dir / "passage_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "result_table.csv",
        index=False,
    )
    contrasts.to_csv(
        output_dir / "paired_contrasts.csv",
        index=False,
    )
    write_results_markdown(output_dir, summary, contrasts, sigmas)
    plot_paths = write_plots(output_dir, summary, contrasts)
    parameter_rms = sigmas["same_rms_symmetric"][0]
    ratio4_sigmas = sigmas["fixed_ratio4_same_rms"]
    audit = {
        "analysis": "frozen_et1_human_provo_and_ob1_behavior_validation",
        "source_accuracy": SOURCE_ACCURACY,
        "learned_sigma_left": SIGMA_LEFT,
        "learned_sigma_right": SIGMA_RIGHT,
        "learned_right_left_ratio": SIGMA_RIGHT / SIGMA_LEFT,
        "quadratic_side_scale_rms": parameter_rms,
        "learned_et1_weighted_realized_displacement_second_moment": (
            learned_second_moment
        ),
        "learned_et1_weighted_realized_rms_displacement": math.sqrt(
            learned_second_moment
        ),
        "realized_spread_matched_symmetric_sigma": (
            realized_spread_sigma
        ),
        "realized_spread_match_uses_human_or_ob1_target": False,
        "realized_spread_match_scope": (
            "exact full-passage valid T5-token supports, production "
            "source normalization, weighted by actual ET1 source mass"
        ),
        "fixed_ratio4_same_rms_sigma_left": ratio4_sigmas[0],
        "fixed_ratio4_same_rms_sigma_right": ratio4_sigmas[1],
        "sigma_values_selected_without_provo_or_ob1": True,
        "actual_passage_specific_et1_values_used": True,
        "redistribution_coordinate": "native T5 token index",
        "word_aggregation": "sum native T5-token values by word coordinate",
        "special_token_policy": (
            "production-faithful attention mask includes one valid EOS "
            "position per passage during redistribution"
        ),
        "mass_policy": (
            "full valid-token mass is conserved; mass retained on the "
            "2,686-word evaluation grid is separately audited"
        ),
        "human_targets": {
            "human_trt_conditional": (
                "participant mean over positive IA_DWELL_TIME observations"
            ),
            "human_trt_unconditional": (
                "participant mean over all IA_DWELL_TIME observations, "
                "including zero"
            ),
        },
        "ob1_target": "mean word TVT over 100 simulations",
        "passages": int(word_values["passage_id_zero_based"].nunique()),
        "word_rows": int(len(word_values)),
        "token_rows": int(len(token_values)),
        "human_reader_count_values": sorted(
            {
                int(value)
                for value in word_values["human_reader_count"].unique()
            }
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "raw_word_reconstruction_max_abs_error": (
            raw_reconstruction_error
        ),
        "maximum_full_token_mass_error": float(
            mass_audit["absolute_mass_error"].max()
        ),
        "metric_scope": (
            "Spearman on common passage words; JS base 2, Hellinger, TV, "
            "and overlap on passage-normalized word allocations"
        ),
        "interpretation_boundary": (
            "Human word TRT validates behavior-level correspondence but "
            "does not directly estimate a covert perceptual-span width"
        ),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "plots": plot_paths,
    }
    (output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "passages": audit["passages"],
        "word_rows": audit["word_rows"],
        "token_rows": audit["token_rows"],
        "maximum_full_token_mass_error": (
            audit["maximum_full_token_mass_error"]
        ),
        "raw_word_reconstruction_max_abs_error": (
            audit["raw_word_reconstruction_max_abs_error"]
        ),
        "plots": plot_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply frozen ET1 Gaussian redistributions on native T5 tokens "
            "and compare word allocations with Human Provo TRT and OB1 TVT."
        )
    )
    parser.add_argument("--token-path", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument(
        "--raw-word-path",
        type=Path,
        default=DEFAULT_RAW_WORD_PATH,
    )
    parser.add_argument("--human-path", type=Path, default=DEFAULT_HUMAN_PATH)
    parser.add_argument("--ob1-path", type=Path, default=DEFAULT_OB1_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> None:
    """Run the command-line analysis."""
    args = build_parser().parse_args()
    result = run_analysis(
        token_path=args.token_path,
        raw_word_path=args.raw_word_path,
        human_path=args.human_path,
        ob1_path=args.ob1_path,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

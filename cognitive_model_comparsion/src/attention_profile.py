"""Compare redistribution kernels with OB1's internal attention profile."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance

from cognitive_model_comparsion.src.evaluate import (
    bootstrap_mean,
    paired_sign_flip_pvalue,
)


PROFILE_METHODS = (
    "raw_delta",
    "width_matched_symmetric",
    "fixed_ob1_gaussian",
    "learned_asymmetric",
)
PROFILE_DISPLAY_NAMES = {
    "raw_delta": "ET1 raw (delta kernel)",
    "width_matched_symmetric": "Width-matched symmetric",
    "fixed_ob1_gaussian": "Fixed OB1-fitted Gaussian",
    "learned_asymmetric": "Learned asymmetric",
}
PROFILE_METRIC_DIRECTIONS = {
    "profile_spearman": "higher",
    "js_divergence": "lower",
    "token_offset_wasserstein": "lower",
}
PROFILE_COMPARISONS = (
    ("learned_asymmetric", "raw_delta"),
    ("learned_asymmetric", "width_matched_symmetric"),
    ("fixed_ob1_gaussian", "raw_delta"),
    ("fixed_ob1_gaussian", "width_matched_symmetric"),
    ("learned_asymmetric", "fixed_ob1_gaussian"),
)
OB1_MIN_ATTENTION_WIDTH = 3.0
OB1_MAX_ATTENTION_WIDTH = 5.0
OB1_RESIDUAL_ATTENTION = 0.25
OB1_WINDOW_WORDS_LEFT = 1
OB1_WINDOW_WORDS_RIGHT = 3


def clean_ob1_word(value: str) -> str:
    """Apply the vendored OB1 word normalization."""
    return re.sub(r"[^\w\s]", "", str(value)).lower().strip()


def whitespace_word_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open source-text spans for whitespace-delimited words."""
    return [match.span() for match in re.finditer(r"\S+", str(text))]


def build_t5_letter_geometry(
    passage_text: str,
    passage_tokens: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Map native T5 text tokens into OB1's cleaned-letter coordinates."""
    source_text = str(passage_text)
    word_spans = whitespace_word_spans(source_text)
    clean_words = [
        clean_ob1_word(source_text[start:end])
        for start, end in word_spans
    ]
    if not clean_words or any(not word for word in clean_words):
        raise ValueError("OB1 projection requires nonempty cleaned words")
    word_starts = []
    cursor = 0
    for word in clean_words:
        word_starts.append(cursor)
        cursor += len(word) + 1

    rows = []
    for token in passage_tokens.sort_values("token_index").itertuples():
        word_id = int(token.word_id_zero_based)
        if word_id < 0 or word_id >= len(word_spans):
            raise ValueError("T5 token word ID is outside the passage words")
        word_start, word_end = word_spans[word_id]
        clean_positions = [
            source_index
            for source_index in range(word_start, word_end)
            if re.match(r"\w", source_text[source_index])
        ]
        rank_by_source = {
            source_index: rank
            for rank, source_index in enumerate(clean_positions)
        }
        token_clean_ranks = [
            rank_by_source[source_index]
            for source_index in range(
                max(int(token.character_start), word_start),
                min(int(token.character_end), word_end),
            )
            if source_index in rank_by_source
        ]
        if token_clean_ranks:
            center_in_word = float(np.mean(token_clean_ranks))
            punctuation_only = False
        else:
            raw_width = max(word_end - word_start, 1)
            raw_center = (
                float(token.character_start)
                + float(token.character_end)
            ) / 2.0
            relative_center = np.clip(
                (raw_center - word_start) / raw_width,
                0.0,
                1.0,
            )
            center_in_word = relative_center * max(
                len(clean_words[word_id]) - 1,
                0,
            )
            punctuation_only = True
        rows.append(
            {
                "token_index": int(token.token_index),
                "word_id_zero_based": word_id,
                "ob1_letter_center": (
                    float(word_starts[word_id]) + center_in_word
                ),
                "punctuation_only_token": punctuation_only,
            }
        )
    geometry = pd.DataFrame(rows)
    if geometry.empty or geometry["token_index"].duplicated().any():
        raise ValueError("T5 letter geometry must contain unique tokens")
    return geometry, clean_words, word_starts


def effective_ob1_attention_width(
    recorded_width: float,
    saccade_type: object,
) -> float:
    """Recover the width used after OB1's per-fixation width update."""
    width = float(recorded_width)
    if not math.isfinite(width):
        raise ValueError("OB1 attentional width must be finite")
    normalized_type = (
        "" if pd.isna(saccade_type) else str(saccade_type).strip().lower()
    )
    if normalized_type == "regression":
        return max(width - 1.0, OB1_MIN_ATTENTION_WIDTH)
    return min(width + 0.5, OB1_MAX_ATTENTION_WIDTH)


def ob1_attention_weight(
    eccentricity: np.ndarray,
    attention_width: float,
    attention_skew: float,
    profile_component: str = "focused",
) -> np.ndarray:
    """Evaluate the vendored OB1 asymmetric attention equation."""
    eccentricity = np.asarray(eccentricity, dtype=float)
    if attention_width <= 0 or attention_skew < 1:
        raise ValueError("OB1 width must be positive and skew must be at least one")
    sigma = np.where(
        eccentricity < 0,
        attention_width / attention_skew,
        attention_width,
    )
    focused = (
        np.exp(-0.5 * (np.abs(eccentricity) / sigma) ** 2)
        / attention_width
    )
    if profile_component == "focused":
        return focused
    if profile_component == "full":
        return focused + OB1_RESIDUAL_ATTENTION
    raise ValueError("OB1 profile component must be focused or full")


def unique_et1_token_grid(token_values: pd.DataFrame) -> pd.DataFrame:
    """Validate repeated checkpoint metadata and retain one ET1 token grid."""
    required = {
        "checkpoint_id",
        "passage_id_zero_based",
        "token_index",
        "character_start",
        "character_end",
        "is_special",
        "attention_mask",
        "word_id_zero_based",
    }
    missing = sorted(required - set(token_values.columns))
    if missing:
        raise ValueError(f"ET1 token table is missing columns: {missing}")
    metadata_columns = [
        "passage_id_zero_based",
        "token_index",
        "character_start",
        "character_end",
        "is_special",
        "attention_mask",
        "word_id_zero_based",
    ]
    distinct = token_values[metadata_columns].drop_duplicates()
    duplicate_keys = distinct.duplicated(
        ["passage_id_zero_based", "token_index"],
        keep=False,
    )
    if bool(duplicate_keys.any()):
        raise ValueError("ET1 checkpoint rows disagree on token metadata")
    grid = distinct.loc[
        distinct["attention_mask"].eq(1)
        & distinct["is_special"].eq(0)
        & distinct["word_id_zero_based"].notna()
    ].copy()
    if grid.empty:
        raise ValueError("ET1 token grid contains no evaluable text tokens")
    grid["passage_id_zero_based"] = grid[
        "passage_id_zero_based"
    ].astype(int)
    grid["token_index"] = grid["token_index"].astype(int)
    grid["word_id_zero_based"] = grid["word_id_zero_based"].astype(int)
    return grid.sort_values(
        ["passage_id_zero_based", "token_index"]
    ).reset_index(drop=True)


def gaussian_weights(
    relative_offsets: np.ndarray,
    sigma_left: float,
    sigma_right: float,
) -> np.ndarray:
    """Return one normalized production-shaped kernel on visible offsets."""
    offsets = np.asarray(relative_offsets, dtype=float)
    if sigma_left <= 0 or sigma_right <= 0:
        raise ValueError("Gaussian sigmas must be positive")
    sigmas = np.where(offsets < 0, sigma_left, sigma_right)
    weights = np.exp(-0.5 * (np.abs(offsets) / sigmas) ** 2)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Gaussian kernel has invalid mass")
    return weights / total


def delta_weights(relative_offsets: np.ndarray) -> np.ndarray:
    """Return a normalized no-redistribution delta on the anchor token."""
    offsets = np.asarray(relative_offsets, dtype=int)
    weights = offsets.eq(0) if isinstance(offsets, pd.Series) else offsets == 0
    weights = np.asarray(weights, dtype=float)
    if int(weights.sum()) != 1:
        raise ValueError("Every fixation geometry must contain one anchor token")
    return weights


def project_ob1_attention_to_t5(
    passages: pd.DataFrame,
    token_values: pd.DataFrame,
    ob1_fixations: pd.DataFrame,
    attention_skews: tuple[float, ...],
    fixation_weighting: str,
    profile_component: str,
) -> tuple[
    dict[float, dict[int, dict[int, float]]],
    np.ndarray,
    dict,
]:
    """Project fixation-onset OB1 attention into native T5 offsets."""
    if fixation_weighting not in {"duration", "equal"}:
        raise ValueError("fixation weighting must be duration or equal")
    if profile_component not in {"focused", "full"}:
        raise ValueError("OB1 profile component must be focused or full")
    required_passages = {
        "passage_id_zero_based",
        "passage_text",
    }
    required_fixations = {
        "simulation_id",
        "seed",
        "text_id",
        "fixation_counter",
        "word_id",
        "word",
        "fixation_duration",
        "saccade_type",
        "attentional_width",
        "eye_position",
    }
    missing_passages = sorted(required_passages - set(passages.columns))
    missing_fixations = sorted(required_fixations - set(ob1_fixations.columns))
    if missing_passages:
        raise ValueError(
            f"Passage table is missing columns: {missing_passages}"
        )
    if missing_fixations:
        raise ValueError(
            f"OB1 fixation table is missing columns: {missing_fixations}"
        )
    skews = tuple(float(value) for value in attention_skews)
    if not skews or len(skews) != len(set(skews)):
        raise ValueError("OB1 attention skews must be nonempty and unique")
    if any(value < 1 for value in skews):
        raise ValueError("OB1 attention skews must be at least one")

    grid = unique_et1_token_grid(token_values)
    passage_lookup = {
        int(row.passage_id_zero_based): str(row.passage_text)
        for row in passages.itertuples()
    }
    fixation_passages = set(ob1_fixations["text_id"].astype(int).unique())
    if fixation_passages != set(passage_lookup):
        raise ValueError(
            "OB1 fixation passages do not match the canonical passage grid"
        )
    if set(grid["passage_id_zero_based"].unique()) != set(passage_lookup):
        raise ValueError(
            "ET1 token passages do not match the canonical passage grid"
        )

    sparse_profiles = {
        skew: {
            passage_id: defaultdict(float)
            for passage_id in passage_lookup
        }
        for skew in skews
    }
    fixation_count = 0
    duration_total = 0.0
    widths_used = []
    characters_per_token_values = []
    visible_token_counts = []
    anchor_token_indices = set()
    punctuation_only_token_count = 0
    minimum_offset = 0
    maximum_offset = 0

    for passage_id in sorted(passage_lookup):
        passage_tokens = grid.loc[
            grid["passage_id_zero_based"].eq(passage_id)
        ].copy()
        geometry, clean_words, word_starts = build_t5_letter_geometry(
            passage_lookup[passage_id],
            passage_tokens,
        )
        clean_character_count = sum(len(word) for word in clean_words)
        token_count = len(geometry)
        if clean_character_count < 1 or token_count < 1:
            raise ValueError("Passage has no clean characters or T5 tokens")
        characters_per_token = clean_character_count / token_count
        characters_per_token_values.append(characters_per_token)
        punctuation_only_token_count += int(
            geometry["punctuation_only_token"].sum()
        )
        passage_fixations = ob1_fixations.loc[
            ob1_fixations["text_id"].astype(int).eq(passage_id)
        ].sort_values(["simulation_id", "fixation_counter"])
        if passage_fixations.empty:
            raise ValueError(
                f"OB1 has no fixations for passage {passage_id}"
            )
        passage_total_weight = 0.0
        for fixation in passage_fixations.itertuples():
            word_id = int(fixation.word_id)
            if word_id < 0 or word_id >= len(clean_words):
                raise ValueError("OB1 fixation word ID is outside the passage")
            if clean_ob1_word(fixation.word) != clean_words[word_id]:
                raise ValueError("OB1 fixation word does not match the passage")
            window_start = max(0, word_id - OB1_WINDOW_WORDS_LEFT)
            window_end = min(
                len(clean_words) - 1,
                word_id + OB1_WINDOW_WORDS_RIGHT,
            )
            visible = geometry.loc[
                geometry["word_id_zero_based"].between(
                    window_start,
                    window_end,
                )
            ].copy()
            anchor_candidates = visible.loc[
                visible["word_id_zero_based"].eq(word_id)
            ]
            if visible.empty or anchor_candidates.empty:
                raise ValueError("OB1 fixation has no visible or anchor token")
            local_eye_position = float(fixation.eye_position)
            if not math.isfinite(local_eye_position):
                raise ValueError("OB1 eye position must be finite")
            global_eye_position = (
                float(word_starts[window_start]) + local_eye_position
            )
            anchor_distance = (
                anchor_candidates["ob1_letter_center"]
                - global_eye_position
            ).abs()
            anchor_index = int(
                anchor_candidates.loc[
                    anchor_distance.idxmin(),
                    "token_index",
                ]
            )
            anchor_token_indices.add((passage_id, anchor_index))
            relative_offsets = (
                visible["token_index"].to_numpy(dtype=int) - anchor_index
            )
            eccentricity = (
                visible["ob1_letter_center"].to_numpy(dtype=float)
                - global_eye_position
            )
            minimum_offset = min(
                minimum_offset,
                int(relative_offsets.min()),
            )
            maximum_offset = max(
                maximum_offset,
                int(relative_offsets.max()),
            )
            visible_token_counts.append(len(visible))
            width = effective_ob1_attention_width(
                fixation.attentional_width,
                fixation.saccade_type,
            )
            fixation_duration = float(fixation.fixation_duration)
            if not math.isfinite(fixation_duration) or fixation_duration <= 0:
                raise ValueError("OB1 fixation duration must be positive")
            fixation_weight = (
                fixation_duration
                if fixation_weighting == "duration"
                else 1.0
            )
            for skew in skews:
                weights = ob1_attention_weight(
                    eccentricity,
                    width,
                    skew,
                    profile_component,
                )
                weights = weights / weights.sum()
                for offset, value in zip(relative_offsets, weights):
                    sparse_profiles[skew][passage_id][int(offset)] += (
                        fixation_weight * float(value)
                    )
            passage_total_weight += fixation_weight
            fixation_count += 1
            duration_total += fixation_duration
            widths_used.append(width)

        for skew in skews:
            sparse_profiles[skew][passage_id] = {
                offset: value / passage_total_weight
                for offset, value in sparse_profiles[skew][
                    passage_id
                ].items()
            }
    support = np.arange(minimum_offset, maximum_offset + 1, dtype=int)
    reference_profiles = {skew: {} for skew in skews}
    for skew in skews:
        for passage_id, sparse in sparse_profiles[skew].items():
            profile = dictionary_profile(sparse, support)
            reference_profiles[skew][passage_id] = {
                int(offset): float(value)
                for offset, value in zip(support, profile)
            }
    audit = {
        "fixation_count": int(fixation_count),
        "fixation_duration_total_ms": float(duration_total),
        "fixation_weighting": fixation_weighting,
        "passage_count": int(len(passage_lookup)),
        "simulation_count": int(
            ob1_fixations["simulation_id"].nunique()
        ),
        "seed_count": int(ob1_fixations["seed"].nunique()),
        "minimum_relative_token_offset": int(minimum_offset),
        "maximum_relative_token_offset": int(maximum_offset),
        "minimum_visible_t5_token_count": int(min(visible_token_counts)),
        "maximum_visible_t5_token_count": int(max(visible_token_counts)),
        "mean_visible_t5_token_count": float(
            np.mean(visible_token_counts)
        ),
        "distinct_anchor_t5_tokens": int(len(anchor_token_indices)),
        "punctuation_only_t5_token_count": int(
            punctuation_only_token_count
        ),
        "minimum_effective_attention_width": float(min(widths_used)),
        "maximum_effective_attention_width": float(max(widths_used)),
        "minimum_clean_characters_per_t5_token": float(
            min(characters_per_token_values)
        ),
        "maximum_clean_characters_per_t5_token": float(
            max(characters_per_token_values)
        ),
        "mean_clean_characters_per_t5_token": float(
            np.mean(characters_per_token_values)
        ),
        "visible_words_left": OB1_WINDOW_WORDS_LEFT,
        "visible_words_right": OB1_WINDOW_WORDS_RIGHT,
        "residual_attention": OB1_RESIDUAL_ATTENTION,
        "profile_component": profile_component,
        "residual_attention_included": profile_component == "full",
        "special_tokens_included": False,
        "projection_method": (
            "fixation-onset OB1 letter attention evaluated at native T5 "
            "token centers; relative offsets are centered on the nearest "
            "T5 token in the fixated word"
        ),
    }
    return reference_profiles, support, audit


def dictionary_profile(
    values: dict[int, float],
    support: np.ndarray,
) -> np.ndarray:
    """Convert one sparse offset profile into a normalized dense array."""
    profile = np.asarray(
        [float(values.get(int(offset), 0.0)) for offset in support],
        dtype=float,
    )
    total = float(profile.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Offset profile has invalid mass")
    return profile / total


def candidate_profile(
    support: np.ndarray,
    method: str,
    sigma_left: float | None = None,
    sigma_right: float | None = None,
) -> np.ndarray:
    """Build one candidate kernel on the common relative-token support."""
    if method == "raw_delta":
        return delta_weights(support)
    if sigma_left is None or sigma_right is None:
        raise ValueError("Gaussian candidate requires both sigmas")
    return gaussian_weights(support, sigma_left, sigma_right)


def fit_ob1_gaussian_prior(
    reference_profile: np.ndarray,
    support: np.ndarray,
) -> dict:
    """Fit a fixed asymmetric Gaussian to one projected OB1 profile."""
    reference = np.asarray(reference_profile, dtype=float)
    reference = reference / reference.sum()

    def objective(parameters: np.ndarray) -> float:
        sigma_left = math.exp(float(parameters[0]))
        sigma_right = sigma_left * math.exp(float(parameters[1]))
        candidate = candidate_profile(
            support,
            "fixed_ob1_gaussian",
            float(sigma_left),
            float(sigma_right),
        )
        return float(jensenshannon(reference, candidate, base=2.0) ** 2)

    result = minimize(
        objective,
        x0=np.asarray([math.log(1.0), math.log(4.0)]),
        method="L-BFGS-B",
        bounds=[
            (math.log(0.05), math.log(30.0)),
            (math.log(1.001), math.log(20.0)),
        ],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"OB1 Gaussian fit failed: {result.message}")
    sigma_left = math.exp(float(result.x[0]))
    sigma_right = sigma_left * math.exp(float(result.x[1]))
    return {
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "right_left_ratio": float(sigma_right / sigma_left),
        "fit_js_divergence": float(result.fun),
        "optimizer": "L-BFGS-B",
        "optimizer_iterations": int(result.nit),
        "fit_objective": "Jensen-Shannon divergence in T5 token-offset space",
    }


def profile_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    """Compute profile shape metrics against projected OB1 attention."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    rho = float(spearmanr(reference, candidate).statistic)
    if not math.isfinite(rho):
        raise ValueError("Attention-profile Spearman is not finite")
    return {
        "profile_spearman": rho,
        "js_divergence": float(
            jensenshannon(reference, candidate, base=2.0) ** 2
        ),
        "token_offset_wasserstein": float(
            wasserstein_distance(
                support,
                support,
                u_weights=reference,
                v_weights=candidate,
            )
        ),
    }


def summarize_profile_metrics(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap method means over the 55 Provo passages."""
    rng = np.random.default_rng(seed)
    records = []
    for (skew, method), group in passage_metrics.groupby(
        ["ob1_attention_skew", "method"],
        sort=True,
    ):
        record = {
            "ob1_attention_skew": float(skew),
            "method": method,
            "display_name": PROFILE_DISPLAY_NAMES[method],
            "passages": int(len(group)),
        }
        for metric in PROFILE_METRIC_DIRECTIONS:
            mean, low, high = bootstrap_mean(
                group[metric].to_numpy(),
                bootstrap_samples,
                rng,
            )
            record[metric] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        records.append(record)
    return pd.DataFrame(records)


def profile_paired_contrasts(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired passage-level kernel-profile contrasts."""
    bootstrap_rng = np.random.default_rng(seed)
    permutation_rng = np.random.default_rng(seed)
    records = []
    for skew, skew_metrics in passage_metrics.groupby(
        "ob1_attention_skew",
        sort=True,
    ):
        for candidate_name, baseline_name in PROFILE_COMPARISONS:
            candidate = skew_metrics.loc[
                skew_metrics["method"].eq(candidate_name)
            ].set_index("passage_id_zero_based")
            baseline = skew_metrics.loc[
                skew_metrics["method"].eq(baseline_name)
            ].set_index("passage_id_zero_based")
            common = candidate.index.intersection(baseline.index)
            if len(common) < 2:
                raise ValueError("Profile contrast needs at least two passages")
            for metric, direction in PROFILE_METRIC_DIRECTIONS.items():
                if direction == "higher":
                    differences = (
                        candidate.loc[common, metric]
                        - baseline.loc[common, metric]
                    ).to_numpy()
                else:
                    differences = (
                        baseline.loc[common, metric]
                        - candidate.loc[common, metric]
                    ).to_numpy()
                mean, low, high = bootstrap_mean(
                    differences,
                    bootstrap_samples,
                    bootstrap_rng,
                )
                p_value = paired_sign_flip_pvalue(
                    differences,
                    bootstrap_samples,
                    permutation_rng,
                )
                records.append(
                    {
                        "ob1_attention_skew": float(skew),
                        "candidate": candidate_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "positive_means_improvement": True,
                        "passages": int(len(common)),
                        "mean_paired_improvement": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "permutation_p_two_sided": p_value,
                    }
                )
    return pd.DataFrame(records)


def compare_attention_profiles(
    passages: pd.DataFrame,
    token_values: pd.DataFrame,
    ob1_fixations: pd.DataFrame,
    sigma_record: dict,
    attention_skews: tuple[float, ...] = (3.0, 4.0),
    fixation_weighting: str = "duration",
    profile_component: str = "focused",
    bootstrap_samples: int = 10000,
    seed: int = 20260725,
) -> dict:
    """Run projected OB1 attention and Gaussian-kernel comparisons."""
    required_sigma_keys = {
        "checkpoint_id",
        "sigma_left",
        "sigma_right",
        "sigma_symmetric",
    }
    missing_sigma = sorted(required_sigma_keys - set(sigma_record))
    if missing_sigma:
        raise ValueError(f"Sigma record is missing keys: {missing_sigma}")
    learned_left = float(sigma_record["sigma_left"])
    learned_right = float(sigma_record["sigma_right"])
    symmetric_sigma = float(sigma_record["sigma_symmetric"])
    references, support, collection_audit = (
        project_ob1_attention_to_t5(
            passages,
            token_values,
            ob1_fixations,
            attention_skews,
            fixation_weighting,
            profile_component,
        )
    )
    profile_rows = []
    metric_rows = []
    fixed_prior_records = []

    for skew in sorted(references):
        passage_references = {
            passage_id: dictionary_profile(values, support)
            for passage_id, values in references[skew].items()
        }
        reference_global = np.average(
            np.stack(list(passage_references.values())),
            axis=0,
            weights=[
                float(
                    ob1_fixations.loc[
                        ob1_fixations["text_id"].astype(int).eq(
                            passage_id
                        ),
                        "fixation_duration",
                    ].sum()
                )
                if fixation_weighting == "duration"
                else int(
                    ob1_fixations["text_id"].astype(int).eq(
                        passage_id
                    ).sum()
                )
                for passage_id in passage_references
            ],
        )
        reference_global = reference_global / reference_global.sum()
        fixed_prior = fit_ob1_gaussian_prior(
            reference_global,
            support,
        )
        fixed_prior_records.append(
            {
                "ob1_attention_skew": float(skew),
                "checkpoint_id": sigma_record["checkpoint_id"],
                "profile_component": profile_component,
                "fixation_weighting": fixation_weighting,
                "projection_coordinate": "relative_native_t5_token_index",
                **fixed_prior,
            }
        )
        method_sigmas = {
            "raw_delta": (None, None),
            "width_matched_symmetric": (
                symmetric_sigma,
                symmetric_sigma,
            ),
            "fixed_ob1_gaussian": (
                fixed_prior["sigma_left"],
                fixed_prior["sigma_right"],
            ),
            "learned_asymmetric": (
                learned_left,
                learned_right,
            ),
        }
        global_candidates = {}
        for method, (left, right) in method_sigmas.items():
            global_candidates[method] = candidate_profile(
                support,
                method,
                left,
                right,
            )
        for index, offset in enumerate(support):
            profile_rows.append(
                {
                    "ob1_attention_skew": float(skew),
                    "relative_t5_token_offset": int(offset),
                    "ob1_attention_profile": float(
                        reference_global[index]
                    ),
                    **{
                        method: float(profile[index])
                        for method, profile in global_candidates.items()
                    },
                }
            )
        for passage_id, reference in passage_references.items():
            for method, (left, right) in method_sigmas.items():
                candidate = candidate_profile(
                    support,
                    method,
                    left,
                    right,
                )
                metric_rows.append(
                    {
                        "checkpoint_id": sigma_record["checkpoint_id"],
                        "ob1_attention_skew": float(skew),
                        "passage_id_zero_based": int(passage_id),
                        "method": method,
                        **profile_metrics(
                            reference,
                            candidate,
                            support,
                        ),
                    }
                )

    passage_metrics = pd.DataFrame(metric_rows)
    result_table = summarize_profile_metrics(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    contrasts = profile_paired_contrasts(
        passage_metrics,
        bootstrap_samples,
        seed,
    )
    audit = {
        **collection_audit,
        "checkpoint_id": sigma_record["checkpoint_id"],
        "learned_sigma_left": learned_left,
        "learned_sigma_right": learned_right,
        "learned_right_left_ratio": learned_right / learned_left,
        "width_matched_symmetric_sigma": symmetric_sigma,
        "attention_skews": [
            float(value) for value in sorted(references)
        ],
        "attention_skew_trajectory_policy": (
            "reweight saved fixation geometries without rerunning OB1; "
            "the vendored simulation trajectory uses attention_skew=3"
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "reference_coordinate": "OB1 cleaned-letter coordinates",
        "comparison_coordinate": "relative native T5 token index",
        "reference_profile": (
            f"OB1 fixation-onset {profile_component} attention component; "
            "excludes acuity, within-fixation attention shifts, lexical "
            "activation, saccade control, and final TVT"
        ),
        "attention_width_update": (
            "regression: max(recorded-1,3); otherwise min(recorded+0.5,5)"
        ),
        "fixed_prior_fit_scope": (
            "all projected OB1 fixations; its alignment to the same OB1 "
            "profile is descriptive, not held-out validation"
        ),
    }
    attention_profiles = pd.DataFrame(profile_rows)
    return {
        "attention_profiles": attention_profiles,
        "directionality": summarize_directionality(attention_profiles),
        "passage_metrics": passage_metrics,
        "result_table": result_table,
        "contrasts": contrasts,
        "fixed_priors": fixed_prior_records,
        "audit": audit,
    }


def write_attention_profile_outputs(
    output_dir: Path,
    artifacts: dict,
) -> None:
    """Write attention-profile tables, fixed priors, and provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts["attention_profiles"].to_csv(
        output_dir / "kernel_profiles.csv",
        index=False,
    )
    artifacts["directionality"].to_csv(
        output_dir / "kernel_directionality.csv",
        index=False,
    )
    artifacts["passage_metrics"].to_csv(
        output_dir / "kernel_alignment_by_passage.csv",
        index=False,
    )
    artifacts["result_table"].to_csv(
        output_dir / "kernel_alignment_result_table.csv",
        index=False,
    )
    artifacts["contrasts"].to_csv(
        output_dir / "kernel_alignment_contrasts.csv",
        index=False,
    )
    with (output_dir / "fixed_ob1_priors.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            artifacts["fixed_priors"],
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    with (output_dir / "attention_profile_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            artifacts["audit"],
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    plot_attention_profiles(
        artifacts["attention_profiles"],
        output_dir / "kernel_profiles.png",
    )


def plot_attention_profiles(
    profiles: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot projected OB1 attention and all four token-space kernels."""
    skews = sorted(profiles["ob1_attention_skew"].unique())
    figure, axes = plt.subplots(
        len(skews),
        1,
        figsize=(9, 4.2 * len(skews)),
        sharex=True,
        squeeze=False,
    )
    series = [
        ("ob1_attention_profile", "Projected OB1", 2.5),
        ("raw_delta", PROFILE_DISPLAY_NAMES["raw_delta"], 1.5),
        (
            "width_matched_symmetric",
            PROFILE_DISPLAY_NAMES["width_matched_symmetric"],
            1.5,
        ),
        (
            "fixed_ob1_gaussian",
            PROFILE_DISPLAY_NAMES["fixed_ob1_gaussian"],
            1.5,
        ),
        (
            "learned_asymmetric",
            PROFILE_DISPLAY_NAMES["learned_asymmetric"],
            1.8,
        ),
    ]
    for axis, skew in zip(axes[:, 0], skews):
        skew_profiles = profiles.loc[
            profiles["ob1_attention_skew"].eq(skew)
        ].sort_values("relative_t5_token_offset")
        for column, label, width in series:
            axis.plot(
                skew_profiles["relative_t5_token_offset"],
                skew_profiles[column],
                marker="o",
                markersize=3,
                linewidth=width,
                label=label,
            )
        axis.axvline(0, color="black", linewidth=0.8, alpha=0.45)
        axis.set_title(f"OB1 attention skew = {skew:g}")
        axis.set_ylabel("Normalized mass")
        axis.legend(frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("Relative native T5 token offset")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def summarize_directionality(profiles: pd.DataFrame) -> pd.DataFrame:
    """Summarize left, center, and right mass for every compared profile."""
    profile_columns = [
        "ob1_attention_profile",
        *PROFILE_METHODS,
    ]
    records = []
    for skew, skew_profiles in profiles.groupby(
        "ob1_attention_skew",
        sort=True,
    ):
        offsets = skew_profiles["relative_t5_token_offset"]
        for method in profile_columns:
            left = float(skew_profiles.loc[offsets.lt(0), method].sum())
            center = float(skew_profiles.loc[offsets.eq(0), method].sum())
            right = float(skew_profiles.loc[offsets.gt(0), method].sum())
            noncenter = left + right
            records.append(
                {
                    "ob1_attention_skew": float(skew),
                    "method": method,
                    "left_mass": left,
                    "center_mass": center,
                    "right_mass": right,
                    "right_minus_left_mass": right - left,
                    "right_share_of_noncenter_mass": (
                        right / noncenter if noncenter > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)

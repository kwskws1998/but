"""Aggregate a 12-checkpoint Provo attention-profile robustness analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COGNITIVE_ROOT = HERE.parent
REPOSITORY_ROOT = COGNITIVE_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cognitive_model_comparsion.src.evaluate import (
    bootstrap_mean,
    paired_sign_flip_pvalue,
)


EXPECTED_OB1_SHA256 = (
    "a8d84e1c264d23e7ab9efac20c1a727e08d16208be8c475ccad9b096fc1fb647"
)
EXPECTED_ET1_SHA256 = (
    "e2bd70e0d9fee9e956639509642e231c15786ac9b148a6fdcaa2b53e66de8fbd"
)
EXPECTED_OB1_MANIFEST_SHA256 = (
    "65f2af0c1a351006b18ebef514611d0685bbdc6bdbd2163108392712927071bb"
)
EXPECTED_SIGMA_SHA256 = (
    "3ced76e695053211c65deb38be04c588452418574ff3b78eaa0227cbc7cc3504"
)
METRIC_DIRECTIONS = {
    "profile_spearman": "higher",
    "js_divergence": "lower",
    "hellinger_distance": "lower",
    "total_variation_distance": "lower",
}
METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "support_centered_sd_symmetric",
    "support_centered_sd_ratio4",
    "mirrored_learned",
    "learned_asymmetric",
)
EXPECTED_SOURCE_METHODS = (
    "raw_delta",
    "fixed_symmetric_sigma1",
    "rms_side_scale_symmetric",
    "fixed_ratio4_same_rms",
    "support_centered_sd_symmetric",
    "support_centered_sd_ratio4",
    "mirrored_learned",
    "learned_asymmetric",
)
METHOD_LABELS = {
    "raw_delta": "No redistribution",
    "fixed_symmetric_sigma1": "Fixed symmetric sigma=1",
    "support_centered_sd_symmetric": (
        "Symmetric, matched centered token-offset SD"
    ),
    "support_centered_sd_ratio4": (
        "Fixed 4:1, matched centered token-offset SD"
    ),
    "mirrored_learned": "Mirrored learned",
    "learned_asymmetric": "Learned asymmetric",
}
CONTRASTS = (
    ("learned_asymmetric", "raw_delta"),
    ("learned_asymmetric", "fixed_symmetric_sigma1"),
    (
        "learned_asymmetric",
        "support_centered_sd_symmetric",
    ),
    (
        "learned_asymmetric",
        "support_centered_sd_ratio4",
    ),
    (
        "support_centered_sd_ratio4",
        "support_centered_sd_symmetric",
    ),
    ("learned_asymmetric", "mirrored_learned"),
)
PROFILE_COLORS = {
    "ob1_attention_profile": "#2F8FF3",
    "fixed_symmetric_sigma1": "#F28E2B",
    "support_centered_sd_symmetric": "#8C8C8C",
    "support_centered_sd_ratio4": "#9467BD",
    "mirrored_learned": "#59A14F",
    "learned_asymmetric": "#D96AA7",
}
PROFILE_LABELS = {
    "ob1_attention_profile": "OB1 focused attention",
    **METHOD_LABELS,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(
    analysis_dir: Path,
    expected_checkpoints: int,
    expected_passages: int,
    expected_ob1_sha256: str,
    expected_et1_sha256: str,
    expected_ob1_manifest_sha256: str,
    expected_sigma_sha256: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Load complete profile artifacts and enforce the current-run contract."""
    paths = {
        "passages": analysis_dir / "kernel_alignment_by_passage.csv",
        "results": analysis_dir / "kernel_alignment_result_table.csv",
        "profiles": analysis_dir / "kernel_profiles.csv",
        "diagnostics": analysis_dir / "gaussian_parameter_diagnostics.csv",
        "audit": analysis_dir / "attention_profile_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing attention artifacts: {missing}")
    passage_metrics = pd.read_csv(paths["passages"])
    results = pd.read_csv(paths["results"])
    profiles = pd.read_csv(paths["profiles"])
    diagnostics = pd.read_csv(paths["diagnostics"])
    with paths["audit"].open(encoding="utf-8") as handle:
        audit = json.load(handle)

    expected_audit = {
        "checkpoint_count": expected_checkpoints,
        "passage_count": expected_passages,
        "ob1_fixations_sha256": expected_ob1_sha256,
        "et1_token_values_sha256": expected_et1_sha256,
        "ob1_worker_manifest_sha256": expected_ob1_manifest_sha256,
        "sigma_json_sha256": expected_sigma_sha256,
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
        raise ValueError(
            "Attention audit does not match the requested current-run "
            f"contract: {mismatches}"
        )
    checkpoint_ids = sorted(
        passage_metrics["checkpoint_id"].astype(str).unique()
    )
    if len(checkpoint_ids) != expected_checkpoints:
        raise ValueError(
            f"Expected {expected_checkpoints} checkpoints, "
            f"found {len(checkpoint_ids)}"
        )
    source_methods = set(passage_metrics["method"].astype(str))
    if source_methods != set(EXPECTED_SOURCE_METHODS):
        raise ValueError(
            "Passage metric methods differ from the complete expected set: "
            f"expected {sorted(EXPECTED_SOURCE_METHODS)}, "
            f"found {sorted(source_methods)}"
        )
    required_columns = {
        "checkpoint_id",
        "ob1_attention_skew",
        "passage_id_zero_based",
        "method",
        "source_accuracy",
        "learned_sigma_left",
        "learned_sigma_right",
        "learned_right_left_ratio",
        "candidate_support_policy",
        "profile_component",
        "fixation_weighting",
        "actual_et1_trt_magnitudes_used",
        "trajectory_attention_skew",
        "requested_skew_matches_trajectory",
        *METRIC_DIRECTIONS,
    }
    missing_columns = sorted(required_columns - set(passage_metrics.columns))
    if missing_columns:
        raise ValueError(
            f"Passage metrics are missing columns: {missing_columns}"
        )
    if passage_metrics[list(required_columns)].isna().any().any():
        raise ValueError("Passage metrics contain missing required values")
    selected = passage_metrics.loc[
        passage_metrics["method"].isin(METHODS)
    ].copy()
    duplicate_keys = [
        "checkpoint_id",
        "ob1_attention_skew",
        "passage_id_zero_based",
        "method",
    ]
    if selected.duplicated(duplicate_keys).any():
        raise ValueError("Passage metric coordinates are not unique")
    counts = selected.groupby(
        ["checkpoint_id", "ob1_attention_skew", "method"],
        sort=True,
    )["passage_id_zero_based"].nunique()
    if not bool(counts.eq(expected_passages).all()):
        raise ValueError(
            "Every checkpoint, skew, and method must contain the same "
            f"{expected_passages} passages"
        )
    if set(selected["ob1_attention_skew"].astype(float)) != {3.0, 4.0}:
        raise ValueError("Expected separate OB1 attention skew 3 and 4 results")
    source_key_columns = [
        "checkpoint_id",
        "ob1_attention_skew",
        "passage_id_zero_based",
        "method",
    ]
    expected_source_rows = (
        expected_checkpoints
        * 2
        * expected_passages
        * len(EXPECTED_SOURCE_METHODS)
    )
    if len(passage_metrics) != expected_source_rows:
        raise ValueError(
            f"Expected {expected_source_rows} passage metric rows, "
            f"found {len(passage_metrics)}"
        )
    if passage_metrics.duplicated(source_key_columns).any():
        raise ValueError("Complete passage metric coordinates are not unique")
    source_counts = passage_metrics.groupby(
        ["checkpoint_id", "ob1_attention_skew", "method"],
        sort=True,
    )["passage_id_zero_based"].nunique()
    expected_source_groups = (
        expected_checkpoints * 2 * len(EXPECTED_SOURCE_METHODS)
    )
    if len(source_counts) != expected_source_groups:
        raise ValueError(
            f"Expected {expected_source_groups} source method grids, "
            f"found {len(source_counts)}"
        )
    if not source_counts.eq(expected_passages).all():
        raise ValueError(
            "Every source checkpoint, skew, and method must contain "
            f"{expected_passages} passages"
        )
    expected_passage_ids = set(range(expected_passages))
    passage_sets = passage_metrics.groupby(
        ["checkpoint_id", "ob1_attention_skew", "method"],
        sort=True,
    )["passage_id_zero_based"].agg(lambda values: set(map(int, values)))
    if not all(
        passage_ids == expected_passage_ids
        for passage_ids in passage_sets
    ):
        raise ValueError(
            "Every source method grid must use passage IDs 0 through "
            f"{expected_passages - 1}"
        )
    expected_result_rows = expected_checkpoints * 2 * len(
        EXPECTED_SOURCE_METHODS
    )
    if len(results) != expected_result_rows:
        raise ValueError(
            f"Expected {expected_result_rows} result rows, found {len(results)}"
        )
    result_methods = set(results["method"].astype(str))
    if result_methods != set(EXPECTED_SOURCE_METHODS):
        raise ValueError("Result-table methods differ from passage metrics")
    result_keys = ["checkpoint_id", "ob1_attention_skew", "method"]
    if results.duplicated(result_keys).any():
        raise ValueError("Result-table coordinates are not unique")
    metric_values = passage_metrics[list(METRIC_DIRECTIONS)].to_numpy(
        dtype=float
    )
    if not np.isfinite(metric_values).all():
        raise ValueError("Passage metrics contain non-finite values")
    spearman = passage_metrics["profile_spearman"].to_numpy(dtype=float)
    if np.any((spearman < -1.0) | (spearman > 1.0)):
        raise ValueError("Spearman values must lie in [-1, 1]")
    for metric in (
        "js_divergence",
        "hellinger_distance",
        "total_variation_distance",
    ):
        values = passage_metrics[metric].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{metric} values must lie in [0, 1]")
    profile_keys = [
        "checkpoint_id",
        "ob1_attention_skew",
        "relative_t5_token_offset",
    ]
    if profiles.duplicated(profile_keys).any():
        raise ValueError("Kernel-profile coordinates are not unique")
    profile_group_counts = profiles.groupby(
        ["checkpoint_id", "ob1_attention_skew"],
        sort=True,
    )["relative_t5_token_offset"].nunique()
    if (
        len(profile_group_counts) != expected_checkpoints * 2
        or profile_group_counts.nunique() != 1
        or int(profile_group_counts.iloc[0]) != 26
    ):
        raise ValueError(
            "Every checkpoint and skew must use the exact 26-offset grid"
        )
    row_contract_columns = {
        "candidate_support_policy": "fixation_matched",
        "profile_component": "focused",
        "fixation_weighting": "duration",
        "actual_et1_trt_magnitudes_used": False,
        "trajectory_attention_skew": 3.0,
    }
    for column, expected in row_contract_columns.items():
        values = passage_metrics[column].drop_duplicates().tolist()
        if values != [expected]:
            raise ValueError(
                f"Passage metric {column} must equal {expected}, "
                f"found {values}"
            )
    skew_roles = (
        passage_metrics[
            [
                "ob1_attention_skew",
                "requested_skew_matches_trajectory",
            ]
        ]
        .drop_duplicates()
        .sort_values("ob1_attention_skew")
    )
    expected_roles = [(3.0, True), (4.0, False)]
    found_roles = [
        (
            float(row.ob1_attention_skew),
            bool(row.requested_skew_matches_trajectory),
        )
        for row in skew_roles.itertuples(index=False)
    ]
    if found_roles != expected_roles:
        raise ValueError(
            f"Unexpected trajectory-match roles: {found_roles}"
        )
    centered_matches = audit.get("support_centered_sd_matches")
    if not isinstance(centered_matches, dict):
        raise ValueError("Centered-SD match audit is missing")
    if set(centered_matches) != set(checkpoint_ids):
        raise ValueError("Centered-SD match audit checkpoint IDs differ")
    for checkpoint_id, match_record in centered_matches.items():
        for control in ("symmetric_ratio1", "fixed_ratio4"):
            fit = match_record.get(control)
            if not isinstance(fit, dict):
                raise ValueError(
                    f"Missing {control} centered-SD fit for {checkpoint_id}"
                )
            error = float(fit.get("absolute_match_error", math.inf))
            if not math.isfinite(error) or error > 1e-7:
                raise ValueError(
                    f"Centered-SD match error for {checkpoint_id} "
                    f"{control} is {error}"
                )
    diagnostic_methods = {
        "fixed_symmetric_sigma1",
        "fixed_same_rms_symmetric",
        "fixed_ratio4_same_rms",
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
        "mirrored_learned",
        "learned_fixed",
    }
    diagnostic_columns = {
        "checkpoint_id",
        "ob1_attention_skew",
        "model_id",
        "realized_mean_token_offset",
        "realized_centered_token_offset_sd",
        "realized_rms_token_displacement",
        "fitted_to_same_ob1_profile",
    }
    missing_diagnostic_columns = sorted(
        diagnostic_columns - set(diagnostics.columns)
    )
    if missing_diagnostic_columns:
        raise ValueError(
            "Parameter diagnostics are missing columns: "
            f"{missing_diagnostic_columns}"
        )
    selected_diagnostics = diagnostics.loc[
        diagnostics["model_id"].isin(diagnostic_methods)
    ].copy()
    diagnostic_keys = [
        "checkpoint_id",
        "ob1_attention_skew",
        "model_id",
    ]
    if selected_diagnostics.duplicated(diagnostic_keys).any():
        raise ValueError("Parameter diagnostic coordinates are not unique")
    diagnostic_counts = selected_diagnostics.groupby(
        "model_id",
        sort=True,
    ).size()
    if (
        set(diagnostic_counts.index) != diagnostic_methods
        or not diagnostic_counts.eq(expected_checkpoints * 2).all()
    ):
        raise ValueError("Parameter diagnostic method grids are incomplete")
    diagnostic_values = selected_diagnostics[
        [
            "realized_mean_token_offset",
            "realized_centered_token_offset_sd",
            "realized_rms_token_displacement",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(diagnostic_values).all():
        raise ValueError("Parameter diagnostics contain non-finite values")
    decomposition_error = np.abs(
        diagnostic_values[:, 2] ** 2
        - diagnostic_values[:, 1] ** 2
        - diagnostic_values[:, 0] ** 2
    )
    if float(decomposition_error.max()) > 1e-8:
        raise ValueError("RMS, centered SD, and mean fail their decomposition")
    if selected_diagnostics["fitted_to_same_ob1_profile"].astype(bool).any():
        raise ValueError("Selected width diagnostics must not be OB1-fitted")
    diagnostic_sd = selected_diagnostics.pivot(
        index=["checkpoint_id", "ob1_attention_skew"],
        columns="model_id",
        values="realized_centered_token_offset_sd",
    )
    learned_sd = diagnostic_sd["learned_fixed"].to_numpy(dtype=float)
    for method in (
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
    ):
        if not np.allclose(
            diagnostic_sd[method].to_numpy(dtype=float),
            learned_sd,
            rtol=1e-7,
            atol=1e-8,
        ):
            raise ValueError(
                f"{method} does not match learned centered token-offset SD"
            )
    return selected, results, profiles, selected_diagnostics, audit


def checkpoint_metadata(passage_metrics: pd.DataFrame) -> pd.DataFrame:
    """Extract one immutable sigma and accuracy row per checkpoint."""
    columns = [
        "checkpoint_id",
        "source_accuracy",
        "learned_sigma_left",
        "learned_sigma_right",
        "learned_right_left_ratio",
        "learned_support_centered_token_sd",
        "support_centered_sd_symmetric_sigma",
        "support_centered_sd_ratio4_sigma_left",
        "support_centered_sd_ratio4_sigma_right",
    ]
    available = [column for column in columns if column in passage_metrics]
    metadata = passage_metrics[available].drop_duplicates()
    if metadata["checkpoint_id"].duplicated().any():
        raise ValueError("Checkpoint metadata change across metric rows")
    metadata = metadata.sort_values("checkpoint_id").reset_index(drop=True)
    metadata["rightward_learned"] = metadata[
        "learned_sigma_right"
    ].gt(metadata["learned_sigma_left"])
    return metadata


def aggregate_width_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize directional mean, centered width, and source-centered RMS."""
    value_columns = [
        "realized_mean_token_offset",
        "realized_centered_token_offset_sd",
        "realized_rms_token_displacement",
    ]
    records = []
    for (skew, model_id), group in diagnostics.groupby(
        ["ob1_attention_skew", "model_id"],
        sort=True,
    ):
        record = {
            "ob1_attention_skew": float(skew),
            "model_id": model_id,
            "checkpoint_count": int(group["checkpoint_id"].nunique()),
        }
        for column in value_columns:
            values = group[column].to_numpy(dtype=float)
            record[f"mean_{column}"] = float(values.mean())
            record[f"sd_across_checkpoints_{column}"] = float(
                np.std(values, ddof=1)
            )
        records.append(record)
    return pd.DataFrame(records)


def checkpoint_passage_matrix(
    values: pd.DataFrame,
    value_column: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Build a complete checkpoint-by-passage numeric matrix."""
    matrix = values.pivot(
        index="checkpoint_id",
        columns="passage_id_zero_based",
        values=value_column,
    ).sort_index(axis=0).sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError(
            f"Incomplete checkpoint-by-passage matrix for {value_column}"
        )
    numeric = matrix.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite matrix values for {value_column}")
    return (
        numeric,
        matrix.index.astype(str).tolist(),
        matrix.columns.astype(int).tolist(),
    )


def crossed_bootstrap_mean(
    matrix: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> tuple[float, float, float]:
    """Bootstrap checkpoints and passages independently with replacement."""
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or min(values.shape) < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Crossed bootstrap needs a finite matrix with at least two "
            "checkpoints and passages"
        )
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    draws = np.empty(samples, dtype=float)
    checkpoint_count, passage_count = values.shape
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        size = stop - start
        checkpoint_indices = rng.integers(
            0,
            checkpoint_count,
            size=(size, checkpoint_count),
        )
        passage_indices = rng.integers(
            0,
            passage_count,
            size=(size, passage_count),
        )
        resampled = values[
            checkpoint_indices[:, :, None],
            passage_indices[:, None, :],
        ]
        draws[start:stop] = resampled.mean(axis=(1, 2))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def aggregate_method_metrics(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Average checkpoints within passage and bootstrap the 55 passages."""
    passage_rng = np.random.default_rng(seed)
    crossed_rng = np.random.default_rng(seed + 1)
    records = []
    for (skew, method), group in passage_metrics.groupby(
        ["ob1_attention_skew", "method"],
        sort=True,
    ):
        checkpoint_count = int(group["checkpoint_id"].nunique())
        passage_count = int(group["passage_id_zero_based"].nunique())
        for metric in METRIC_DIRECTIONS:
            matrix, _, _ = checkpoint_passage_matrix(group, metric)
            checkpoint_values = matrix.mean(axis=1)
            passage_values = matrix.mean(axis=0)
            mean, passage_low, passage_high = bootstrap_mean(
                passage_values,
                bootstrap_samples,
                passage_rng,
            )
            crossed_mean, crossed_low, crossed_high = (
                crossed_bootstrap_mean(
                    matrix,
                    bootstrap_samples,
                    crossed_rng,
                )
            )
            if not math.isclose(
                mean,
                crossed_mean,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("Aggregation point estimates disagree")
            records.append(
                {
                    "ob1_attention_skew": float(skew),
                    "method": method,
                    "display_name": METHOD_LABELS[method],
                    "metric": metric,
                    "checkpoint_count": checkpoint_count,
                    "passage_count": passage_count,
                    "equal_checkpoint_mean": mean,
                    "passage_bootstrap_ci_low": passage_low,
                    "passage_bootstrap_ci_high": passage_high,
                    "crossed_bootstrap_ci_low": crossed_low,
                    "crossed_bootstrap_ci_high": crossed_high,
                    "checkpoint_sd": float(
                        np.std(checkpoint_values, ddof=1)
                    ),
                    "checkpoint_median": float(
                        np.median(checkpoint_values)
                    ),
                    "checkpoint_q1": float(
                        np.quantile(checkpoint_values, 0.25)
                    ),
                    "checkpoint_q3": float(
                        np.quantile(checkpoint_values, 0.75)
                    ),
                    "aggregation_order": (
                        "point estimate averages all checkpoint-passage "
                        "cells equally; passage CI averages checkpoints "
                        "within passage before resampling passages; crossed "
                        "CI independently resamples checkpoints and passages"
                    ),
                }
            )
    return pd.DataFrame(records)


def paired_improvement(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Build one checkpoint-by-passage improvement table."""
    keys = [
        "checkpoint_id",
        "ob1_attention_skew",
        "passage_id_zero_based",
    ]
    merged = candidate[keys + [metric]].merge(
        baseline[keys + [metric]],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Candidate and baseline passage grids differ")
    merged = merged.drop(columns="_merge")
    if METRIC_DIRECTIONS[metric] == "higher":
        merged["improvement"] = (
            merged[f"{metric}_candidate"]
            - merged[f"{metric}_baseline"]
        )
    else:
        merged["improvement"] = (
            merged[f"{metric}_baseline"]
            - merged[f"{metric}_candidate"]
        )
    return merged


def aggregate_paired_contrasts(
    passage_metrics: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate within-checkpoint paired improvements without pseudoreplication."""
    passage_rng = np.random.default_rng(seed)
    crossed_rng = np.random.default_rng(seed + 1)
    permutation_rng = np.random.default_rng(seed + 2)
    aggregate_records = []
    checkpoint_records = []
    cell_records = []
    passage_records = []
    for skew in sorted(passage_metrics["ob1_attention_skew"].unique()):
        skew_metrics = passage_metrics.loc[
            passage_metrics["ob1_attention_skew"].eq(skew)
        ]
        for candidate_name, baseline_name in CONTRASTS:
            candidate = skew_metrics.loc[
                skew_metrics["method"].eq(candidate_name)
            ]
            baseline = skew_metrics.loc[
                skew_metrics["method"].eq(baseline_name)
            ]
            for metric in METRIC_DIRECTIONS:
                differences = paired_improvement(
                    candidate,
                    baseline,
                    metric,
                )
                matrix, checkpoint_ids, passage_ids = (
                    checkpoint_passage_matrix(
                        differences,
                        "improvement",
                    )
                )
                checkpoint_values = matrix.mean(axis=1)
                passage_values = matrix.mean(axis=0)
                mean, passage_low, passage_high = bootstrap_mean(
                    passage_values,
                    bootstrap_samples,
                    passage_rng,
                )
                crossed_mean, crossed_low, crossed_high = (
                    crossed_bootstrap_mean(
                        matrix,
                        bootstrap_samples,
                        crossed_rng,
                    )
                )
                if not math.isclose(
                    mean,
                    crossed_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "Paired aggregation point estimates disagree"
                    )
                passage_p_value = paired_sign_flip_pvalue(
                    passage_values,
                    bootstrap_samples,
                    permutation_rng,
                )
                checkpoint_p_value = paired_sign_flip_pvalue(
                    checkpoint_values,
                    1 << len(checkpoint_values),
                    permutation_rng,
                )
                aggregate_records.append(
                    {
                        "ob1_attention_skew": float(skew),
                        "candidate": candidate_name,
                        "candidate_name": METHOD_LABELS[candidate_name],
                        "baseline": baseline_name,
                        "baseline_name": METHOD_LABELS[baseline_name],
                        "metric": metric,
                        "positive_means_candidate_better": True,
                        "checkpoint_count": int(len(checkpoint_values)),
                        "passage_count": int(len(passage_values)),
                        "equal_checkpoint_mean_paired_improvement": mean,
                        "passage_bootstrap_ci_low": passage_low,
                        "passage_bootstrap_ci_high": passage_high,
                        "crossed_bootstrap_ci_low": crossed_low,
                        "crossed_bootstrap_ci_high": crossed_high,
                        "passage_sign_flip_p_two_sided": passage_p_value,
                        "checkpoint_exact_sign_flip_p_two_sided": (
                            checkpoint_p_value
                        ),
                        "checkpoint_sd": float(
                            np.std(checkpoint_values, ddof=1)
                        ),
                        "checkpoint_median": float(
                            np.median(checkpoint_values)
                        ),
                        "checkpoint_q1": float(
                            np.quantile(checkpoint_values, 0.25)
                        ),
                        "checkpoint_q3": float(
                            np.quantile(checkpoint_values, 0.75)
                        ),
                        "candidate_wins": int(
                            np.count_nonzero(checkpoint_values > 0)
                        ),
                        "candidate_ties": int(
                            np.count_nonzero(checkpoint_values == 0)
                        ),
                        "aggregation_order": (
                            "paired checkpoint-passage differences; equal "
                            "cell mean for point estimate; passage CI after "
                            "averaging checkpoints within passage; crossed "
                            "CI resamples both axes independently"
                        ),
                    }
                )
                for checkpoint_id, value in zip(
                    checkpoint_ids,
                    checkpoint_values,
                ):
                    checkpoint_records.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "ob1_attention_skew": float(skew),
                            "candidate": candidate_name,
                            "baseline": baseline_name,
                            "metric": metric,
                            "positive_means_candidate_better": True,
                            "mean_paired_improvement": float(value),
                        }
                    )
                for passage_id, value in zip(passage_ids, passage_values):
                    passage_records.append(
                        {
                            "passage_id_zero_based": passage_id,
                            "ob1_attention_skew": float(skew),
                            "candidate": candidate_name,
                            "baseline": baseline_name,
                            "metric": metric,
                            "positive_means_candidate_better": True,
                            "mean_checkpoint_paired_improvement": float(value),
                        }
                    )
                for row in differences.itertuples(index=False):
                    cell_records.append(
                        {
                            "checkpoint_id": row.checkpoint_id,
                            "passage_id_zero_based": (
                                row.passage_id_zero_based
                            ),
                            "ob1_attention_skew": float(skew),
                            "candidate": candidate_name,
                            "baseline": baseline_name,
                            "metric": metric,
                            "positive_means_candidate_better": True,
                            "paired_improvement": float(row.improvement),
                        }
                    )
    return (
        pd.DataFrame(aggregate_records),
        pd.DataFrame(checkpoint_records),
        pd.DataFrame(cell_records),
        pd.DataFrame(passage_records),
    )


def aggregate_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Average normalized kernel profiles across the supplied checkpoints."""
    profile_columns = [
        "ob1_attention_profile",
        "fixed_symmetric_sigma1",
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
        "mirrored_learned",
        "learned_asymmetric",
    ]
    missing = sorted(set(profile_columns) - set(profiles.columns))
    if missing:
        raise ValueError(f"Kernel profiles are missing columns: {missing}")
    records = []
    for (skew, offset), group in profiles.groupby(
        ["ob1_attention_skew", "relative_t5_token_offset"],
        sort=True,
    ):
        for method in profile_columns:
            values = group[method].to_numpy(dtype=float)
            records.append(
                {
                    "ob1_attention_skew": float(skew),
                    "relative_t5_token_offset": int(offset),
                    "method": method,
                    "display_name": PROFILE_LABELS[method],
                    "equal_checkpoint_mean": float(values.mean()),
                    "checkpoint_sd": float(np.std(values, ddof=1)),
                    "checkpoint_q025": float(np.quantile(values, 0.025)),
                    "checkpoint_q975": float(np.quantile(values, 0.975)),
                    "checkpoint_count": int(len(values)),
                }
            )
    return pd.DataFrame(records)


def plot_mean_profiles(profile_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot checkpoint-mean profiles with checkpoint-distribution ribbons."""
    skews = sorted(profile_summary["ob1_attention_skew"].unique())
    figure, axes = plt.subplots(
        len(skews),
        1,
        figsize=(10.5, 4.6 * len(skews)),
        sharex=True,
        squeeze=False,
    )
    plot_methods = (
        "ob1_attention_profile",
        "fixed_symmetric_sigma1",
        "support_centered_sd_symmetric",
        "support_centered_sd_ratio4",
        "learned_asymmetric",
    )
    for axis, skew in zip(axes[:, 0], skews):
        skew_rows = profile_summary.loc[
            profile_summary["ob1_attention_skew"].eq(skew)
        ]
        for method in plot_methods:
            rows = (
                skew_rows.loc[skew_rows["method"].eq(method)]
                .sort_values("relative_t5_token_offset")
            )
            x = rows["relative_t5_token_offset"].to_numpy(dtype=float)
            y = rows["equal_checkpoint_mean"].to_numpy(dtype=float)
            color = PROFILE_COLORS[method]
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.5,
                linewidth=2.0,
                color=color,
                label=PROFILE_LABELS[method],
            )
            if method in {
                "support_centered_sd_symmetric",
                "support_centered_sd_ratio4",
                "learned_asymmetric",
            }:
                axis.fill_between(
                    x,
                    rows["checkpoint_q025"].to_numpy(dtype=float),
                    rows["checkpoint_q975"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.13,
                )
        axis.axvline(0, color="black", linewidth=0.8, alpha=0.4)
        axis.set_title(
            f"OB1 attention skew={skew:g}; mean over 12 supplied checkpoints"
        )
        axis.set_ylabel("Normalized mass")
        axis.legend(frameon=False, fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Relative native T5-token offset")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_checkpoint_metric_improvements(
    checkpoint_contrasts: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot every primary metric improvement for all checkpoints and skews."""
    selected = checkpoint_contrasts.loc[
        checkpoint_contrasts["candidate"].eq("learned_asymmetric")
        & checkpoint_contrasts["baseline"].eq(
            "support_centered_sd_symmetric"
        )
    ].merge(
        metadata[
            [
                "checkpoint_id",
                "learned_right_left_ratio",
            ]
        ],
        on="checkpoint_id",
        how="left",
        validate="many_to_one",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRIC_DIRECTIONS:
        metric_rows = selected.loc[selected["metric"].eq(metric)]
        skews = sorted(metric_rows["ob1_attention_skew"].unique())
        figure, axes = plt.subplots(
            1,
            len(skews),
            figsize=(6.2 * len(skews), 4.8),
            squeeze=False,
        )
        for axis, skew in zip(axes[0], skews):
            rows = (
                metric_rows.loc[
                    metric_rows["ob1_attention_skew"].eq(skew)
                ]
                .sort_values("learned_right_left_ratio")
                .reset_index(drop=True)
            )
            colors = np.where(
                rows["mean_paired_improvement"].ge(0),
                "#59A14F",
                "#E15759",
            )
            axis.scatter(
                np.arange(len(rows)),
                rows["mean_paired_improvement"],
                c=colors,
                s=52,
            )
            axis.axhline(0, color="black", linewidth=1.0)
            axis.set_xticks(
                np.arange(len(rows)),
                rows["checkpoint_id"],
                rotation=70,
                ha="right",
                fontsize=8,
            )
            axis.set_title(f"OB1 attention skew={skew:g}")
            axis.set_ylabel(
                "Paired improvement\n"
                "(positive = learned asymmetric closer to OB1)"
            )
        figure.suptitle(
            f"{metric}: learned asymmetric vs centered-SD-matched symmetric"
        )
        figure.tight_layout()
        figure.savefig(
            output_dir / f"checkpoint_improvement_{metric}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_width_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot directional mean separately from centered dispersion."""
    model_labels = {
        "learned_fixed": "Learned asymmetric",
        "support_centered_sd_symmetric": "Centered-SD symmetric",
        "support_centered_sd_ratio4": "Centered-SD 4:1",
        "fixed_same_rms_symmetric": "Parameter-RMS symmetric",
        "fixed_ratio4_same_rms": "Parameter-RMS 4:1",
    }
    model_colors = {
        "learned_fixed": "#D96AA7",
        "support_centered_sd_symmetric": "#8C8C8C",
        "support_centered_sd_ratio4": "#9467BD",
        "fixed_same_rms_symmetric": "#4E79A7",
        "fixed_ratio4_same_rms": "#F28E2B",
    }
    selected = diagnostics.loc[
        diagnostics["model_id"].isin(model_labels)
    ]
    skews = sorted(selected["ob1_attention_skew"].unique())
    figure, axes = plt.subplots(
        1,
        len(skews),
        figsize=(6.2 * len(skews), 5.2),
        squeeze=False,
    )
    for axis, skew in zip(axes[0], skews):
        rows = selected.loc[selected["ob1_attention_skew"].eq(skew)]
        for model_id, label in model_labels.items():
            model_rows = rows.loc[rows["model_id"].eq(model_id)]
            x = model_rows["realized_mean_token_offset"].to_numpy(
                dtype=float
            )
            y = model_rows[
                "realized_centered_token_offset_sd"
            ].to_numpy(dtype=float)
            axis.errorbar(
                float(x.mean()),
                float(y.mean()),
                xerr=float(np.std(x, ddof=1)),
                yerr=float(np.std(y, ddof=1)),
                marker="o",
                markersize=7,
                capsize=3,
                linestyle="none",
                color=model_colors[model_id],
                label=label,
            )
        axis.set_title(f"OB1 attention skew={skew:g}")
        axis.set_xlabel("Mean relative token offset")
        axis.set_ylabel("Centered token-offset SD")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Width control diagnostic: directional shift and centered width"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_markdown_report(
    output_dir: Path,
    metadata: pd.DataFrame,
    aggregate_contrasts: pd.DataFrame,
    audit: dict,
) -> None:
    """Write a compact report with the primary paired robustness contrast."""
    primary = aggregate_contrasts.loc[
        aggregate_contrasts["candidate"].eq("learned_asymmetric")
        & aggregate_contrasts["baseline"].eq(
            "support_centered_sd_symmetric"
        )
    ].copy()
    lines = [
        "# Twelve-checkpoint OB1 attention robustness analysis",
        "",
        (
            "The same saved OB1 fixation trajectories and native T5-token "
            "geometry were reused for every supplied learned sigma pair. "
            "OB1 and ET1 were not rerun."
        ),
        "",
        (
            "All checkpoint pairs receive equal weight. Paired improvements "
            "are computed within checkpoint and passage. The primary "
            "interval independently resamples the 12 checkpoints and 55 "
            "passages; the conditional passage-only interval is also saved."
        ),
        "",
        "## Learned asymmetric versus centered-SD-matched symmetric",
        "",
        (
            "| OB1 skew | Metric | Mean improvement | Crossed 95% CI | "
            "Checkpoint wins |"
        ),
        "|---:|---|---:|---:|---:|",
    ]
    for row in primary.sort_values(
        ["ob1_attention_skew", "metric"]
    ).itertuples():
        lines.append(
            f"| {row.ob1_attention_skew:g} | {row.metric} | "
            f"{row.equal_checkpoint_mean_paired_improvement:.4f} | "
            f"[{row.crossed_bootstrap_ci_low:.4f}, "
            f"{row.crossed_bootstrap_ci_high:.4f}] | "
            f"{row.candidate_wins}/{row.checkpoint_count} |"
        )
    lines.extend(
        [
            "",
            "Positive improvement always means that learned asymmetric is "
            "closer to OB1.",
            "",
            (
                "The primary matched control equalizes the pooled "
                "support-conditioned centered token-offset SD, "
                "sqrt(E[(offset-E[offset])^2]), on identical "
                "fixation-visible supports. Its target comes only from each "
                "frozen learned kernel; it is not fitted to OB1 attention."
            ),
            "",
            (
                "The older support-level sqrt(E[offset^2]) controls are "
                "excluded from this sweep because that second moment mixes "
                "directional mean shift with centered width. Parameter-RMS "
                "conditions remain only as descriptive diagnostics."
            ),
            "",
            f"OB1 fixation SHA-256: `{audit['ob1_fixations_sha256']}`.",
            "",
            f"Checkpoint count: {len(metadata)}.",
            "",
            (
                "Rightward learned checkpoints: "
                f"{int(metadata['rightward_learned'].sum())}/{len(metadata)}."
            ),
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_summary(args: argparse.Namespace) -> dict:
    """Run aggregation, inference, plotting, and provenance checks."""
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    (
        passage_metrics,
        results,
        profiles,
        diagnostics,
        source_audit,
    ) = load_and_validate(
        analysis_dir,
        args.expected_checkpoints,
        args.expected_passages,
        args.expected_ob1_sha256,
        args.expected_et1_sha256,
        args.expected_ob1_manifest_sha256,
        args.expected_sigma_sha256,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Summary output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(passage_metrics)
    width_summary = aggregate_width_diagnostics(diagnostics)
    method_summary = aggregate_method_metrics(
        passage_metrics,
        args.bootstrap_samples,
        args.seed,
    )
    (
        aggregate_contrasts,
        checkpoint_contrasts,
        cell_contrasts,
        passage_contrasts,
    ) = (
        aggregate_paired_contrasts(
            passage_metrics,
            args.bootstrap_samples,
            args.seed,
        )
    )
    profile_summary = aggregate_profiles(profiles)

    metadata.to_csv(output_dir / "checkpoint_metadata.csv", index=False)
    results.loc[results["method"].isin(METHODS)].to_csv(
        output_dir / "checkpoint_metric_table.csv",
        index=False,
    )
    method_summary.to_csv(
        output_dir / "aggregate_method_metrics.csv",
        index=False,
    )
    aggregate_contrasts.to_csv(
        output_dir / "aggregate_paired_contrasts.csv",
        index=False,
    )
    checkpoint_contrasts.to_csv(
        output_dir / "checkpoint_paired_contrasts.csv",
        index=False,
    )
    cell_contrasts.to_csv(
        output_dir / "paired_delta_by_checkpoint_passage.csv",
        index=False,
    )
    passage_contrasts.to_csv(
        output_dir / "paired_delta_by_passage.csv",
        index=False,
    )
    profile_summary.to_csv(
        output_dir / "mean_kernel_profiles.csv",
        index=False,
    )
    diagnostics.to_csv(
        output_dir / "checkpoint_width_diagnostics.csv",
        index=False,
    )
    width_summary.to_csv(
        output_dir / "aggregate_width_diagnostics.csv",
        index=False,
    )
    plot_mean_profiles(
        profile_summary,
        output_dir / "mean_kernel_profiles.png",
    )
    plot_checkpoint_metric_improvements(
        checkpoint_contrasts,
        metadata,
        output_dir,
    )
    plot_width_diagnostics(
        diagnostics,
        output_dir / "width_control_diagnostics.png",
    )
    summary_audit = {
        "analysis": "twelve_checkpoint_attention_profile_robustness",
        "source_analysis_dir": str(analysis_dir),
        "source_attention_audit_sha256": sha256_file(
            analysis_dir / "attention_profile_audit.json"
        ),
        "ob1_fixations_sha256": source_audit["ob1_fixations_sha256"],
        "et1_token_values_sha256": source_audit[
            "et1_token_values_sha256"
        ],
        "ob1_simulation_rerun": False,
        "et1_prediction_rerun": False,
        "actual_et1_trt_magnitudes_used": False,
        "checkpoint_count": int(len(metadata)),
        "passage_count": int(args.expected_passages),
        "attention_skews_aggregated_separately": [3.0, 4.0],
        "checkpoint_weighting": "equal",
        "accuracy_weighting_used": False,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "conditional_passage_bootstrap_unit": (
            "passage after averaging checkpoint deltas"
        ),
        "crossed_bootstrap_unit": (
            "checkpoints and passages independently resampled with "
            "replacement"
        ),
        "seed_by_passage_rows_treated_as_independent": False,
        "metric_directions": METRIC_DIRECTIONS,
        "primary_contrast": {
            "candidate": "learned_asymmetric",
            "baseline": "support_centered_sd_symmetric",
        },
        "matched_control_definition": (
            "same pooled support-conditioned centered token-offset SD "
            "sqrt(E[(offset-E[offset])^2]) on identical fixation-visible "
            "supports"
        ),
        "legacy_rms_control_role": (
            "support-level sqrt(E[offset^2]) controls disabled because they "
            "combine centered width and directional mean shift; "
            "parameter-RMS conditions retained only in source diagnostics"
        ),
    }
    with (output_dir / "summary_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary_audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_markdown_report(
        output_dir,
        metadata,
        aggregate_contrasts,
        source_audit,
    )
    return {
        "output_dir": str(output_dir),
        "checkpoint_count": int(len(metadata)),
        "passage_count": int(args.expected_passages),
        "ob1_rerun": False,
        "et1_rerun": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a completed 12-checkpoint Provo/OB1 attention-profile "
            "analysis without treating checkpoint-passage cells as "
            "independent observations."
        )
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=12)
    parser.add_argument("--expected-passages", type=int, default=55)
    parser.add_argument(
        "--expected-ob1-sha256",
        default=EXPECTED_OB1_SHA256,
    )
    parser.add_argument(
        "--expected-et1-sha256",
        default=EXPECTED_ET1_SHA256,
    )
    parser.add_argument(
        "--expected-ob1-manifest-sha256",
        default=EXPECTED_OB1_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-sigma-sha256",
        default=EXPECTED_SIGMA_SHA256,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser


def main() -> None:
    """Run the summary and print one machine-readable completion record."""
    args = build_parser().parse_args()
    print(json.dumps(run_summary(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Optional, Sequence

import torch


ET2_FEATURE_NAMES = ("nFix", "FFD", "GPT", "TRT", "fixProp")
ET2_TRT_INDEX = 3
ET2_RAW_GAZE_CONCAT = "raw_gaze_concat"
ET2_TRT_REDISTRIBUTED_GAZE_CONCAT = "trt_redistributed_gaze_concat"
ET2_GAZE_CONCAT_CONDITIONS = (
    ET2_RAW_GAZE_CONCAT,
    ET2_TRT_REDISTRIBUTED_GAZE_CONCAT,
)


def validate_features_used(features_used: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize the canonical five-feature ET2 mask."""
    normalized = tuple(int(value) for value in features_used)
    if len(normalized) != len(ET2_FEATURE_NAMES):
        raise ValueError(
            "features_used must contain five values in the order "
            "[nFix, FFD, GPT, TRT, fixProp]"
        )
    if any(value not in (0, 1) for value in normalized):
        raise ValueError("features_used values must be either 0 or 1")
    if not any(normalized):
        raise ValueError("features_used must select at least one ET2 feature")
    return normalized


def resolve_et2_gaze_concat_condition(
    *,
    fixations_model_version: int,
    concat: bool,
    use_softprompt: bool,
    features_used: Sequence[int],
    requested_condition: Optional[str],
    use_asym_gaussian_redistributor: bool,
) -> Optional[str]:
    """Resolve and validate the explicit ET2 GazeConcat experiment condition."""
    if fixations_model_version != 2:
        if requested_condition is not None:
            raise ValueError(
                "et2_gaze_concat_condition requires fixations_model_version=2"
            )
        return None

    normalized_features = validate_features_used(features_used)
    is_et2_gaze_concat = (
        concat and use_softprompt
    )

    if requested_condition is not None:
        if requested_condition not in ET2_GAZE_CONCAT_CONDITIONS:
            raise ValueError(
                "et2_gaze_concat_condition must be one of "
                f"{ET2_GAZE_CONCAT_CONDITIONS}"
            )
        if not is_et2_gaze_concat:
            raise ValueError(
                "et2_gaze_concat_condition requires "
                "fixations_model_version=2, concat=True, and use_softprompt=True"
            )

    if not is_et2_gaze_concat or requested_condition is None:
        return None

    if (
        requested_condition == ET2_TRT_REDISTRIBUTED_GAZE_CONCAT
        and normalized_features != (1, 1, 1, 1, 1)
    ):
        raise ValueError(
            "trt_redistributed_gaze_concat requires "
            "features_used=1,1,1,1,1 so TRT is redistributed and the other "
            "four raw gaze features are retained"
        )
    return requested_condition


def apply_et2_feature_condition(
    fixations: torch.Tensor,
    attention_mask: torch.Tensor,
    features_used: Sequence[int],
    redistributor,
    condition: Optional[str],
    legacy_redistribution_enabled: bool,
) -> torch.Tensor:
    """Apply the selected ET2 condition while leaving non-TRT channels raw."""
    normalized_features = validate_features_used(features_used)
    expected_features = sum(normalized_features)
    if fixations.dim() != 3 or fixations.shape[-1] != expected_features:
        raise ValueError(
            "ET2 fixations must have shape [batch, tokens, selected_features]; "
            f"expected {expected_features} features, got {tuple(fixations.shape)}"
        )

    if condition == ET2_RAW_GAZE_CONCAT:
        return fixations
    should_redistribute = (
        condition == ET2_TRT_REDISTRIBUTED_GAZE_CONCAT
        or (condition is None and legacy_redistribution_enabled)
    )
    if not should_redistribute or normalized_features[ET2_TRT_INDEX] == 0:
        return fixations

    filtered_trt_index = sum(normalized_features[:ET2_TRT_INDEX])
    redistributed_trt = redistributor(
        fixations[:, :, filtered_trt_index],
        attention_mask,
    )
    conditioned = fixations.clone()
    conditioned[:, :, filtered_trt_index] = redistributed_trt
    return conditioned

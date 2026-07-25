import pytest
import torch

from models.et2_fusion import (
    ET2_RAW_GAZE_CONCAT,
    ET2_TRT_REDISTRIBUTED_GAZE_CONCAT,
    apply_et2_feature_condition,
    resolve_et2_gaze_concat_condition,
)


class AdditiveRedistributor:
    def __call__(self, trt, attention_mask):
        return trt + (100 * attention_mask)


def test_raw_gaze_concat_leaves_all_five_features_unchanged():
    features = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)
    mask = torch.ones(1, 6)

    result = apply_et2_feature_condition(
        fixations=features,
        attention_mask=mask,
        features_used=[1, 1, 1, 1, 1],
        redistributor=AdditiveRedistributor(),
        condition=ET2_RAW_GAZE_CONCAT,
        legacy_redistribution_enabled=True,
    )

    assert result is features
    assert torch.equal(result, features)


def test_hybrid_condition_changes_only_trt_and_preserves_other_four_features():
    features = torch.arange(30, dtype=torch.float32).reshape(1, 6, 5)
    original = features.clone()
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.float32)

    result = apply_et2_feature_condition(
        fixations=features,
        attention_mask=mask,
        features_used=[1, 1, 1, 1, 1],
        redistributor=AdditiveRedistributor(),
        condition=ET2_TRT_REDISTRIBUTED_GAZE_CONCAT,
        legacy_redistribution_enabled=False,
    )

    assert torch.equal(result[:, :, [0, 1, 2, 4]], original[:, :, [0, 1, 2, 4]])
    assert torch.equal(result[:, :, 3], original[:, :, 3] + (100 * mask))
    assert torch.equal(features, original)


def test_default_condition_preserves_legacy_redistribution_setting():
    common = {
        "fixations_model_version": 2,
        "concat": True,
        "use_softprompt": True,
        "features_used": [1, 1, 1, 1, 1],
        "requested_condition": None,
    }

    assert resolve_et2_gaze_concat_condition(
        **common,
        use_asym_gaussian_redistributor=True,
    ) is None
    assert resolve_et2_gaze_concat_condition(
        **common,
        use_asym_gaussian_redistributor=False,
    ) is None

    features = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.ones(1, 6)
    redistributed = apply_et2_feature_condition(
        fixations=features,
        attention_mask=mask,
        features_used=[0, 0, 1, 1, 0],
        redistributor=AdditiveRedistributor(),
        condition=None,
        legacy_redistribution_enabled=True,
    )
    unchanged = apply_et2_feature_condition(
        fixations=features,
        attention_mask=mask,
        features_used=[0, 0, 1, 1, 0],
        redistributor=AdditiveRedistributor(),
        condition=None,
        legacy_redistribution_enabled=False,
    )

    assert torch.equal(redistributed[:, :, 0], features[:, :, 0])
    assert torch.equal(redistributed[:, :, 1], features[:, :, 1] + 100)
    assert unchanged is features


def test_hybrid_condition_requires_all_five_features():
    with pytest.raises(ValueError, match="features_used=1,1,1,1,1"):
        resolve_et2_gaze_concat_condition(
            fixations_model_version=2,
            concat=True,
            use_softprompt=True,
            features_used=[1, 1, 0, 1, 1],
            requested_condition=ET2_TRT_REDISTRIBUTED_GAZE_CONCAT,
            use_asym_gaussian_redistributor=True,
        )


def test_explicit_et2_gaze_concat_condition_requires_et2_concat():
    with pytest.raises(ValueError, match="requires"):
        resolve_et2_gaze_concat_condition(
            fixations_model_version=2,
            concat=False,
            use_softprompt=False,
            features_used=[1, 1, 1, 1, 1],
            requested_condition=ET2_RAW_GAZE_CONCAT,
            use_asym_gaussian_redistributor=False,
        )

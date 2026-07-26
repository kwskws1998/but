"""Tests for the ET2 Provo provenance audit."""

from pathlib import Path

import pytest

from cognitive_model_comparsion.src.audit_et2_provo import (
    ET2_PATH,
    RAW_PATH,
    compare_et2_provo,
)


@pytest.mark.skipif(
    not RAW_PATH.is_file() or not ET2_PATH.is_file(),
    reason="Downloaded Provo and ET2 reference assets are not installed",
)
def test_official_provo_reproduces_et2_training_table():
    """The official Provo raw file reproduces the distributed ET2 table."""
    report = compare_et2_provo(
        Path(RAW_PATH),
        Path(ET2_PATH),
        tolerance=1e-10,
    )

    assert report["reconstructed_rows"] == 2659
    assert report["distributed_rows"] == 2659
    assert report["keys_match"]
    assert report["all_match"]

"""Load a fixed OB1-fitted Gaussian prior with reproducible provenance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


REDISTRIBUTOR_MIN_SIGMA = 1e-6


def load_fixed_ob1_prior(path: str | Path, attention_skew: float) -> dict:
    """Select one checked effective-sigma pair from profile output JSON."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Fixed OB1 prior JSON not found: {resolved}")
    payload = resolved.read_bytes()
    records = json.loads(payload.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Fixed OB1 prior JSON must contain a nonempty list")
    requested_skew = float(attention_skew)
    matches = [
        record
        for record in records
        if math.isclose(
            float(record.get("ob1_attention_skew", float("nan"))),
            requested_skew,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "Fixed OB1 prior JSON must contain exactly one record for "
            f"attention skew {requested_skew}"
        )
    selected = matches[0]
    if selected.get("profile_component") != "focused":
        raise ValueError(
            "Fixed RM prior must be fitted to the focused OB1 attention "
            "component"
        )
    if (
        selected.get("projection_coordinate")
        != "relative_native_t5_token_index"
    ):
        raise ValueError(
            "Fixed RM prior must use relative native T5 token coordinates"
        )
    sigma_left = float(selected["sigma_left"])
    sigma_right = float(selected["sigma_right"])
    if (
        not math.isfinite(sigma_left)
        or not math.isfinite(sigma_right)
        or sigma_left <= REDISTRIBUTOR_MIN_SIGMA
        or sigma_right <= sigma_left
    ):
        raise ValueError(
            "Fixed OB1 prior must contain finite sigmas above the "
            "redistributor minimum, with sigma_right greater than sigma_left"
        )
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "ob1_attention_skew": requested_skew,
        "profile_component": selected["profile_component"],
        "fixation_weighting": selected["fixation_weighting"],
        "projection_coordinate": selected["projection_coordinate"],
        "sigma_left": sigma_left,
        "sigma_right": sigma_right,
        "initializer_sigma_left": sigma_left - REDISTRIBUTOR_MIN_SIGMA,
        "initializer_sigma_right": sigma_right - REDISTRIBUTOR_MIN_SIGMA,
        "redistributor_min_sigma": REDISTRIBUTOR_MIN_SIGMA,
        "right_left_ratio": sigma_right / sigma_left,
        "fit_js_divergence": float(selected["fit_js_divergence"]),
    }

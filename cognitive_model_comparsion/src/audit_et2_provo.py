"""Reproduce and audit the Provo table distributed with the ET2 training code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/Provo_Corpus-Eyetracking_Data.csv"
ET2_PATH = ROOT / "third_party/et2_torontocl_cmcl_2021/data/provo.csv"
KEY_COLUMNS = ["sentence_id", "word_id", "word"]
FEATURE_COLUMNS = ["nFix", "FFD", "GPT", "TRT", "fixProp"]
TARGET_MEANS = {
    "nFix": 15.10,
    "FFD": 3.19,
    "GPT": 6.35,
    "TRT": 5.31,
    "fixProp": 67.06,
}
TARGET_STANDARD_DEVIATIONS = {
    "nFix": 9.42,
    "FFD": 1.42,
    "GPT": 5.91,
    "TRT": 3.64,
    "fixProp": 26.06,
}


def reproduce_et2_provo(raw_path: Path) -> pd.DataFrame:
    """Apply the published ET2 repository's ProvoProcess transformation."""
    raw = pd.read_csv(raw_path, encoding="cp1252")
    frame = pd.DataFrame(
        {
            "participant_id": raw["Participant_ID"],
            "text_id": raw["Text_ID"],
            "orig_sentence_id": raw["Sentence_Number"],
            "word_id": raw["Word_In_Sentence_Number"],
            "word": raw["Word"],
            "nFix": raw["IA_FIXATION_COUNT"],
            "FFD": raw["IA_FIRST_FIXATION_DURATION"],
            "GPT": raw["IA_REGRESSION_PATH_DURATION"],
            "TRT": raw["IA_DWELL_TIME"],
        }
    ).fillna(0)
    frame["orig_sentence_id"] = frame["orig_sentence_id"].astype(int)
    frame["word_id"] = frame["word_id"].astype(int)
    frame["nFix"] = frame["nFix"].astype(float)
    frame["TRT"] = frame["TRT"].astype(float)
    frame = frame[
        ~((frame["orig_sentence_id"] == 0) | (frame["word_id"] == 0))
    ].copy()

    sentence_ids: dict[tuple[int, int], int] = {}
    sentence_keys = list(zip(frame["text_id"], frame["orig_sentence_id"]))
    for key in sentence_keys:
        if key not in sentence_ids:
            sentence_ids[key] = len(sentence_ids)
    frame["sentence_id"] = [sentence_ids[key] for key in sentence_keys]

    group_columns = ["sentence_id", "word_id", "word"]
    averaged = (
        frame.groupby(group_columns, sort=True)[["nFix", "FFD", "GPT", "TRT"]]
        .mean()
        .reset_index()
    )
    fixation_proportion = (
        frame.groupby(group_columns, sort=True)["nFix"]
        .apply(lambda values: (values != 0).sum() / len(values))
        .reset_index(name="fixProp")
    )
    averaged = averaged.merge(
        fixation_proportion,
        on=group_columns,
        validate="one_to_one",
    )

    features = averaged[FEATURE_COLUMNS]
    averaged[FEATURE_COLUMNS] = (
        features - features.mean(axis=0)
    ) / features.std(axis=0)
    for feature in FEATURE_COLUMNS:
        averaged[feature] = (
            TARGET_MEANS[feature]
            + TARGET_STANDARD_DEVIATIONS[feature] * averaged[feature]
        )
    return averaged


def compare_et2_provo(
    raw_path: Path,
    et2_path: Path,
    tolerance: float = 1e-10,
) -> dict:
    """Compare a fresh reconstruction with the pinned ET2 Provo table."""
    reconstructed = reproduce_et2_provo(raw_path)
    distributed = pd.read_csv(et2_path)
    merged = reconstructed.merge(
        distributed,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_reconstructed", "_distributed"),
        indicator=True,
        validate="one_to_one",
    )

    key_counts = merged["_merge"].value_counts().to_dict()
    maximum_absolute_differences = {}
    feature_match = {}
    for feature in FEATURE_COLUMNS:
        difference = (
            merged[f"{feature}_reconstructed"]
            - merged[f"{feature}_distributed"]
        ).abs()
        maximum_absolute_differences[feature] = float(difference.max())
        feature_match[feature] = bool(
            np.allclose(
                merged[f"{feature}_reconstructed"],
                merged[f"{feature}_distributed"],
                rtol=0.0,
                atol=tolerance,
                equal_nan=False,
            )
        )

    keys_match = (
        key_counts.get("left_only", 0) == 0
        and key_counts.get("right_only", 0) == 0
        and key_counts.get("both", 0) == len(reconstructed) == len(distributed)
    )
    return {
        "raw_path": str(raw_path),
        "et2_path": str(et2_path),
        "reconstructed_rows": len(reconstructed),
        "distributed_rows": len(distributed),
        "key_counts": key_counts,
        "keys_match": keys_match,
        "tolerance": tolerance,
        "maximum_absolute_differences": maximum_absolute_differences,
        "features_match": feature_match,
        "all_match": keys_match and all(feature_match.values()),
    }


def parse_args() -> argparse.Namespace:
    """Parse local asset paths and numerical tolerance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--et2-provo", type=Path, default=ET2_PATH)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    """Run the ET2 Provo reconstruction audit."""
    args = parse_args()
    report = compare_et2_provo(args.raw, args.et2_provo, args.tolerance)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_match"]:
        raise SystemExit("ET2 Provo reconstruction did not match")


if __name__ == "__main__":
    main()

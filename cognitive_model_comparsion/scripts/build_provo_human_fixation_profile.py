"""Build an ET1-free Human Provo next-fixation displacement profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from cognitive_model_comparsion.src.prepare_provo import normalize_ob1_word


DEFAULT_FIXATION_PATH = (
    COGNITIVE_ROOT
    / "data"
    / "raw"
    / "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv"
)
DEFAULT_WORD_PATH = (
    COGNITIVE_ROOT / "data" / "processed" / "provo" / "provo_words.csv"
)
DEFAULT_PASSAGE_PATH = (
    COGNITIVE_ROOT / "data" / "processed" / "provo" / "provo_passages.csv"
)
DEFAULT_OUTPUT_DIR = (
    COGNITIVE_ROOT
    / "outputs"
    / "provo_human_fixation_profile_20260729"
)
EXPECTED_FIXATION_SHA256 = (
    "0d961a6508ed6caafdb4bc1025c067ecc97a0be07b13d3de0acafb5ef6c4fb7e"
)
OFFSET_VALUES = np.arange(-3, 7, dtype=int)
OFFSET_LABELS = (
    "≤−3",
    "−2",
    "−1",
    "0",
    "+1",
    "+2",
    "+3",
    "+4",
    "+5",
    "≥+6",
)
DIRECTION_ORDER = ("Regressive", "Refixation", "Progressive")
SOURCE_ORDER = ("Human Provo", "OB1 simulation")
SOURCE_COLORS = {
    "Human Provo": "#d96aa7",
    "OB1 simulation": "#2f8ff3",
}
FIXATION_COLUMNS = (
    "RECORDING_SESSION_LABEL",
    "TRIAL_INDEX",
    "trial",
    "CURRENT_FIX_INDEX",
    "CURRENT_FIX_INTEREST_AREA_INDEX",
    "CURRENT_FIX_INTEREST_AREA_LABEL",
    "NEXT_FIX_INTEREST_AREA_INDEX",
    "NEXT_FIX_INTEREST_AREA_LABEL",
    "CURRENT_FIX_BLINK_AROUND",
    "NEXT_FIX_BLINK_AROUND",
    "CURRENT_FIX_DURATION",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixation_report(path: Path) -> pd.DataFrame:
    """Load the official Provo fixation report and validate its schema."""
    frame = pd.read_csv(
        path,
        encoding="cp1252",
        usecols=list(FIXATION_COLUMNS),
        low_memory=False,
    )
    missing = sorted(set(FIXATION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Fixation report is missing columns: {missing}")
    return frame


def build_interest_area_inventory(fixations: pd.DataFrame) -> pd.DataFrame:
    """Collect one stable interest-area label for each passage and index."""
    current = fixations[
        [
            "trial",
            "CURRENT_FIX_INTEREST_AREA_INDEX",
            "CURRENT_FIX_INTEREST_AREA_LABEL",
        ]
    ].rename(
        columns={
            "CURRENT_FIX_INTEREST_AREA_INDEX": "interest_area_index",
            "CURRENT_FIX_INTEREST_AREA_LABEL": "interest_area_label",
        }
    )
    following = fixations[
        [
            "trial",
            "NEXT_FIX_INTEREST_AREA_INDEX",
            "NEXT_FIX_INTEREST_AREA_LABEL",
        ]
    ].rename(
        columns={
            "NEXT_FIX_INTEREST_AREA_INDEX": "interest_area_index",
            "NEXT_FIX_INTEREST_AREA_LABEL": "interest_area_label",
        }
    )
    inventory = pd.concat([current, following], ignore_index=True)
    inventory["interest_area_index"] = pd.to_numeric(
        inventory["interest_area_index"],
        errors="coerce",
    )
    inventory = inventory.dropna(
        subset=["trial", "interest_area_index", "interest_area_label"]
    ).copy()
    inventory["passage_id_raw"] = inventory["trial"].astype(int)
    inventory["interest_area_index"] = inventory[
        "interest_area_index"
    ].astype(int)
    inventory["interest_area_label"] = (
        inventory["interest_area_label"].astype(str).str.strip()
    )
    inventory["normalized_label"] = inventory[
        "interest_area_label"
    ].map(normalize_ob1_word)
    label_counts = inventory.groupby(
        ["passage_id_raw", "interest_area_index"],
        sort=True,
    )["normalized_label"].nunique()
    unstable = label_counts[label_counts != 1]
    if not unstable.empty:
        raise ValueError(
            "Interest-area labels are inconsistent within passage/index: "
            f"{unstable.to_dict()}"
        )
    return (
        inventory.sort_values(
            ["passage_id_raw", "interest_area_index", "interest_area_label"]
        )
        .drop_duplicates(["passage_id_raw", "interest_area_index"])
        [
            [
                "passage_id_raw",
                "interest_area_index",
                "interest_area_label",
                "normalized_label",
            ]
        ]
        .reset_index(drop=True)
    )


def align_one_passage(
    passage_id_raw: int,
    passage_text: str,
    interest_areas: pd.DataFrame,
) -> pd.DataFrame:
    """Align one interest-area sequence to the canonical passage words."""
    canonical_words = re.findall(r"\S+", passage_text)
    canonical_clean = [
        normalize_ob1_word(word) for word in canonical_words
    ]
    areas = list(
        interest_areas.sort_values("interest_area_index").itertuples(
            index=False
        )
    )
    records = []
    area_position = 0
    word_position = 0
    while area_position < len(areas) and word_position < len(canonical_words):
        area = areas[area_position]
        label_clean = area.normalized_label
        word_clean = canonical_clean[word_position]
        if label_clean == word_clean:
            records.append(
                {
                    "passage_id_raw": passage_id_raw,
                    "passage_id_zero_based": passage_id_raw - 1,
                    "interest_area_index": int(area.interest_area_index),
                    "interest_area_label": area.interest_area_label,
                    "normalized_label": label_clean,
                    "mapping_status": "one_to_one",
                    "canonical_span_start": word_position,
                    "canonical_span_end": word_position,
                    "word_id_zero_based": word_position,
                }
            )
            area_position += 1
            word_position += 1
            continue

        merged_width = None
        for width in range(2, 5):
            if word_position + width > len(canonical_words):
                continue
            merged_clean = "".join(
                canonical_clean[word_position : word_position + width]
            )
            if label_clean == merged_clean:
                merged_width = width
                break
        if merged_width is not None:
            records.append(
                {
                    "passage_id_raw": passage_id_raw,
                    "passage_id_zero_based": passage_id_raw - 1,
                    "interest_area_index": int(area.interest_area_index),
                    "interest_area_label": area.interest_area_label,
                    "normalized_label": label_clean,
                    "mapping_status": "merged_multiple_words",
                    "canonical_span_start": word_position,
                    "canonical_span_end": word_position + merged_width - 1,
                    "word_id_zero_based": np.nan,
                }
            )
            area_position += 1
            word_position += merged_width
            continue

        if area_position + 1 < len(areas):
            following_clean = areas[area_position + 1].normalized_label
            if following_clean == word_clean:
                records.append(
                    {
                        "passage_id_raw": passage_id_raw,
                        "passage_id_zero_based": passage_id_raw - 1,
                        "interest_area_index": int(area.interest_area_index),
                        "interest_area_label": area.interest_area_label,
                        "normalized_label": label_clean,
                        "mapping_status": "extra_noncanonical_area",
                        "canonical_span_start": np.nan,
                        "canonical_span_end": np.nan,
                        "word_id_zero_based": np.nan,
                    }
                )
                area_position += 1
                continue

        raise ValueError(
            "Could not align fixation interest area to passage text: "
            f"passage={passage_id_raw}, "
            f"area_index={area.interest_area_index}, "
            f"area_label={area.interest_area_label!r}, "
            f"canonical_word_index={word_position}, "
            f"canonical_word={canonical_words[word_position]!r}"
        )

    if area_position != len(areas) or word_position != len(canonical_words):
        raise ValueError(
            "Interest-area alignment ended early: "
            f"passage={passage_id_raw}, "
            f"areas={area_position}/{len(areas)}, "
            f"words={word_position}/{len(canonical_words)}"
        )
    return pd.DataFrame(records)


def build_interest_area_map(
    fixations: pd.DataFrame,
    passages: pd.DataFrame,
) -> pd.DataFrame:
    """Align all 55 Provo interest-area grids to canonical word positions."""
    inventory = build_interest_area_inventory(fixations)
    expected_passages = set(passages["passage_id_raw"].astype(int))
    observed_passages = set(inventory["passage_id_raw"].astype(int))
    if expected_passages != observed_passages:
        raise ValueError(
            "Passage mismatch between fixation report and canonical table: "
            f"missing={sorted(expected_passages - observed_passages)}, "
            f"extra={sorted(observed_passages - expected_passages)}"
        )
    aligned = []
    for passage in passages.sort_values("passage_id_raw").itertuples():
        passage_areas = inventory.loc[
            inventory["passage_id_raw"].eq(int(passage.passage_id_raw))
        ]
        aligned.append(
            align_one_passage(
                int(passage.passage_id_raw),
                str(passage.passage_text),
                passage_areas,
            )
        )
    mapping = pd.concat(aligned, ignore_index=True)
    if mapping.duplicated(
        ["passage_id_raw", "interest_area_index"]
    ).any():
        raise ValueError("Aligned interest-area coordinates are not unique")
    return mapping


def add_transition_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Add clipped display bins and directional transition labels."""
    output = frame.copy()
    output["word_displacement"] = (
        output["next_word_id_zero_based"]
        - output["current_word_id_zero_based"]
    ).astype(int)
    output["display_offset"] = output["word_displacement"].clip(
        lower=int(OFFSET_VALUES.min()),
        upper=int(OFFSET_VALUES.max()),
    )
    output["direction"] = np.select(
        [
            output["word_displacement"].lt(0),
            output["word_displacement"].eq(0),
        ],
        ["Regressive", "Refixation"],
        default="Progressive",
    )
    return output


def build_human_transitions(
    fixations: pd.DataFrame,
    mapping: pd.DataFrame,
    canonical_words: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Construct participant-tagged Human Provo next-fixation transitions."""
    audit = {"raw_fixation_rows": int(len(fixations))}
    frame = fixations.loc[fixations["TRIAL_INDEX"].le(55)].copy()
    audit["main_trial_rows"] = int(len(frame))
    audit["excluded_repeat_trial_rows"] = int(len(fixations) - len(frame))
    sequence_check = frame.sort_values(
        [
            "RECORDING_SESSION_LABEL",
            "TRIAL_INDEX",
            "CURRENT_FIX_INDEX",
        ]
    ).copy()
    sequence_check["following_current_area"] = sequence_check.groupby(
        ["RECORDING_SESSION_LABEL", "TRIAL_INDEX"],
        sort=False,
    )["CURRENT_FIX_INTEREST_AREA_INDEX"].shift(-1)
    sequence_check = sequence_check.dropna(
        subset=["following_current_area"]
    )
    sequence_matches = (
        sequence_check["NEXT_FIX_INTEREST_AREA_INDEX"].astype(str)
        == sequence_check["following_current_area"].astype(str)
    )
    if not bool(sequence_matches.all()):
        raise ValueError(
            "NEXT_FIX interest areas do not reproduce the recorded fixation "
            "sequence"
        )
    audit["verified_sequential_next_fixation_rows"] = int(
        len(sequence_check)
    )
    frame["current_interest_area_index"] = pd.to_numeric(
        frame["CURRENT_FIX_INTEREST_AREA_INDEX"],
        errors="coerce",
    )
    frame["next_interest_area_index"] = pd.to_numeric(
        frame["NEXT_FIX_INTEREST_AREA_INDEX"],
        errors="coerce",
    )
    numeric_mask = frame[
        ["current_interest_area_index", "next_interest_area_index"]
    ].notna().all(axis=1)
    audit["numeric_current_and_next_rows"] = int(numeric_mask.sum())
    frame = frame.loc[numeric_mask].copy()
    no_blink_mask = (
        frame["CURRENT_FIX_BLINK_AROUND"].eq("NONE")
        & frame["NEXT_FIX_BLINK_AROUND"].eq("NONE")
    )
    audit["no_blink_current_and_next_rows"] = int(no_blink_mask.sum())
    frame = frame.loc[no_blink_mask].copy()
    frame["passage_id_raw"] = frame["trial"].astype(int)
    frame["passage_id_zero_based"] = frame["passage_id_raw"] - 1
    frame["current_interest_area_index"] = frame[
        "current_interest_area_index"
    ].astype(int)
    frame["next_interest_area_index"] = frame[
        "next_interest_area_index"
    ].astype(int)
    coordinate_map = {
        (int(row.passage_id_raw), int(row.interest_area_index)): (
            int(row.word_id_zero_based)
            if row.mapping_status == "one_to_one"
            else None
        )
        for row in mapping.itertuples()
    }
    frame["current_word_id_zero_based"] = [
        coordinate_map.get((passage, area))
        for passage, area in zip(
            frame["passage_id_raw"],
            frame["current_interest_area_index"],
        )
    ]
    frame["next_word_id_zero_based"] = [
        coordinate_map.get((passage, area))
        for passage, area in zip(
            frame["passage_id_raw"],
            frame["next_interest_area_index"],
        )
    ]
    mapped_mask = frame[
        ["current_word_id_zero_based", "next_word_id_zero_based"]
    ].notna().all(axis=1)
    audit["one_to_one_mapped_rows"] = int(mapped_mask.sum())
    audit["excluded_ambiguous_or_extra_area_rows"] = int(
        len(frame) - mapped_mask.sum()
    )
    frame = frame.loc[mapped_mask].copy()
    frame["current_word_id_zero_based"] = frame[
        "current_word_id_zero_based"
    ].astype(int)
    frame["next_word_id_zero_based"] = frame[
        "next_word_id_zero_based"
    ].astype(int)
    canonical_coordinates = set(
        zip(
            canonical_words["passage_id_zero_based"].astype(int),
            canonical_words["word_id_zero_based"].astype(int),
        )
    )
    canonical_mask = [
        (passage, current) in canonical_coordinates
        and (passage, following) in canonical_coordinates
        for passage, current, following in zip(
            frame["passage_id_zero_based"],
            frame["current_word_id_zero_based"],
            frame["next_word_id_zero_based"],
        )
    ]
    canonical_mask = np.asarray(canonical_mask, dtype=bool)
    audit["canonical_endpoint_rows"] = int(canonical_mask.sum())
    audit["excluded_non_evaluable_endpoint_rows"] = int(
        len(frame) - canonical_mask.sum()
    )
    frame = frame.loc[canonical_mask].copy()
    frame = frame.rename(
        columns={
            "RECORDING_SESSION_LABEL": "unit_id",
            "CURRENT_FIX_INDEX": "fixation_index",
            "CURRENT_FIX_DURATION": "fixation_duration_ms",
        }
    )
    frame["unit_id"] = frame["unit_id"].astype(str)
    frame["source"] = "Human Provo"
    transitions = add_transition_categories(frame)
    audit.update(
        {
            "transition_rows": int(len(transitions)),
            "reader_count": int(transitions["unit_id"].nunique()),
            "passage_count": int(
                transitions["passage_id_zero_based"].nunique()
            ),
        }
    )
    selected_columns = [
        "source",
        "unit_id",
        "passage_id_raw",
        "passage_id_zero_based",
        "fixation_index",
        "current_interest_area_index",
        "next_interest_area_index",
        "current_word_id_zero_based",
        "next_word_id_zero_based",
        "word_displacement",
        "display_offset",
        "direction",
        "fixation_duration_ms",
    ]
    return transitions[selected_columns].reset_index(drop=True), audit


def build_ob1_transitions(
    path: Path,
    canonical_words: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Construct simulation-tagged OB1 next-fixation word transitions."""
    frame = pd.read_csv(path)
    raw_fixation_rows = int(len(frame))
    required = {
        "simulation_id",
        "text_id",
        "fixation_counter",
        "word_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OB1 fixation table is missing columns: {missing}")
    if frame.duplicated(
        ["simulation_id", "text_id", "fixation_counter"]
    ).any():
        raise ValueError("OB1 fixation counters are not unique")
    frame = frame.sort_values(
        ["simulation_id", "text_id", "fixation_counter"]
    ).copy()
    frame["next_word_id_zero_based"] = frame.groupby(
        ["simulation_id", "text_id"],
        sort=False,
    )["word_id"].shift(-1)
    frame = frame.dropna(subset=["next_word_id_zero_based"]).copy()
    frame["passage_id_zero_based"] = frame["text_id"].astype(int)
    frame["passage_id_raw"] = frame["passage_id_zero_based"] + 1
    frame["current_word_id_zero_based"] = frame["word_id"].astype(int)
    frame["next_word_id_zero_based"] = frame[
        "next_word_id_zero_based"
    ].astype(int)
    canonical_coordinates = set(
        zip(
            canonical_words["passage_id_zero_based"].astype(int),
            canonical_words["word_id_zero_based"].astype(int),
        )
    )
    canonical_mask = np.asarray(
        [
            (passage, current) in canonical_coordinates
            and (passage, following) in canonical_coordinates
            for passage, current, following in zip(
                frame["passage_id_zero_based"],
                frame["current_word_id_zero_based"],
                frame["next_word_id_zero_based"],
            )
        ],
        dtype=bool,
    )
    raw_transition_rows = int(len(frame))
    frame = frame.loc[canonical_mask].copy()
    frame["unit_id"] = frame["simulation_id"].astype(str)
    frame["source"] = "OB1 simulation"
    frame["fixation_index"] = frame["fixation_counter"].astype(int)
    transitions = add_transition_categories(frame)
    selected_columns = [
        "source",
        "unit_id",
        "passage_id_raw",
        "passage_id_zero_based",
        "fixation_index",
        "current_word_id_zero_based",
        "next_word_id_zero_based",
        "word_displacement",
        "display_offset",
        "direction",
    ]
    audit = {
        "raw_fixation_rows": raw_fixation_rows,
        "raw_transition_rows": raw_transition_rows,
        "canonical_endpoint_rows": int(len(transitions)),
        "excluded_non_evaluable_endpoint_rows": int(
            raw_transition_rows - len(transitions)
        ),
        "simulation_count": int(transitions["unit_id"].nunique()),
        "passage_count": int(
            transitions["passage_id_zero_based"].nunique()
        ),
    }
    return transitions[selected_columns].reset_index(drop=True), audit


def build_unit_profiles(transitions: pd.DataFrame) -> pd.DataFrame:
    """Normalize displacement and direction counts within each reader or run."""
    records = []
    for (source, unit_id), group in transitions.groupby(
        ["source", "unit_id"],
        sort=True,
    ):
        total = len(group)
        offset_counts = (
            group["display_offset"]
            .value_counts()
            .reindex(OFFSET_VALUES, fill_value=0)
        )
        direction_counts = (
            group["direction"]
            .value_counts()
            .reindex(DIRECTION_ORDER, fill_value=0)
        )
        record = {
            "source": source,
            "unit_id": str(unit_id),
            "transition_count": int(total),
        }
        for offset, count in offset_counts.items():
            record[f"offset_{int(offset)}"] = float(count / total)
        for direction, count in direction_counts.items():
            record[f"direction_{direction.lower()}"] = float(count / total)
        records.append(record)
    profiles = pd.DataFrame(records)
    expected_sums = profiles[
        [f"offset_{int(offset)}" for offset in OFFSET_VALUES]
    ].sum(axis=1)
    if not np.allclose(expected_sums, 1.0):
        raise ValueError("Unit-level displacement profiles do not sum to one")
    return profiles


def bootstrap_profile_summary(
    unit_profiles: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize unit-balanced profiles with percentile bootstrap intervals."""
    if bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    rng = np.random.default_rng(seed)
    offset_records = []
    direction_records = []
    for source in SOURCE_ORDER:
        source_profiles = unit_profiles.loc[
            unit_profiles["source"].eq(source)
        ]
        if source_profiles.empty:
            continue
        offset_columns = [
            f"offset_{int(offset)}" for offset in OFFSET_VALUES
        ]
        direction_columns = [
            f"direction_{direction.lower()}"
            for direction in DIRECTION_ORDER
        ]
        values = source_profiles[
            offset_columns + direction_columns
        ].to_numpy(dtype=float)
        point = values.mean(axis=0)
        bootstrap = np.empty((bootstrap_samples, values.shape[1]))
        for sample_index in range(bootstrap_samples):
            draw = rng.integers(0, len(values), size=len(values))
            bootstrap[sample_index] = values[draw].mean(axis=0)
        lower = np.quantile(bootstrap, 0.025, axis=0)
        upper = np.quantile(bootstrap, 0.975, axis=0)
        for index, (offset, label) in enumerate(
            zip(OFFSET_VALUES, OFFSET_LABELS)
        ):
            offset_records.append(
                {
                    "source": source,
                    "display_offset": int(offset),
                    "display_label": label,
                    "mean_probability": float(point[index]),
                    "ci_low": float(lower[index]),
                    "ci_high": float(upper[index]),
                    "unit_count": int(len(values)),
                }
            )
        direction_start = len(offset_columns)
        for direction_index, direction in enumerate(DIRECTION_ORDER):
            index = direction_start + direction_index
            direction_records.append(
                {
                    "source": source,
                    "direction": direction,
                    "mean_probability": float(point[index]),
                    "ci_low": float(lower[index]),
                    "ci_high": float(upper[index]),
                    "unit_count": int(len(values)),
                }
            )
    return pd.DataFrame(offset_records), pd.DataFrame(direction_records)


def plot_profiles(
    offset_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Render a reviewer-facing profile and directional-mass figure."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 10.5),
        gridspec_kw={"height_ratios": [1.45, 1.0]},
        constrained_layout=True,
    )
    top, bottom = axes
    available_sources = [
        source
        for source in SOURCE_ORDER
        if source in set(offset_summary["source"])
    ]
    for source in available_sources:
        profile = (
            offset_summary.loc[offset_summary["source"].eq(source)]
            .sort_values("display_offset")
        )
        x = profile["display_offset"].to_numpy(dtype=float)
        y = profile["mean_probability"].to_numpy(dtype=float)
        low = profile["ci_low"].to_numpy(dtype=float)
        high = profile["ci_high"].to_numpy(dtype=float)
        color = SOURCE_COLORS[source]
        top.plot(
            x,
            y,
            marker="o",
            linewidth=2.8,
            markersize=7,
            color=color,
            label=source,
        )
        top.fill_between(x, low, high, color=color, alpha=0.17)
    top.set_title(
        "Next-fixation word-displacement profile",
        fontsize=17,
        loc="left",
        pad=14,
    )
    top.text(
        0.0,
        1.01,
        (
            "Observed gaze transitions on the aligned 55-passage Provo grid; "
            "edge bins retain the full tails"
        ),
        transform=top.transAxes,
        fontsize=10.5,
        color="#555555",
        va="bottom",
    )
    top.set_xlabel("Next fixated word − currently fixated word", fontsize=12)
    top.set_ylabel("Normalized transition probability", fontsize=12)
    top.set_xticks(OFFSET_VALUES, OFFSET_LABELS)
    top.set_xlim(OFFSET_VALUES.min(), OFFSET_VALUES.max())
    top.set_ylim(
        0,
        max(0.05, float(offset_summary["ci_high"].max()) * 1.15),
    )
    top.legend(frameon=False, fontsize=11, ncol=len(available_sources))

    x_positions = np.arange(len(DIRECTION_ORDER), dtype=float)
    width = 0.34 if len(available_sources) > 1 else 0.52
    for source_index, source in enumerate(available_sources):
        profile = (
            direction_summary.loc[
                direction_summary["source"].eq(source)
            ]
            .set_index("direction")
            .loc[list(DIRECTION_ORDER)]
        )
        if len(available_sources) == 1:
            positions = x_positions
        else:
            positions = x_positions + (
                source_index - (len(available_sources) - 1) / 2
            ) * width
        values = profile["mean_probability"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                values - profile["ci_low"].to_numpy(dtype=float),
                profile["ci_high"].to_numpy(dtype=float) - values,
            ]
        )
        bars = bottom.bar(
            positions,
            values,
            width=width,
            color=SOURCE_COLORS[source],
            alpha=0.85,
            yerr=errors,
            capsize=4,
            label=source,
        )
        for bar, value in zip(bars, values):
            bottom.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{100 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    bottom.set_title(
        "Pooled transition mass by direction",
        fontsize=17,
        loc="left",
        pad=12,
    )
    bottom.set_xticks(
        x_positions,
        ("Regressive (<0)", "Refixation (=0)", "Progressive (>0)"),
    )
    bottom.set_ylabel("Participant/run-balanced probability", fontsize=12)
    bottom.set_ylim(
        0,
        min(1.0, max(0.25, float(direction_summary["ci_high"].max()) + 0.12)),
    )
    bottom.yaxis.set_major_formatter(
        lambda value, position: f"{100 * value:.0f}%"
    )
    if len(available_sources) > 1:
        bottom.legend(frameon=False, fontsize=11)
    figure.savefig(
        output_dir / "next_fixation_word_displacement.png",
        dpi=240,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "next_fixation_word_displacement.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def write_results_markdown(
    output_dir: Path,
    offset_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    audit: dict,
) -> None:
    """Write a compact interpretation and scope statement."""
    human_directions = (
        direction_summary.loc[
            direction_summary["source"].eq("Human Provo")
        ]
        .set_index("direction")
        .loc[list(DIRECTION_ORDER)]
    )
    lines = [
        "# Human Provo next-fixation profile",
        "",
        (
            "This analysis uses the official fixation-by-fixation Provo report. "
            "It does not use ET1 values, ET1 tokenization, or a redistribution "
            "kernel."
        ),
        "",
        (
            "The estimand is overt reading behavior: the word-position "
            "displacement from the current fixation to the next fixation."
        ),
        "",
        "## Directional mass",
        "",
        "| Direction | Mean | 95% reader-bootstrap CI |",
        "|---|---:|---:|",
    ]
    for direction, row in human_directions.iterrows():
        lines.append(
            f"| {direction} | {100 * row['mean_probability']:.2f}% | "
            f"[{100 * row['ci_low']:.2f}%, "
            f"{100 * row['ci_high']:.2f}%] |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "This figure can be compared directly with OB1 fixation "
                "trajectories after the exact current OB1 fixation CSV is "
                "provided. It must not be overlaid as if it were the same "
                "quantity as OB1 fixation-onset covert attention or a "
                "Gaussian redistribution kernel."
            ),
            "",
            f"Human transition rows: {audit['human']['transition_rows']:,}.",
            "",
            f"Human readers: {audit['human']['reader_count']}.",
            "",
            f"Aligned passages: {audit['human']['passage_count']}.",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> dict:
    """Run the complete ET1-free Human Provo profile analysis."""
    fixation_path = args.fixation_report.expanduser().resolve()
    word_path = args.canonical_words.expanduser().resolve()
    passage_path = args.canonical_passages.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (fixation_path, word_path, passage_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fixation_sha256 = sha256_file(fixation_path)
    if (
        not args.allow_unverified_fixation_report
        and fixation_sha256 != EXPECTED_FIXATION_SHA256
    ):
        raise ValueError(
            "Unexpected Provo fixation-report SHA-256: "
            f"{fixation_sha256}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations = load_fixation_report(fixation_path)
    canonical_words = pd.read_csv(word_path)
    passages = pd.read_csv(passage_path)
    mapping = build_interest_area_map(
        fixations.loc[fixations["TRIAL_INDEX"].le(55)],
        passages,
    )
    human_transitions, human_audit = build_human_transitions(
        fixations,
        mapping,
        canonical_words,
    )
    transitions = [human_transitions]
    ob1_audit = None
    ob1_input = None
    if args.ob1_fixations is not None:
        if args.expected_ob1_sha256 is None:
            raise ValueError(
                "--expected-ob1-sha256 is required with --ob1-fixations"
            )
        ob1_path = args.ob1_fixations.expanduser().resolve()
        if not ob1_path.is_file():
            raise FileNotFoundError(ob1_path)
        ob1_sha256 = sha256_file(ob1_path)
        if ob1_sha256 != args.expected_ob1_sha256:
            raise ValueError(
                "OB1 fixation SHA-256 mismatch: "
                f"expected {args.expected_ob1_sha256}, found {ob1_sha256}"
            )
        ob1_transitions, ob1_audit = build_ob1_transitions(
            ob1_path,
            canonical_words,
        )
        transitions.append(ob1_transitions)
        ob1_input = {
            "path": str(ob1_path),
            "sha256": ob1_sha256,
        }
    combined_transitions = pd.concat(transitions, ignore_index=True)
    unit_profiles = build_unit_profiles(combined_transitions)
    offset_summary, direction_summary = bootstrap_profile_summary(
        unit_profiles,
        args.bootstrap_samples,
        args.seed,
    )
    audit = {
        "analysis": "provo_next_fixation_word_displacement",
        "estimand": (
            "unit-balanced distribution of next fixated word position minus "
            "current fixated word position"
        ),
        "corpus": "Provo",
        "uses_et1_values": False,
        "uses_et1_tokenization": False,
        "uses_redistribution_kernel": False,
        "behavior_is_covert_attention": False,
        "main_trial_policy": "TRIAL_INDEX <= 55",
        "blink_policy": (
            "CURRENT_FIX_BLINK_AROUND and NEXT_FIX_BLINK_AROUND must both "
            "equal NONE"
        ),
        "coordinate_policy": (
            "both transition endpoints must map one-to-one to the frozen "
            "2686-word OB1-aligned Provo evaluation grid"
        ),
        "display_bin_policy": (
            "word displacements below -3 are retained in <=-3; "
            "displacements above +6 are retained in >=+6"
        ),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "bootstrap_unit_human": "reader",
        "bootstrap_unit_ob1": (
            "simulation" if ob1_audit is not None else None
        ),
        "inputs": {
            "fixation_report": {
                "path": str(fixation_path),
                "bytes": fixation_path.stat().st_size,
                "sha256": fixation_sha256,
            },
            "canonical_words": {
                "path": str(word_path),
                "sha256": sha256_file(word_path),
                "rows": int(len(canonical_words)),
            },
            "canonical_passages": {
                "path": str(passage_path),
                "sha256": sha256_file(passage_path),
                "rows": int(len(passages)),
            },
            "ob1_fixations": ob1_input,
        },
        "interest_area_mapping_status_counts": {
            str(key): int(value)
            for key, value in mapping["mapping_status"].value_counts().items()
        },
        "human": human_audit,
        "ob1": ob1_audit,
    }
    mapping.to_csv(output_dir / "interest_area_alignment.csv", index=False)
    human_transitions.to_csv(
        output_dir / "human_transition_events.csv.gz",
        index=False,
        compression="gzip",
    )
    if len(transitions) > 1:
        transitions[1].to_csv(
            output_dir / "ob1_transition_events.csv.gz",
            index=False,
            compression="gzip",
        )
    unit_profiles.to_csv(output_dir / "unit_level_profiles.csv", index=False)
    offset_summary.to_csv(output_dir / "offset_profile.csv", index=False)
    direction_summary.to_csv(
        output_dir / "direction_profile.csv",
        index=False,
    )
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_profiles(offset_summary, direction_summary, output_dir)
    write_results_markdown(
        output_dir,
        offset_summary,
        direction_summary,
        audit,
    )
    return {
        "output_dir": str(output_dir),
        "human_readers": human_audit["reader_count"],
        "passages": human_audit["passage_count"],
        "human_transitions": human_audit["transition_rows"],
        "ob1_included": ob1_audit is not None,
        "uses_et1": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an ET1-free Human Provo next-fixation word-displacement "
            "profile, with an optional SHA-verified OB1 trajectory overlay."
        )
    )
    parser.add_argument(
        "--fixation-report",
        type=Path,
        default=DEFAULT_FIXATION_PATH,
    )
    parser.add_argument(
        "--canonical-words",
        type=Path,
        default=DEFAULT_WORD_PATH,
    )
    parser.add_argument(
        "--canonical-passages",
        type=Path,
        default=DEFAULT_PASSAGE_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--ob1-fixations", type=Path)
    parser.add_argument("--expected-ob1-sha256")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--allow-unverified-fixation-report",
        action="store_true",
    )
    return parser


def main() -> None:
    """Run the analysis and print a machine-readable completion record."""
    args = build_parser().parse_args()
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Build canonical OneStop tables from the compressed Ordinary Reading CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/onestop"
PROCESSED_DIR = ROOT / "data/processed/onestop"
ONESTOP_ZIP_FILENAME = "ia_Paragraph_ordinary.csv.zip"
ONESTOP_ZIP_MEMBER = "ia_Paragraph_ordinary.csv"
DEFAULT_ONESTOP_ZIP_PATH = RAW_DIR / ONESTOP_ZIP_FILENAME
DEFAULT_CHUNK_SIZE = 50_000

REQUIRED_COLUMNS = (
    "participant_id",
    "TRIAL_INDEX",
    "TRIAL_IA_COUNT",
    "IA_ID",
    "IA_LABEL",
    "IA_DWELL_TIME",
    "IA_FIXATION_COUNT",
    "IA_SKIP",
    "IA_LEFT",
    "IA_TOP",
    "IA_RIGHT",
    "IA_BOTTOM",
    "question_preview",
    "article_batch",
    "article_id",
    "paragraph_id",
    "practice_trial",
    "repeated_reading_trial",
    "difficulty_level",
    "paragraph",
)
BASE_KEY = ["article_batch", "article_id_raw", "paragraph_id"]
TRIAL_KEY = ["participant_id", *BASE_KEY, "TRIAL_INDEX"]
PARTICIPANT_WORD_KEY = ["participant_id", *BASE_KEY, "IA_ID"]
SELECTION_RULE = (
    "max_participant_count_then_lexical_text_then_sha256"
)

EXPECTED_STRICT_PARTICIPANTS = 180
EXPECTED_STRICT_ARTICLES = 30
EXPECTED_STRICT_PASSAGES = 162
EXPECTED_STRICT_CANONICAL_WORDS = 19_440
EXPECTED_STRICT_PARTICIPANT_TRIALS = 4_859
EXPECTED_STRICT_SELECTED_TRIALS = 4_759
EXPECTED_STRICT_MULTI_VARIANT_PASSAGES = 11
EXPECTED_STRICT_READER_COUNT_MIN = 17
EXPECTED_STRICT_READER_COUNT_MAX = 30
EXPECTED_STRICT_PUNCTUATION_ONLY_WORDS = 95


def stable_text_hash(text: str) -> str:
    """Return a deterministic SHA-256 ID for an exact stimulus string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_word(value: str) -> str:
    """Apply the punctuation-removing word normalization used by OB1."""
    return re.sub(r"[^\w\s]", "", str(value)).lower().strip()


def _boolean_series(series: pd.Series, column: str) -> pd.Series:
    """Parse a CSV boolean column without silently accepting unknown values."""

    def parse(value: object) -> bool:
        if pd.isna(value):
            raise ValueError(f"{column} contains a missing value")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"false", "0"}:
            return False
        if text in {"true", "1"}:
            return True
        raise ValueError(f"{column} contains invalid boolean value {value!r}")

    return series.map(parse).astype(bool)


def _integer_series(series: pd.Series, column: str) -> pd.Series:
    """Convert a required numeric column to exact integers."""
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} contains missing or non-numeric values")
    if not bool((values == np.floor(values)).all()):
        raise ValueError(f"{column} contains non-integral values")
    return values.astype("int64")


def _float_series(series: pd.Series, column: str) -> pd.Series:
    """Convert a required numeric column to finite floating-point values."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.isna().any() or not bool(np.isfinite(values).all()):
        raise ValueError(f"{column} contains missing or non-finite values")
    return values


def _zip_member(archive: zipfile.ZipFile) -> str:
    """Resolve the official CSV member and reject ambiguous archives."""
    matches = [
        name
        for name in archive.namelist()
        if Path(name).name == ONESTOP_ZIP_MEMBER
        and not name.startswith("__MACOSX/")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {ONESTOP_ZIP_MEMBER!r} member, found {matches}"
        )
    return matches[0]


def _load_filtered_rows(
    zip_path: Path,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stream required columns and retain primary Advanced Gathering rows."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    participant_chunks: list[pd.DataFrame] = []
    trial_chunks: list[pd.DataFrame] = []
    raw_rows = 0
    filtered_rows = 0
    chunks_processed = 0

    with zipfile.ZipFile(zip_path) as archive:
        member = _zip_member(archive)
        with archive.open(member) as handle:
            reader = pd.read_csv(
                handle,
                usecols=list(REQUIRED_COLUMNS),
                dtype={
                    "participant_id": "string",
                    "IA_LABEL": "string",
                    "question_preview": "string",
                    "practice_trial": "string",
                    "repeated_reading_trial": "string",
                    "difficulty_level": "string",
                    "paragraph": "string",
                },
                keep_default_na=False,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                chunks_processed += 1
                raw_rows += len(chunk)
                question_preview = _boolean_series(
                    chunk["question_preview"],
                    "question_preview",
                )
                repeated = _boolean_series(
                    chunk["repeated_reading_trial"],
                    "repeated_reading_trial",
                )
                if bool(question_preview.any()):
                    raise ValueError(
                        "Ordinary archive contains question_preview=True rows"
                    )
                if bool(repeated.any()):
                    raise ValueError(
                        "Ordinary archive contains "
                        "repeated_reading_trial=True rows"
                    )

                practice = _boolean_series(
                    chunk["practice_trial"],
                    "practice_trial",
                )
                difficulty = (
                    chunk["difficulty_level"].astype("string").str.strip()
                )
                if bool(difficulty.eq("").any()):
                    raise ValueError(
                        "difficulty_level contains an empty value"
                    )
                primary = (~practice) & difficulty.eq("Adv")
                selected = chunk.loc[primary].copy()
                if selected.empty:
                    continue
                filtered_rows += len(selected)

                if selected[
                    ["participant_id", "IA_LABEL", "paragraph"]
                ].isna().any().any():
                    raise ValueError(
                        "Primary OneStop rows contain missing identifiers "
                        "or text"
                    )
                empty_text = selected[
                    ["participant_id", "IA_LABEL", "paragraph"]
                ].apply(
                    lambda column: (
                        column.astype("string").str.strip().eq("")
                    )
                )
                if bool(empty_text.any().any()):
                    raise ValueError(
                        "Primary OneStop rows contain empty identifiers "
                        "or text"
                    )
                selected["participant_id"] = selected[
                    "participant_id"
                ].astype(str)
                selected["IA_LABEL"] = selected["IA_LABEL"].astype(str)
                selected["paragraph"] = selected["paragraph"].astype(str)
                selected = selected.rename(
                    columns={"article_id": "article_id_raw"}
                )

                for column in (
                    "TRIAL_INDEX",
                    "TRIAL_IA_COUNT",
                    "IA_ID",
                    "IA_FIXATION_COUNT",
                    "IA_SKIP",
                    "article_batch",
                    "article_id_raw",
                    "paragraph_id",
                ):
                    selected[column] = _integer_series(
                        selected[column],
                        column,
                    )
                for column in (
                    "IA_DWELL_TIME",
                    "IA_LEFT",
                    "IA_TOP",
                    "IA_RIGHT",
                    "IA_BOTTOM",
                ):
                    selected[column] = _float_series(
                        selected[column],
                        column,
                    )
                if bool((selected["IA_DWELL_TIME"] < 0).any()):
                    raise ValueError("IA_DWELL_TIME contains negative values")
                if bool((selected["IA_FIXATION_COUNT"] < 0).any()):
                    raise ValueError(
                        "IA_FIXATION_COUNT contains negative values"
                    )
                if not set(selected["IA_SKIP"].unique()).issubset({0, 1}):
                    raise ValueError("IA_SKIP must contain only 0 and 1")
                fixation_zero = selected["IA_FIXATION_COUNT"].eq(0)
                dwell_zero = selected["IA_DWELL_TIME"].eq(0)
                if bool(fixation_zero.ne(dwell_zero).any()):
                    raise ValueError(
                        "IA_DWELL_TIME and IA_FIXATION_COUNT disagree on "
                        "whether a word was ever fixated"
                    )

                selected["variant_hash"] = selected["paragraph"].map(
                    stable_text_hash
                )
                compact_columns = [
                    *TRIAL_KEY,
                    "TRIAL_IA_COUNT",
                    "IA_ID",
                    "IA_LABEL",
                    "IA_DWELL_TIME",
                    "IA_FIXATION_COUNT",
                    "IA_SKIP",
                    "IA_LEFT",
                    "IA_TOP",
                    "IA_RIGHT",
                    "IA_BOTTOM",
                    "variant_hash",
                ]
                participant_chunks.append(selected[compact_columns])
                trial_chunks.append(
                    selected[
                        [
                            *TRIAL_KEY,
                            "TRIAL_IA_COUNT",
                            "variant_hash",
                            "paragraph",
                        ]
                    ].drop_duplicates()
                )

    if not participant_chunks:
        raise ValueError("No non-practice Advanced rows were found")

    participant_rows = pd.concat(
        participant_chunks,
        ignore_index=True,
    )
    trial_fragments = pd.concat(
        trial_chunks,
        ignore_index=True,
    ).drop_duplicates()
    stream_audit = {
        "zip_member": member,
        "chunksize": int(chunksize),
        "chunks_processed": int(chunks_processed),
        "raw_rows": int(raw_rows),
        "primary_filtered_rows": int(filtered_rows),
        "ordinary_question_preview_asserted_false": True,
        "ordinary_repeated_reading_asserted_false": True,
    }
    return participant_rows, trial_fragments, stream_audit


def _validate_source_grid(
    participant_rows: pd.DataFrame,
    trial_fragments: pd.DataFrame,
) -> pd.DataFrame:
    """Reject duplicate participant words and inconsistent trial metadata."""
    duplicate_mask = participant_rows.duplicated(
        PARTICIPANT_WORD_KEY,
        keep=False,
    )
    if bool(duplicate_mask.any()):
        example = (
            participant_rows.loc[duplicate_mask, PARTICIPANT_WORD_KEY]
            .head(5)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate participant-word rows: {example}")

    metadata_columns = ["TRIAL_IA_COUNT", "variant_hash", "paragraph"]
    metadata_counts = trial_fragments.groupby(
        TRIAL_KEY,
        sort=False,
    )[metadata_columns].nunique(dropna=False)
    invalid_metadata = metadata_counts.ne(1).any(axis=1)
    if bool(invalid_metadata.any()):
        raise ValueError(
            "A trial contains inconsistent text or IA-count metadata"
        )
    trials = trial_fragments.drop_duplicates(TRIAL_KEY).copy()

    participant_base_counts = trials.groupby(
        ["participant_id", *BASE_KEY],
        sort=False,
    )["TRIAL_INDEX"].nunique()
    if bool((participant_base_counts != 1).any()):
        raise ValueError(
            "A participant has multiple first-reading trials for one paragraph"
        )

    hash_text_counts = trials.groupby(
        "variant_hash",
        sort=False,
    )["paragraph"].nunique()
    if bool((hash_text_counts != 1).any()):
        raise ValueError("A SHA-256 variant hash maps to multiple texts")
    return trials


def _validate_all_trial_grids(
    participant_rows: pd.DataFrame,
    trials: pd.DataFrame,
) -> None:
    """Validate complete word grids before any text variant is excluded."""
    metadata = {
        tuple(getattr(row, column) for column in TRIAL_KEY): row
        for row in trials.itertuples(index=False)
    }
    observed_trial_keys = set()
    for key, trial in participant_rows.groupby(TRIAL_KEY, sort=False):
        key_tuple = tuple(key) if isinstance(key, tuple) else (key,)
        observed_trial_keys.add(key_tuple)
        if key_tuple not in metadata:
            raise ValueError(f"Missing trial metadata for {key_tuple}")
        trial_metadata = metadata[key_tuple]
        trial = trial.sort_values("IA_ID")
        expected_tokens = str(trial_metadata.paragraph).split()
        expected_ids = list(range(1, len(expected_tokens) + 1))
        actual_ids = trial["IA_ID"].astype(int).tolist()
        if actual_ids != expected_ids:
            raise ValueError(
                f"Incomplete IA grid or non-sequential IA_ID at "
                f"{key_tuple}: {actual_ids}"
            )
        if int(trial_metadata.TRIAL_IA_COUNT) != len(expected_tokens):
            raise ValueError(
                f"TRIAL_IA_COUNT does not match paragraph at {key_tuple}"
            )
        if trial["IA_LABEL"].astype(str).tolist() != expected_tokens:
            raise ValueError(
                f"IA_LABEL sequence does not match paragraph.split() at "
                f"{key_tuple}"
            )
    if observed_trial_keys != set(metadata):
        raise ValueError("Trial metadata and participant-word grids differ")


def _variant_table(
    participant_rows: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Count and deterministically rank exact paragraph-text variants."""
    variants = (
        trials.groupby(
            [*BASE_KEY, "variant_hash", "paragraph"],
            sort=True,
            as_index=False,
        )
        .agg(
            participant_count=("participant_id", "nunique"),
            participant_trial_count=("participant_id", "size"),
        )
    )
    word_counts = (
        participant_rows.groupby(
            [*BASE_KEY, "variant_hash"],
            sort=True,
        )
        .size()
        .rename("participant_word_rows")
        .reset_index()
    )
    variants = variants.merge(
        word_counts,
        on=[*BASE_KEY, "variant_hash"],
        how="left",
        validate="one_to_one",
    )
    variants = variants.sort_values(
        [*BASE_KEY, "participant_count", "paragraph", "variant_hash"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    variants["variant_rank"] = (
        variants.groupby(BASE_KEY, sort=False).cumcount() + 1
    )
    variants["selected"] = variants["variant_rank"].eq(1)
    variants["excluded_participant_count"] = np.where(
        variants["selected"],
        0,
        variants["participant_count"],
    ).astype(int)
    variants["selection_rule"] = SELECTION_RULE
    return variants


def _passage_mapping(variants: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic article, cluster, and passage identifiers."""
    articles = (
        variants[["article_batch", "article_id_raw"]]
        .drop_duplicates()
        .sort_values(["article_batch", "article_id_raw"])
        .reset_index(drop=True)
    )
    articles["article_id"] = np.arange(1, len(articles) + 1, dtype=int)
    articles["cluster_id"] = articles["article_id"]

    selected = variants.loc[variants["selected"]].copy()
    selected = selected.sort_values(BASE_KEY).reset_index(drop=True)
    selected["passage_id_raw"] = np.arange(
        1,
        len(selected) + 1,
        dtype=int,
    )
    selected["passage_id_zero_based"] = selected["passage_id_raw"] - 1
    selected = selected.merge(
        articles,
        on=["article_batch", "article_id_raw"],
        how="left",
        validate="many_to_one",
    )
    selected = selected.rename(
        columns={
            "paragraph": "passage_text",
            "participant_count": "selected_reader_count",
        }
    )
    variant_summary = (
        variants.groupby(BASE_KEY, sort=True)
        .agg(
            text_variant_count=("variant_hash", "size"),
            total_reader_trials=("participant_count", "sum"),
            excluded_variant_reader_count=(
                "excluded_participant_count",
                "sum",
            ),
        )
        .reset_index()
    )
    selected = selected.merge(
        variant_summary,
        on=BASE_KEY,
        how="left",
        validate="one_to_one",
    )
    selected["word_count"] = selected["passage_text"].map(
        lambda text: len(text.split())
    )
    return selected


def _selected_participant_rows(
    participant_rows: pd.DataFrame,
    passage_map: pd.DataFrame,
) -> pd.DataFrame:
    """Keep majority text variants and attach canonical passage coordinates."""
    mapping_columns = [
        *BASE_KEY,
        "variant_hash",
        "passage_id_raw",
        "passage_id_zero_based",
        "passage_text",
        "article_id",
        "cluster_id",
    ]
    mapping = passage_map[mapping_columns].rename(
        columns={"variant_hash": "selected_variant_hash"}
    )
    selected = participant_rows.merge(
        mapping,
        on=BASE_KEY,
        how="inner",
        validate="many_to_one",
    )
    selected = selected.loc[
        selected["variant_hash"] == selected["selected_variant_hash"]
    ].copy()
    if selected.empty:
        raise ValueError("Variant selection removed every participant row")
    return selected


def _validate_selected_trials(selected: pd.DataFrame) -> None:
    """Validate complete sequential word grids against exact selected text."""
    duplicate_mask = selected.duplicated(
        ["participant_id", "passage_id_zero_based", "IA_ID"],
        keep=False,
    )
    if bool(duplicate_mask.any()):
        raise ValueError("Selected rows contain duplicate participant words")

    group_columns = [
        "participant_id",
        "passage_id_zero_based",
        "TRIAL_INDEX",
    ]
    for key, trial in selected.groupby(group_columns, sort=False):
        trial = trial.sort_values("IA_ID")
        text_values = trial["passage_text"].unique()
        count_values = trial["TRIAL_IA_COUNT"].unique()
        if len(text_values) != 1 or len(count_values) != 1:
            raise ValueError(f"Inconsistent selected trial metadata at {key}")
        expected_tokens = str(text_values[0]).split()
        expected_ids = list(range(1, len(expected_tokens) + 1))
        actual_ids = trial["IA_ID"].astype(int).tolist()
        if actual_ids != expected_ids:
            raise ValueError(
                f"Incomplete IA grid or non-sequential IA_ID at {key}: "
                f"{actual_ids}"
            )
        if int(count_values[0]) != len(expected_tokens):
            raise ValueError(
                f"TRIAL_IA_COUNT does not match selected text at {key}"
            )
        if trial["IA_LABEL"].astype(str).tolist() != expected_tokens:
            raise ValueError(
                f"IA_LABEL sequence does not match paragraph.split() at {key}"
            )


def _build_passages(passage_map: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical passage table expected by ET and OB1 runners."""
    passages = passage_map[
        [
            "passage_id_raw",
            "passage_id_zero_based",
            "passage_text",
            "article_id",
            "cluster_id",
            "article_batch",
            "article_id_raw",
            "paragraph_id",
            "word_count",
            "selected_reader_count",
            "total_reader_trials",
            "text_variant_count",
            "excluded_variant_reader_count",
            "variant_hash",
        ]
    ].copy()
    passages["character_count"] = passages["passage_text"].str.len()
    passages["word_count_after_text_correction"] = passages["word_count"]
    passages["evaluable_word_count"] = passages["word_count"]
    passages = passages.rename(
        columns={"variant_hash": "selected_variant_hash"}
    )
    return passages.sort_values("passage_id_zero_based").reset_index(drop=True)


def _build_words(
    selected: pd.DataFrame,
    passages: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate exact participant rows into canonical Human TRT targets."""
    passage_lookup = passages.set_index("passage_id_zero_based")
    records: list[dict] = []
    for (passage_id, ia_id), group in selected.groupby(
        ["passage_id_zero_based", "IA_ID"],
        sort=True,
    ):
        passage = passage_lookup.loc[int(passage_id)]
        labels = group["IA_LABEL"].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(
                f"Multiple IA labels at passage {passage_id}, IA {ia_id}"
            )
        word_raw = labels[0]
        matches = list(re.finditer(r"\S+", str(passage["passage_text"])))
        word_id_zero_based = int(ia_id) - 1
        if word_id_zero_based < 0 or word_id_zero_based >= len(matches):
            raise ValueError(
                f"IA_ID {ia_id} is outside passage {passage_id}"
            )
        match = matches[word_id_zero_based]
        if match.group() != word_raw:
            raise ValueError(
                f"Offset word mismatch at passage {passage_id}, IA {ia_id}"
            )

        trt = group["IA_DWELL_TIME"].astype(float)
        positive = trt[trt > 0]
        ob1_word_normalized = normalize_word(word_raw)
        records.append(
            {
                "passage_id_raw": int(passage["passage_id_raw"]),
                "passage_id_zero_based": int(passage_id),
                "word_number_one_based": int(ia_id),
                "word_id_zero_based": word_id_zero_based,
                "word_raw": word_raw,
                "word_clean": ob1_word_normalized,
                "ob1_word_normalized": ob1_word_normalized,
                "ob1_evaluable": bool(ob1_word_normalized),
                "character_start": int(match.start()),
                "character_end": int(match.end()),
                "eye_word_number_raw": str(int(ia_id)),
                "eye_word_raw": word_raw,
                "eye_word_clean": ob1_word_normalized,
                "cluster_id": int(passage["cluster_id"]),
                "article_id": int(passage["article_id"]),
                "article_batch": int(passage["article_batch"]),
                "article_id_raw": int(passage["article_id_raw"]),
                "paragraph_id": int(passage["paragraph_id"]),
                "human_reader_count": int(group["participant_id"].nunique()),
                "human_positive_reader_count": int((trt > 0).sum()),
                "human_total_skip_count": int((trt == 0).sum()),
                "human_first_pass_skip_count": int(
                    group["IA_SKIP"].astype(int).sum()
                ),
                "human_trt_unconditional": float(trt.mean()),
                "human_trt_conditional": (
                    float(positive.mean()) if len(positive) else np.nan
                ),
                "alignment_status": "exact_whitespace_token",
                "source_variant_hash": passage["selected_variant_hash"],
            }
        )
    return pd.DataFrame(records).sort_values(
        ["passage_id_zero_based", "word_id_zero_based"]
    ).reset_index(drop=True)


def _attach_ob1_compatibility_counts(
    passages: pd.DataFrame,
    words: pd.DataFrame,
) -> pd.DataFrame:
    """Attach punctuation-only word counts without changing the word grid."""
    excluded = (
        ~words["ob1_evaluable"].astype(bool)
    ).groupby(words["passage_id_zero_based"]).sum()
    result = passages.copy()
    result["ob1_excluded_word_count"] = (
        result["passage_id_zero_based"]
        .map(excluded)
        .fillna(0)
        .astype(int)
    )
    return result


def _build_participant_words(selected: pd.DataFrame) -> pd.DataFrame:
    """Create the selected participant-by-word table for later audit."""
    frame = selected.copy()
    frame["word_number_one_based"] = frame["IA_ID"].astype(int)
    frame["word_id_zero_based"] = frame["word_number_one_based"] - 1
    frame["word_raw"] = frame["IA_LABEL"].astype(str)
    frame["human_trt"] = frame["IA_DWELL_TIME"].astype(float)
    frame["is_total_skip"] = frame["IA_DWELL_TIME"].eq(0)
    frame["is_first_pass_skip"] = frame["IA_SKIP"].eq(1)
    columns = [
        "participant_id",
        "passage_id_raw",
        "passage_id_zero_based",
        "word_number_one_based",
        "word_id_zero_based",
        "word_raw",
        "cluster_id",
        "article_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
        "TRIAL_INDEX",
        "TRIAL_IA_COUNT",
        "human_trt",
        "IA_DWELL_TIME",
        "IA_FIXATION_COUNT",
        "IA_SKIP",
        "is_total_skip",
        "is_first_pass_skip",
        "IA_LEFT",
        "IA_TOP",
        "IA_RIGHT",
        "IA_BOTTOM",
        "variant_hash",
    ]
    return frame[columns].sort_values(
        ["passage_id_zero_based", "participant_id", "word_id_zero_based"]
    ).reset_index(drop=True)


def _attach_variant_coordinates(
    variants: pd.DataFrame,
    passage_map: pd.DataFrame,
) -> pd.DataFrame:
    """Attach canonical article and passage IDs to every variant row."""
    mapping = passage_map[
        [
            *BASE_KEY,
            "passage_id_raw",
            "passage_id_zero_based",
            "article_id",
            "cluster_id",
        ]
    ]
    result = variants.merge(
        mapping,
        on=BASE_KEY,
        how="left",
        validate="many_to_one",
    ).rename(columns={"paragraph": "variant_text"})
    columns = [
        "passage_id_raw",
        "passage_id_zero_based",
        "cluster_id",
        "article_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
        "variant_rank",
        "variant_hash",
        "variant_text",
        "participant_count",
        "participant_trial_count",
        "participant_word_rows",
        "selected",
        "excluded_participant_count",
        "selection_rule",
    ]
    return result[columns].sort_values(
        ["passage_id_zero_based", "variant_rank"]
    ).reset_index(drop=True)


def validate_onestop_tables(
    passages: pd.DataFrame,
    words: pd.DataFrame,
    participant_words: pd.DataFrame,
    variant_audit: pd.DataFrame,
    audit: dict,
    strict: bool = False,
) -> None:
    """Enforce general invariants and optional official-release dimensions."""
    if passages.empty or words.empty or participant_words.empty:
        raise ValueError("OneStop canonical tables must not be empty")
    expected_passage_ids = list(range(len(passages)))
    if passages["passage_id_zero_based"].astype(int).tolist() != (
        expected_passage_ids
    ):
        raise ValueError("Passage IDs must be contiguous and zero based")
    expected_clusters = list(range(1, passages["cluster_id"].nunique() + 1))
    actual_clusters = sorted(passages["cluster_id"].astype(int).unique())
    if actual_clusters != expected_clusters:
        raise ValueError("Cluster IDs must be contiguous and one based")
    if not bool((passages["article_id"] == passages["cluster_id"]).all()):
        raise ValueError("Global article_id and cluster_id must match")
    if words.duplicated(
        ["passage_id_zero_based", "word_id_zero_based"]
    ).any():
        raise ValueError("Canonical word coordinates are not unique")
    if participant_words.duplicated(
        ["participant_id", "passage_id_zero_based", "word_id_zero_based"]
    ).any():
        raise ValueError("Participant-word coordinates are not unique")
    required_ob1_columns = {
        "ob1_word_normalized",
        "ob1_evaluable",
    }
    missing_ob1_columns = required_ob1_columns.difference(words.columns)
    if missing_ob1_columns:
        raise ValueError(
            f"Canonical words are missing OB1 fields: "
            f"{sorted(missing_ob1_columns)}"
        )
    if "ob1_excluded_word_count" not in passages.columns:
        raise ValueError(
            "Canonical passages are missing ob1_excluded_word_count"
        )
    expected_normalized = words["word_raw"].map(normalize_word)
    if not words["ob1_word_normalized"].astype(str).equals(
        expected_normalized.astype(str)
    ):
        raise ValueError(
            "ob1_word_normalized does not match normalize_word(word_raw)"
        )
    expected_evaluable = expected_normalized.ne("")
    if not words["ob1_evaluable"].astype(bool).equals(expected_evaluable):
        raise ValueError(
            "ob1_evaluable must be true exactly for nonempty normalized words"
        )
    if int(passages["word_count"].sum()) != len(words):
        raise ValueError(
            "Canonical human word grid changed while marking OB1 compatibility"
        )
    excluded_by_passage = (
        ~expected_evaluable
    ).groupby(words["passage_id_zero_based"]).sum()
    expected_excluded_counts = (
        passages["passage_id_zero_based"]
        .map(excluded_by_passage)
        .fillna(0)
        .astype(int)
    )
    actual_excluded_counts = passages[
        "ob1_excluded_word_count"
    ].astype(int)
    if not actual_excluded_counts.equals(expected_excluded_counts):
        raise ValueError(
            "Per-passage OB1 excluded-word counts disagree with word flags"
        )
    punctuation_only_words = int((~expected_evaluable).sum())
    if int(actual_excluded_counts.sum()) != punctuation_only_words:
        raise ValueError(
            "Total OB1 excluded-word count disagrees with word flags"
        )
    if int(audit.get("ob1_punctuation_only_words", -1)) != (
        punctuation_only_words
    ):
        raise ValueError(
            "OB1 punctuation-only audit count disagrees with word flags"
        )
    if not bool(
        (
            variant_audit.groupby("passage_id_zero_based")["selected"].sum()
            == 1
        ).all()
    ):
        raise ValueError(
            "Every passage must have exactly one selected variant"
        )

    for passage in passages.itertuples():
        passage_words = words.loc[
            words["passage_id_zero_based"]
            == int(passage.passage_id_zero_based)
        ].sort_values("word_id_zero_based")
        expected_word_ids = list(range(int(passage.word_count)))
        if passage_words["word_id_zero_based"].astype(int).tolist() != (
            expected_word_ids
        ):
            raise ValueError(
                f"Non-contiguous canonical word IDs in passage "
                f"{passage.passage_id_zero_based}"
            )
        if not bool(
            (
                passage_words["human_reader_count"]
                == int(passage.selected_reader_count)
            ).all()
        ):
            raise ValueError(
                f"Missing participant-word rows in passage "
                f"{passage.passage_id_zero_based}"
            )

    if strict:
        expectations = {
            "participants": EXPECTED_STRICT_PARTICIPANTS,
            "articles": EXPECTED_STRICT_ARTICLES,
            "passages": EXPECTED_STRICT_PASSAGES,
            "canonical_words": EXPECTED_STRICT_CANONICAL_WORDS,
            "participant_trials_before_variant_selection": (
                EXPECTED_STRICT_PARTICIPANT_TRIALS
            ),
            "participant_trials_selected": (
                EXPECTED_STRICT_SELECTED_TRIALS
            ),
            "multi_variant_passages": (
                EXPECTED_STRICT_MULTI_VARIANT_PASSAGES
            ),
            "reader_count_min": EXPECTED_STRICT_READER_COUNT_MIN,
            "reader_count_max": EXPECTED_STRICT_READER_COUNT_MAX,
            "ob1_punctuation_only_words": (
                EXPECTED_STRICT_PUNCTUATION_ONLY_WORDS
            ),
        }
        for field, expected in expectations.items():
            actual = int(audit[field])
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} {field}, found {actual}"
                )


def validate_loaded_onestop_model_tables(
    passages: pd.DataFrame,
    words: pd.DataFrame,
    audit: dict,
    strict: bool = True,
) -> None:
    """Validate saved model-facing tables without loading participant rows."""
    required_passage_columns = {
        "passage_id_raw",
        "passage_id_zero_based",
        "passage_text",
        "article_id",
        "cluster_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
        "word_count",
        "word_count_after_text_correction",
        "evaluable_word_count",
        "selected_reader_count",
        "ob1_excluded_word_count",
    }
    required_word_columns = {
        "passage_id_raw",
        "passage_id_zero_based",
        "word_number_one_based",
        "word_id_zero_based",
        "word_raw",
        "word_clean",
        "ob1_word_normalized",
        "ob1_evaluable",
        "character_start",
        "character_end",
        "cluster_id",
        "article_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
        "human_reader_count",
        "human_trt_unconditional",
        "human_trt_conditional",
    }
    missing_passage_columns = required_passage_columns.difference(
        passages.columns
    )
    missing_word_columns = required_word_columns.difference(words.columns)
    if missing_passage_columns or missing_word_columns:
        raise ValueError(
            "Saved OneStop tables have a stale or incomplete schema: "
            f"passages={sorted(missing_passage_columns)}, "
            f"words={sorted(missing_word_columns)}"
        )
    if passages.empty or words.empty:
        raise ValueError("Saved OneStop model tables must not be empty")

    passage_ids = _integer_series(
        passages["passage_id_zero_based"],
        "passage_id_zero_based",
    )
    passage_raw_ids = _integer_series(
        passages["passage_id_raw"],
        "passage_id_raw",
    )
    if passage_ids.tolist() != list(range(len(passages))):
        raise ValueError(
            "Saved OneStop passage IDs must be sorted and contiguous from zero"
        )
    if not passage_raw_ids.equals(passage_ids + 1):
        raise ValueError(
            "Saved OneStop raw passage IDs must equal zero-based IDs plus one"
        )
    if passages["passage_text"].isna().any():
        raise ValueError("Saved OneStop passages contain missing text")
    if passages.duplicated(
        ["article_batch", "article_id_raw", "paragraph_id"]
    ).any():
        raise ValueError("Saved OneStop raw passage keys are not unique")

    article_ids = _integer_series(passages["article_id"], "article_id")
    cluster_ids = _integer_series(passages["cluster_id"], "cluster_id")
    if not article_ids.equals(cluster_ids):
        raise ValueError(
            "Saved OneStop article and cluster coordinates must match"
        )
    expected_clusters = list(range(1, int(cluster_ids.nunique()) + 1))
    if sorted(cluster_ids.unique().tolist()) != expected_clusters:
        raise ValueError(
            "Saved OneStop cluster IDs must be contiguous from one"
        )

    coordinate_columns = [
        "passage_id_zero_based",
        "word_id_zero_based",
    ]
    if words.duplicated(coordinate_columns).any():
        raise ValueError("Saved OneStop word coordinates are not unique")
    word_passage_ids = _integer_series(
        words["passage_id_zero_based"],
        "word passage_id_zero_based",
    )
    if set(word_passage_ids.unique()) != set(passage_ids):
        raise ValueError(
            "Saved OneStop passage and word tables have different passage IDs"
        )
    word_raw_passage_ids = _integer_series(
        words["passage_id_raw"],
        "word passage_id_raw",
    )
    if not word_raw_passage_ids.equals(word_passage_ids + 1):
        raise ValueError(
            "Saved OneStop word raw passage IDs do not match their passages"
        )

    passage_metadata_columns = [
        "passage_id_zero_based",
        "article_id",
        "cluster_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
        "selected_reader_count",
    ]
    attached = words.merge(
        passages[passage_metadata_columns],
        on="passage_id_zero_based",
        how="left",
        suffixes=("_word", "_passage"),
        indicator=True,
        validate="many_to_one",
    )
    if not bool(attached["_merge"].eq("both").all()):
        raise ValueError("Saved OneStop words reference unknown passages")
    for column in (
        "article_id",
        "cluster_id",
        "article_batch",
        "article_id_raw",
        "paragraph_id",
    ):
        word_values = _integer_series(
            attached[f"{column}_word"],
            f"word {column}",
        )
        passage_values = _integer_series(
            attached[f"{column}_passage"],
            f"passage {column}",
        )
        if not word_values.equals(passage_values):
            raise ValueError(
                f"Saved OneStop word {column} values do not match passages"
            )
    reader_counts = _integer_series(
        attached["human_reader_count"],
        "human_reader_count",
    )
    selected_reader_counts = _integer_series(
        attached["selected_reader_count"],
        "selected_reader_count",
    )
    if bool((reader_counts <= 0).any()):
        raise ValueError("Saved OneStop Human reader counts must be positive")
    if bool((selected_reader_counts <= 0).any()):
        raise ValueError(
            "Saved OneStop selected reader counts must be positive"
        )
    if not reader_counts.equals(selected_reader_counts):
        raise ValueError(
            "Saved OneStop Human reader counts do not match passages"
        )
    unconditional_trt = _float_series(
        words["human_trt_unconditional"],
        "human_trt_unconditional",
    )
    if bool((unconditional_trt < 0).any()):
        raise ValueError(
            "Saved OneStop unconditional Human TRT must be nonnegative"
        )
    conditional_raw = words["human_trt_conditional"]
    conditional_trt = pd.to_numeric(
        conditional_raw,
        errors="coerce",
    ).astype(float)
    invalid_conditional = conditional_raw.notna() & conditional_trt.isna()
    if bool(invalid_conditional.any()):
        raise ValueError(
            "Saved OneStop conditional Human TRT contains non-numeric values"
        )
    finite_conditional = conditional_trt.dropna()
    if not bool(np.isfinite(finite_conditional).all()):
        raise ValueError(
            "Saved OneStop conditional Human TRT must be finite when present"
        )
    if bool((finite_conditional <= 0).any()):
        raise ValueError(
            "Saved OneStop conditional Human TRT must be positive when present"
        )

    word_numbers = _integer_series(
        words["word_number_one_based"],
        "word_number_one_based",
    )
    word_ids = _integer_series(
        words["word_id_zero_based"],
        "word_id_zero_based",
    )
    if not word_numbers.equals(word_ids + 1):
        raise ValueError(
            "Saved OneStop one-based and zero-based word IDs disagree"
        )
    normalized_words = words["word_raw"].astype(str).map(normalize_word)
    stored_normalized = (
        words["ob1_word_normalized"].fillna("").astype(str)
    )
    if not stored_normalized.equals(normalized_words):
        raise ValueError(
            "Saved OneStop OB1 normalized words do not match word_raw"
        )
    stored_clean = words["word_clean"].fillna("").astype(str)
    if not stored_clean.equals(normalized_words):
        raise ValueError(
            "Saved OneStop cleaned words do not match word_raw"
        )
    ob1_evaluable = _boolean_series(
        words["ob1_evaluable"],
        "ob1_evaluable",
    )
    if not ob1_evaluable.equals(normalized_words.ne("")):
        raise ValueError(
            "Saved OneStop OB1 eligibility flags do not match normalization"
        )

    excluded_by_passage = (
        ~ob1_evaluable
    ).groupby(word_passage_ids).sum()
    for passage in passages.itertuples(index=False):
        passage_id = int(passage.passage_id_zero_based)
        passage_words = words.loc[
            word_passage_ids == passage_id
        ].sort_values("word_id_zero_based")
        tokens = str(passage.passage_text).split()
        expected_count = len(tokens)
        count_fields = (
            int(passage.word_count),
            int(passage.word_count_after_text_correction),
            int(passage.evaluable_word_count),
        )
        if count_fields != (
            expected_count,
            expected_count,
            expected_count,
        ):
            raise ValueError(
                f"Saved OneStop passage {passage_id} word counts disagree "
                "with passage text"
            )
        if passage_words["word_id_zero_based"].astype(int).tolist() != (
            list(range(expected_count))
        ):
            raise ValueError(
                f"Saved OneStop passage {passage_id} has a non-contiguous "
                "word grid"
            )
        if passage_words["word_raw"].astype(str).tolist() != tokens:
            raise ValueError(
                f"Saved OneStop passage {passage_id} word text does not "
                "match passage_text"
            )
        matches = list(re.finditer(r"\S+", str(passage.passage_text)))
        if passage_words["character_start"].astype(int).tolist() != [
            int(match.start()) for match in matches
        ]:
            raise ValueError(
                f"Saved OneStop passage {passage_id} character starts changed"
            )
        if passage_words["character_end"].astype(int).tolist() != [
            int(match.end()) for match in matches
        ]:
            raise ValueError(
                f"Saved OneStop passage {passage_id} character ends changed"
            )
        expected_excluded = int(excluded_by_passage.get(passage_id, 0))
        if int(passage.ob1_excluded_word_count) != expected_excluded:
            raise ValueError(
                f"Saved OneStop passage {passage_id} OB1 exclusion count "
                "disagrees with word flags"
            )

    derived_audit = {
        "passages": len(passages),
        "articles": int(article_ids.nunique()),
        "canonical_words": len(words),
        "reader_count_min": int(reader_counts.min()),
        "reader_count_max": int(reader_counts.max()),
        "ob1_punctuation_only_words": int((~ob1_evaluable).sum()),
    }
    for field, expected in derived_audit.items():
        if field not in audit or int(audit[field]) != expected:
            raise ValueError(
                f"Saved OneStop audit field {field} disagrees with tables"
            )
    if strict:
        strict_expectations = {
            "participants": EXPECTED_STRICT_PARTICIPANTS,
            "articles": EXPECTED_STRICT_ARTICLES,
            "passages": EXPECTED_STRICT_PASSAGES,
            "canonical_words": EXPECTED_STRICT_CANONICAL_WORDS,
            "participant_trials_before_variant_selection": (
                EXPECTED_STRICT_PARTICIPANT_TRIALS
            ),
            "participant_trials_selected": (
                EXPECTED_STRICT_SELECTED_TRIALS
            ),
            "multi_variant_passages": (
                EXPECTED_STRICT_MULTI_VARIANT_PASSAGES
            ),
            "reader_count_min": EXPECTED_STRICT_READER_COUNT_MIN,
            "reader_count_max": EXPECTED_STRICT_READER_COUNT_MAX,
            "ob1_punctuation_only_words": (
                EXPECTED_STRICT_PUNCTUATION_ONLY_WORDS
            ),
        }
        for field, expected in strict_expectations.items():
            if field not in audit or int(audit[field]) != expected:
                raise ValueError(
                    f"Expected {expected} saved OneStop {field}, "
                    f"found {audit.get(field)!r}"
                )


def build_onestop_tables(
    zip_path: Path,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build passage, word, participant-word, variant, and audit artifacts."""
    participant_rows, trial_fragments, stream_audit = _load_filtered_rows(
        Path(zip_path),
        chunksize,
    )
    trials = _validate_source_grid(participant_rows, trial_fragments)
    _validate_all_trial_grids(participant_rows, trials)
    variants = _variant_table(participant_rows, trials)
    passage_map = _passage_mapping(variants)
    selected = _selected_participant_rows(participant_rows, passage_map)
    _validate_selected_trials(selected)

    passages = _build_passages(passage_map)
    words = _build_words(selected, passages)
    passages = _attach_ob1_compatibility_counts(passages, words)
    participant_words = _build_participant_words(selected)
    variant_audit = _attach_variant_coordinates(variants, passage_map)

    audit = {
        **stream_audit,
        "selection_rule": SELECTION_RULE,
        "participants": int(trials["participant_id"].nunique()),
        "articles": int(
            trials[["article_batch", "article_id_raw"]]
            .drop_duplicates()
            .shape[0]
        ),
        "passages": int(len(passages)),
        "participant_trials_before_variant_selection": int(len(trials)),
        "participant_trials_selected": int(
            selected[TRIAL_KEY].drop_duplicates().shape[0]
        ),
        "participant_trials_excluded_as_minority_variants": int(
            variant_audit["excluded_participant_count"].sum()
        ),
        "text_variants": int(len(variant_audit)),
        "multi_variant_passages": int(
            (passages["text_variant_count"] > 1).sum()
        ),
        "canonical_words": int(len(words)),
        "ob1_punctuation_only_words": int(
            (~words["ob1_evaluable"].astype(bool)).sum()
        ),
        "selected_participant_word_rows": int(len(participant_words)),
        "reader_count_min": int(words["human_reader_count"].min()),
        "reader_count_max": int(words["human_reader_count"].max()),
        "zero_dwell_participant_word_rows": int(
            participant_words["is_total_skip"].sum()
        ),
        "first_pass_skip_positive_dwell_rows": int(
            (
                participant_words["is_first_pass_skip"]
                & ~participant_words["is_total_skip"]
            ).sum()
        ),
    }
    validate_onestop_tables(
        passages,
        words,
        participant_words,
        variant_audit,
        audit,
        strict=strict,
    )
    return passages, words, participant_words, variant_audit, audit


def write_audit_json(path: Path, audit: dict) -> None:
    """Write a deterministic JSON audit record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_onestop_tables(
    output_dir: Path,
    passages: pd.DataFrame,
    words: pd.DataFrame,
    participant_words: pd.DataFrame,
    variant_audit: pd.DataFrame,
    audit: dict,
) -> None:
    """Write deterministic canonical OneStop artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    passages.to_csv(output_dir / "onestop_passages.csv", index=False)
    words.to_csv(output_dir / "onestop_words.csv", index=False)
    participant_words.to_csv(
        output_dir / "onestop_participant_words.csv",
        index=False,
    )
    variant_audit.to_csv(
        output_dir / "onestop_variant_audit.csv",
        index=False,
    )
    write_audit_json(
        output_dir / "onestop_prepare_audit.json",
        audit,
    )


def parse_args() -> argparse.Namespace:
    """Parse the official ZIP, streaming, and output options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=DEFAULT_ONESTOP_ZIP_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build, validate, save, and summarize canonical OneStop tables."""
    args = parse_args()
    artifacts = build_onestop_tables(
        args.input_zip,
        chunksize=args.chunksize,
        strict=args.strict,
    )
    write_onestop_tables(args.output_dir, *artifacts)
    print(json.dumps(artifacts[-1], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

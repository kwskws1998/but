"""Tests for streaming canonical OneStop preparation."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

import cognitive_model_comparsion.src.prepare_onestop as prepare_onestop
from cognitive_model_comparsion.src.prepare_onestop import (
    DEFAULT_ONESTOP_ZIP_PATH,
    ONESTOP_ZIP_MEMBER,
    REQUIRED_COLUMNS,
    ROOT,
    build_onestop_tables,
    validate_loaded_onestop_model_tables,
    write_onestop_tables,
)


def add_trial(
    rows: list[dict],
    participant: str,
    trial_index: int,
    article_batch: int,
    article_id: int,
    paragraph_id: int,
    text: str,
    dwell_times: list[float],
    skips: list[int],
    *,
    practice: bool = False,
    repeated: bool = False,
    question_preview: bool = False,
    difficulty: str = "Adv",
) -> None:
    """Append one complete synthetic OneStop IA trial."""
    tokens = text.split()
    assert len(tokens) == len(dwell_times) == len(skips)
    for ia_id, (token, dwell, skip) in enumerate(
        zip(tokens, dwell_times, skips),
        start=1,
    ):
        left = 100 + (ia_id - 1) * 20
        rows.append(
            {
                "participant_id": participant,
                "TRIAL_INDEX": trial_index,
                "TRIAL_IA_COUNT": len(tokens),
                "IA_ID": ia_id,
                "IA_LABEL": token,
                "IA_DWELL_TIME": dwell,
                "IA_FIXATION_COUNT": int(dwell > 0),
                "IA_SKIP": skip,
                "IA_LEFT": left,
                "IA_TOP": 200,
                "IA_RIGHT": left + 15,
                "IA_BOTTOM": 230,
                "question_preview": question_preview,
                "article_batch": article_batch,
                "article_id": article_id,
                "paragraph_id": paragraph_id,
                "practice_trial": practice,
                "repeated_reading_trial": repeated,
                "difficulty_level": difficulty,
                "paragraph": text,
            }
        )


def write_archive(tmp_path, rows: list[dict]):
    """Write a comma-delimited synthetic official-member ZIP."""
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    path = tmp_path / "ia_Paragraph_ordinary.csv.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ONESTOP_ZIP_MEMBER, buffer.getvalue())
    return path


def base_rows() -> list[dict]:
    """Create majority/minority, zero, and first-pass-skip examples."""
    rows: list[dict] = []
    add_trial(
        rows,
        "p1",
        1,
        1,
        1,
        1,
        "alpha beta gamma",
        [0, 50, 100],
        [1, 1, 0],
    )
    add_trial(
        rows,
        "p2",
        2,
        1,
        1,
        1,
        "alpha beta gamma",
        [20, 70, 0],
        [0, 0, 1],
    )
    add_trial(
        rows,
        "p3",
        3,
        1,
        1,
        1,
        "alpha be- ta gamma",
        [10, 20, 30, 40],
        [0, 0, 0, 0],
    )
    add_trial(
        rows,
        "p1",
        4,
        2,
        1,
        1,
        "delta epsilon",
        [30, 0],
        [0, 1],
    )
    add_trial(
        rows,
        "p2",
        5,
        2,
        1,
        1,
        "delta epsilon",
        [50, 90],
        [0, 0],
    )
    add_trial(
        rows,
        "practice",
        6,
        1,
        0,
        1,
        "practice text",
        [10, 10],
        [0, 0],
        practice=True,
    )
    add_trial(
        rows,
        "elementary",
        7,
        1,
        1,
        1,
        "easy text",
        [10, 10],
        [0, 0],
        difficulty="Ele",
    )
    return rows


def test_streaming_build_selects_majority_and_preserves_trt_semantics(
    tmp_path,
):
    """Small chunks preserve positive post-skip TRT and unconditional zeros."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, participant_words, variants, audit = (
        build_onestop_tables(archive, chunksize=2)
    )

    assert len(passages) == 2
    assert passages["cluster_id"].tolist() == [1, 2]
    assert passages["article_id"].tolist() == [1, 2]
    assert passages["passage_text"].tolist() == [
        "alpha beta gamma",
        "delta epsilon",
    ]
    assert passages["word_count"].tolist() == [3, 2]
    assert len(words) == 5
    assert not (tmp_path / ONESTOP_ZIP_MEMBER).exists()
    assert {
        "passage_id_raw",
        "passage_id_zero_based",
        "word_id_zero_based",
        "word_raw",
        "character_start",
        "character_end",
        "cluster_id",
        "article_id",
        "paragraph_id",
        "human_reader_count",
        "human_trt_unconditional",
        "human_trt_conditional",
        "ob1_word_normalized",
        "ob1_evaluable",
    }.issubset(words.columns)
    assert "ob1_excluded_word_count" in passages.columns
    assert passages["ob1_excluded_word_count"].tolist() == [0, 0]

    alpha = words.loc[
        (words["passage_id_zero_based"] == 0)
        & (words["word_raw"] == "alpha")
    ].iloc[0]
    assert alpha["human_reader_count"] == 2
    assert alpha["human_total_skip_count"] == 1
    assert alpha["human_trt_unconditional"] == pytest.approx(10.0)
    assert alpha["human_trt_conditional"] == pytest.approx(20.0)
    assert alpha["character_start"] == 0
    assert alpha["character_end"] == 5

    beta = words.loc[
        (words["passage_id_zero_based"] == 0)
        & (words["word_raw"] == "beta")
    ].iloc[0]
    assert beta["human_first_pass_skip_count"] == 1
    assert beta["human_total_skip_count"] == 0
    assert beta["human_trt_unconditional"] == pytest.approx(60.0)
    assert beta["human_trt_conditional"] == pytest.approx(60.0)

    p1_beta = participant_words.loc[
        (participant_words["participant_id"] == "p1")
        & (participant_words["word_raw"] == "beta")
    ].iloc[0]
    assert p1_beta["human_trt"] == 50
    assert bool(p1_beta["is_first_pass_skip"])
    assert not bool(p1_beta["is_total_skip"])

    first_passage_variants = variants.loc[
        variants["passage_id_zero_based"] == 0
    ]
    assert first_passage_variants["participant_count"].tolist() == [2, 1]
    assert first_passage_variants["selected"].tolist() == [True, False]
    assert audit["multi_variant_passages"] == 1
    assert audit["participant_trials_excluded_as_minority_variants"] == 1
    assert audit["first_pass_skip_positive_dwell_rows"] == 1
    assert audit["ob1_punctuation_only_words"] == 0
    assert audit["chunks_processed"] > 1


def test_punctuation_only_word_preserves_human_grid_and_trt(tmp_path):
    """OB1 incompatibility is marked without dropping the human TRT word."""
    rows: list[dict] = []
    add_trial(
        rows,
        "p1",
        1,
        1,
        1,
        1,
        "alpha — gamma",
        [10, 40, 30],
        [0, 0, 0],
    )
    add_trial(
        rows,
        "p2",
        2,
        1,
        1,
        1,
        "alpha — gamma",
        [20, 60, 50],
        [0, 0, 0],
    )
    archive = write_archive(tmp_path, rows)

    passages, words, participant_words, _, audit = (
        build_onestop_tables(archive, chunksize=2)
    )

    assert passages["word_count"].tolist() == [3]
    assert passages["evaluable_word_count"].tolist() == [3]
    assert passages["ob1_excluded_word_count"].tolist() == [1]
    assert len(words) == 3
    punctuation = words.loc[words["word_raw"] == "—"].iloc[0]
    assert punctuation["word_id_zero_based"] == 1
    assert punctuation["ob1_word_normalized"] == ""
    assert not bool(punctuation["ob1_evaluable"])
    assert punctuation["human_reader_count"] == 2
    assert punctuation["human_trt_unconditional"] == pytest.approx(50.0)
    assert punctuation["human_trt_conditional"] == pytest.approx(50.0)
    participant_punctuation = participant_words.loc[
        participant_words["word_raw"] == "—"
    ]
    assert len(participant_punctuation) == 2
    assert participant_punctuation["human_trt"].tolist() == [40.0, 60.0]
    assert audit["canonical_words"] == 3
    assert audit["ob1_punctuation_only_words"] == 1


def test_variant_tie_uses_lexical_text_rule(tmp_path):
    """Equal participant counts select the lexically first exact text."""
    rows: list[dict] = []
    add_trial(
        rows,
        "p1",
        1,
        1,
        1,
        1,
        "zeta one",
        [10, 20],
        [0, 0],
    )
    add_trial(
        rows,
        "p2",
        2,
        1,
        1,
        1,
        "alpha one",
        [30, 40],
        [0, 0],
    )
    archive = write_archive(tmp_path, rows)

    passages, _, _, variants, _ = build_onestop_tables(
        archive,
        chunksize=1,
    )

    assert passages.iloc[0]["passage_text"] == "alpha one"
    selected = variants.loc[variants["selected"]].iloc[0]
    assert selected["variant_text"] == "alpha one"


def test_missing_participant_word_is_rejected(tmp_path):
    """A missing IA row is not silently converted into a zero TRT."""
    rows = base_rows()
    rows = [
        row
        for row in rows
        if not (
            row["participant_id"] == "p2"
            and row["article_batch"] == 1
            and row["article_id"] == 1
            and row["paragraph_id"] == 1
            and row["IA_ID"] == 2
        )
    ]
    archive = write_archive(tmp_path, rows)

    with pytest.raises(ValueError, match="Incomplete IA grid"):
        build_onestop_tables(archive, chunksize=2)


def test_missing_minority_variant_word_is_rejected_before_selection(tmp_path):
    """Minority variants are counted only after complete-grid auditing."""
    rows = base_rows()
    rows = [
        row
        for row in rows
        if not (
            row["participant_id"] == "p3"
            and row["article_batch"] == 1
            and row["article_id"] == 1
            and row["paragraph_id"] == 1
            and row["IA_ID"] == 2
        )
    ]
    archive = write_archive(tmp_path, rows)

    with pytest.raises(ValueError, match="Incomplete IA grid"):
        build_onestop_tables(archive, chunksize=2)


def test_duplicate_participant_word_is_rejected(tmp_path):
    """Duplicate participant-word records fail before aggregation."""
    rows = base_rows()
    rows.append(dict(rows[0]))
    archive = write_archive(tmp_path, rows)

    with pytest.raises(ValueError, match="Duplicate participant-word"):
        build_onestop_tables(archive, chunksize=3)


def test_literal_na_and_leading_zero_identifier_are_chunk_stable(tmp_path):
    """NA inference cannot alter tokens or participant identifiers."""
    rows: list[dict] = []
    add_trial(
        rows,
        "001",
        1,
        1,
        1,
        1,
        "NA value",
        [10, 20],
        [0, 0],
    )
    archive = write_archive(tmp_path, rows)

    small = build_onestop_tables(archive, chunksize=1)
    large = build_onestop_tables(archive, chunksize=100)

    assert small[0].equals(large[0])
    assert small[1].equals(large[1])
    assert small[2].equals(large[2])
    assert small[0].iloc[0]["passage_text"] == "NA value"
    assert small[1].iloc[0]["word_raw"] == "NA"
    assert small[2].iloc[0]["participant_id"] == "001"


def test_fixation_count_and_dwell_zero_must_agree(tmp_path):
    """A zero dwell with a positive fixation count is rejected."""
    rows = base_rows()
    rows[0]["IA_FIXATION_COUNT"] = 1
    archive = write_archive(tmp_path, rows)

    with pytest.raises(ValueError, match="disagree"):
        build_onestop_tables(archive, chunksize=3)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (
            "question_preview",
            "question_preview=True",
        ),
        (
            "repeated_reading_trial",
            "repeated_reading_trial=True",
        ),
    ],
)
def test_official_ordinary_assertions_cover_all_rows(
    tmp_path,
    field,
    message,
):
    """Ordinary-regime violations fail even before the primary filter."""
    rows = base_rows()
    rows[-1][field] = True
    archive = write_archive(tmp_path, rows)

    with pytest.raises(ValueError, match=message):
        build_onestop_tables(archive, chunksize=4)


def test_strict_dimensions_are_optional(tmp_path):
    """Synthetic data passes normally and fails official strict dimensions."""
    archive = write_archive(tmp_path, base_rows())

    build_onestop_tables(archive, chunksize=5, strict=False)
    with pytest.raises(ValueError, match="Expected 180 participants"):
        build_onestop_tables(archive, chunksize=5, strict=True)


def test_strict_dimensions_include_official_canonical_word_count(
    tmp_path,
    monkeypatch,
):
    """Strict preparation checks the official word count, not only passages."""
    archive = write_archive(tmp_path, base_rows())
    artifacts = build_onestop_tables(
        archive,
        chunksize=5,
        strict=False,
    )
    audit = artifacts[-1]
    constants = {
        "EXPECTED_STRICT_PARTICIPANTS": "participants",
        "EXPECTED_STRICT_ARTICLES": "articles",
        "EXPECTED_STRICT_PASSAGES": "passages",
        "EXPECTED_STRICT_PARTICIPANT_TRIALS": (
            "participant_trials_before_variant_selection"
        ),
        "EXPECTED_STRICT_SELECTED_TRIALS": (
            "participant_trials_selected"
        ),
        "EXPECTED_STRICT_MULTI_VARIANT_PASSAGES": (
            "multi_variant_passages"
        ),
        "EXPECTED_STRICT_READER_COUNT_MIN": "reader_count_min",
        "EXPECTED_STRICT_READER_COUNT_MAX": "reader_count_max",
        "EXPECTED_STRICT_PUNCTUATION_ONLY_WORDS": (
            "ob1_punctuation_only_words"
        ),
    }
    for constant, field in constants.items():
        monkeypatch.setattr(prepare_onestop, constant, int(audit[field]))
    expected_words = int(audit["canonical_words"]) + 1
    monkeypatch.setattr(
        prepare_onestop,
        "EXPECTED_STRICT_CANONICAL_WORDS",
        expected_words,
    )

    with pytest.raises(
        ValueError,
        match=rf"Expected {expected_words} canonical_words",
    ):
        prepare_onestop.validate_onestop_tables(
            *artifacts[:-1],
            audit,
            strict=True,
        )


def test_saved_model_tables_pass_coordinate_validation(tmp_path):
    """Freshly prepared model tables satisfy the standalone-load contract."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )

    validate_loaded_onestop_model_tables(
        passages,
        words,
        audit,
        strict=False,
    )


def test_saved_model_table_validation_rejects_stale_schema(tmp_path):
    """Pre-eligibility OneStop CSVs cannot enter standalone model commands."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    words = words.drop(columns=["ob1_evaluable"])

    with pytest.raises(ValueError, match="stale or incomplete schema"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


def test_saved_model_table_validation_rejects_word_grid_shift(tmp_path):
    """A shifted saved word ID cannot be paired with Human or OB1 values."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    words.loc[1, "word_id_zero_based"] = 7
    words.loc[1, "word_number_one_based"] = 8

    with pytest.raises(ValueError, match="non-contiguous word grid"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


def test_saved_model_table_validation_rejects_cross_file_raw_id(tmp_path):
    """Saved word rows must retain the passage table's one-based raw ID."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    words.loc[0, "passage_id_raw"] = 99

    with pytest.raises(ValueError, match="raw passage IDs"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


def test_saved_model_table_validation_rejects_ob1_flag_drift(tmp_path):
    """Saved OB1 flags and passage exclusion counts cannot diverge."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    words.loc[0, "ob1_evaluable"] = False

    with pytest.raises(ValueError, match="eligibility flags"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "human_trt_unconditional",
            -1,
            "unconditional Human TRT must be nonnegative",
        ),
        (
            "human_trt_unconditional",
            float("inf"),
            "non-finite",
        ),
        (
            "human_trt_unconditional",
            "invalid",
            "non-finite",
        ),
        (
            "human_trt_conditional",
            0,
            "conditional Human TRT must be positive",
        ),
        (
            "human_trt_conditional",
            float("inf"),
            "conditional Human TRT must be finite",
        ),
        (
            "human_trt_conditional",
            "invalid",
            "conditional Human TRT contains non-numeric",
        ),
    ],
)
def test_saved_model_table_validation_rejects_invalid_human_trt(
    tmp_path,
    column,
    value,
    message,
):
    """Saved Human TRT targets must retain their numeric domain."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    if isinstance(value, str):
        words[column] = words[column].astype(object)
    words.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("words", "human_reader_count"),
        ("passages", "selected_reader_count"),
    ],
)
def test_saved_model_table_validation_rejects_nonpositive_readers(
    tmp_path,
    table,
    column,
):
    """Every retained canonical word and passage needs Human readers."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    target = words if table == "words" else passages
    target.loc[0, column] = 0

    with pytest.raises(ValueError, match="reader counts must be positive"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


def test_saved_model_table_validation_rejects_audit_drift(tmp_path):
    """A stale audit cannot certify model tables from another preparation."""
    archive = write_archive(tmp_path, base_rows())
    passages, words, _, _, audit = build_onestop_tables(
        archive,
        chunksize=2,
    )
    audit["canonical_words"] += 1

    with pytest.raises(ValueError, match="audit field canonical_words"):
        validate_loaded_onestop_model_tables(
            passages,
            words,
            audit,
            strict=False,
        )


def test_write_onestop_tables_includes_json_audit(tmp_path):
    """All canonical tables and a deterministic JSON audit are written."""
    archive = write_archive(tmp_path, base_rows())
    artifacts = build_onestop_tables(archive, chunksize=2)
    output_dir = tmp_path / "processed"

    write_onestop_tables(output_dir, *artifacts)

    expected = {
        "onestop_passages.csv",
        "onestop_words.csv",
        "onestop_participant_words.csv",
        "onestop_variant_audit.csv",
        "onestop_prepare_audit.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    audit = json.loads(
        (output_dir / "onestop_prepare_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["passages"] == 2
    assert audit["selection_rule"].startswith("max_participant_count")


def test_default_input_matches_asset_manifest_destination():
    """The preparation default consumes the downloader's OneStop artifact."""
    manifest = json.loads(
        (ROOT / "asset_manifest.json").read_text(encoding="utf-8")
    )
    destination = manifest["assets"][
        "onestop_ordinary_interest_areas"
    ]["destination"]

    assert DEFAULT_ONESTOP_ZIP_PATH == ROOT / destination

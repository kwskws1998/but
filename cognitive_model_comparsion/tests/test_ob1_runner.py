"""Tests for isolated OB1 runtime preparation and TVT aggregation."""

from pathlib import Path

import pandas as pd
import pytest

from cognitive_model_comparsion.src.ob1_runner import (
    aggregate_ob1_tvt,
    derived_cache_filenames,
    merge_ob1_worker_outputs,
    prepare_ob1_runtime,
    split_seed_chunks,
    stimulus_word_coordinates,
    transform_ob1_passage,
    validate_ob1_passages,
    validate_stimulus_name,
)


def test_ob1_tvt_includes_regressions_and_fills_skips():
    """All fixation durations sum into TVT and unfixated words receive zero."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1, 1],
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [1, 2],
            "word_raw": ["one", "two"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0, 0, 1],
            "seed": [10, 10, 10, 11],
            "text_id": [0, 0, 0, 0],
            "word_id": [1, 1, 2, 1],
            "fixation_duration": [100.0, 50.0, 80.0, 120.0],
            "saccade_type": [None, "regression", "forward", None],
        }
    )

    per_simulation, means, audit = aggregate_ob1_tvt(
        fixations,
        canonical,
    )

    sim0_word1 = per_simulation.query(
        "simulation_id == 0 and word_id_zero_based == 1"
    )["ob1_tvt"].item()
    sim1_word2 = per_simulation.query(
        "simulation_id == 1 and word_id_zero_based == 2"
    )["ob1_tvt"].item()
    mean_word1 = means.query("word_id_zero_based == 1")["ob1_tvt"].item()
    assert sim0_word1 == 150.0
    assert sim1_word2 == 0.0
    assert mean_word1 == pytest.approx(135.0)
    assert audit["regressive_fixations"] == 1


def test_ob1_tvt_converts_numeric_duration_strings_before_summing():
    """CSV-compatible duration strings are summed numerically, not joined."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1],
            "passage_id_zero_based": [0],
            "word_id_zero_based": [0],
            "word_raw": ["one"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0],
            "seed": [10, 10],
            "text_id": [0, 0],
            "word_id": [0, 0],
            "fixation_duration": ["100.0", "50.0"],
        }
    )

    per_simulation, means, _ = aggregate_ob1_tvt(fixations, canonical)

    assert per_simulation["ob1_tvt"].item() == pytest.approx(150.0)
    assert means["ob1_tvt"].item() == pytest.approx(150.0)


def test_ob1_tvt_rejects_fixations_outside_canonical_grid():
    """No simulated fixation may be silently dropped during aggregation."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1],
            "passage_id_zero_based": [0],
            "word_id_zero_based": [0],
            "word_raw": ["one"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0],
            "seed": [10, 10],
            "text_id": [0, 9],
            "word_id": [0, 99],
            "fixation_duration": [100.0, 50.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="outside the canonical word grid.*passage_id_zero_based.*9",
    ):
        aggregate_ob1_tvt(fixations, canonical)


def test_ob1_tvt_allows_reported_fixations_outside_evaluation_grid():
    """Full stimulus coordinates permit intentional Provo grid omissions."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1],
            "passage_id_zero_based": [0],
            "word_id_zero_based": [1],
            "word_raw": ["evaluated"],
        }
    )
    full_coordinates = pd.DataFrame(
        {
            "passage_id_zero_based": [0, 0],
            "word_id_zero_based": [0, 1],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0],
            "seed": [10, 10],
            "text_id": [0, 0],
            "word_id": [0, 1],
            "fixation_duration": [50.0, 100.0],
        }
    )

    per_simulation, means, audit = aggregate_ob1_tvt(
        fixations,
        canonical,
        valid_fixation_coordinates=full_coordinates,
    )

    assert per_simulation["word_id_zero_based"].tolist() == [1]
    assert means["ob1_tvt"].tolist() == [100.0]
    assert audit["fixations_outside_evaluation_grid"] == 1
    assert audit["fixation_duration_outside_evaluation_grid"] == 50.0
    assert audit["fixation_coordinates_outside_evaluation_grid"] == 1


def test_stimulus_word_coordinates_include_eval_omissions():
    """Simulation validation follows full passage text, not Human coverage."""
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["first evaluated final"],
        }
    )

    coordinates = stimulus_word_coordinates(passages)

    assert coordinates.to_dict("records") == [
        {"passage_id_zero_based": 0, "word_id_zero_based": 0},
        {"passage_id_zero_based": 0, "word_id_zero_based": 1},
        {"passage_id_zero_based": 0, "word_id_zero_based": 2},
    ]


def test_ob1_tvt_rejects_missing_simulation_passage_coverage():
    """A missing simulated passage cannot be converted to an all-zero TVT."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1, 2],
            "passage_id_zero_based": [0, 1],
            "word_id_zero_based": [0, 0],
            "word_raw": ["one", "two"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0],
            "seed": [10],
            "text_id": [0],
            "word_id": [0],
            "fixation_duration": [100.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing every fixation.*simulation_id.*passage_id_zero_based",
    ):
        aggregate_ob1_tvt(fixations, canonical)


def test_ob1_tvt_rejects_an_entirely_missing_requested_reader():
    """A requested seed with no fixation rows cannot disappear silently."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1],
            "passage_id_zero_based": [0],
            "word_id_zero_based": [0],
            "word_raw": ["one"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0],
            "seed": [10],
            "text_id": [0],
            "word_id": [0],
            "fixation_duration": [100.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="do not cover every requested virtual reader",
    ):
        aggregate_ob1_tvt(
            fixations,
            canonical,
            expected_seeds=[10, 11],
        )


def test_ob1_tvt_rejects_one_simulation_mapped_to_multiple_seeds():
    """Corrupted simulation-to-seed mappings are not silently overwritten."""
    canonical = pd.DataFrame(
        {
            "passage_id_raw": [1],
            "passage_id_zero_based": [0],
            "word_id_zero_based": [0],
            "word_raw": ["one"],
        }
    )
    fixations = pd.DataFrame(
        {
            "simulation_id": [0, 0],
            "seed": [10, 11],
            "text_id": [0, 0],
            "word_id": [0, 0],
            "fixation_duration": [100.0, 50.0],
        }
    )

    with pytest.raises(ValueError, match="simulation ID.*exactly one seed"):
        aggregate_ob1_tvt(fixations, canonical)


def test_ob1_runtime_invalidates_derived_cache_when_inputs_change(tmp_path):
    """A changed stimulus cannot silently reuse an upstream OB1 lexicon."""
    subtlex = tmp_path / "SUBTLEX_UK.txt"
    subtlex.write_text("Spelling\tLogFreq(Zipf)\none\t4.0\n", encoding="utf-8")
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["One passage."],
        }
    )
    runtime_dir = tmp_path / "runtime"
    prepare_ob1_runtime(passages, runtime_dir, subtlex)
    cache_path = runtime_dir / "data/processed/lexicon.pkl"
    cache_path.write_bytes(b"cache")

    unchanged = prepare_ob1_runtime(passages, runtime_dir, subtlex)
    assert cache_path.is_file()
    assert unchanged["removed_stale_caches"] == []

    passages.loc[0, "passage_text"] = "A changed passage."
    changed = prepare_ob1_runtime(passages, runtime_dir, subtlex)
    assert not cache_path.exists()
    assert changed["removed_stale_caches"] == ["lexicon.pkl"]


def test_ob1_runtime_supports_isolated_named_corpus(tmp_path):
    """A non-Provo corpus receives its own stimulus and cache filenames."""
    subtlex = tmp_path / "SUBTLEX_UK.txt"
    subtlex.write_text("Spelling\tLogFreq(Zipf)\none\t4.0\n", encoding="utf-8")
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["One passage."],
        }
    )

    audit = prepare_ob1_runtime(
        passages,
        tmp_path / "runtime",
        subtlex,
        stimulus_name="OneStop_Corpus",
    )

    assert audit["stimulus_name"] == "OneStop_Corpus"
    assert audit["stimuli_path"].endswith("OneStop_Corpus.csv")
    assert (
        "frequency_map_OneStop_Corpus_continuous_reading_english.json"
        in derived_cache_filenames("OneStop_Corpus")
    )


def test_ob1_runtime_replaces_only_punctuation_only_tokens(tmp_path):
    """OB1 receives nonempty surrogates without changing word coordinates."""
    subtlex = tmp_path / "SUBTLEX_UK.txt"
    subtlex.write_text("Spelling\tLogFreq(Zipf)\none\t4.0\n", encoding="utf-8")
    original = "One  – lexical... \t— end"
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": [original],
        }
    )

    audit = prepare_ob1_runtime(
        passages,
        tmp_path / "runtime",
        subtlex,
        stimulus_name="OneStop_Corpus",
    )
    stimuli = pd.read_csv(audit["stimuli_path"], sep="\t")
    transformations = pd.read_csv(audit["token_transformations_path"])
    transformed = stimuli.loc[0, "all"]

    assert transformed == "One  x lexical... \tx end"
    assert len(transformed) == len(original)
    assert len(transformed.split()) == len(original.split())
    assert transformations[
        [
            "passage_id_zero_based",
            "word_id_zero_based",
            "word_raw",
            "ob1_word",
            "reason",
        ]
    ].to_dict("records") == [
        {
            "passage_id_zero_based": 0,
            "word_id_zero_based": 1,
            "word_raw": "–",
            "ob1_word": "x",
            "reason": "punctuation_only_ob1_empty_token",
        },
        {
            "passage_id_zero_based": 0,
            "word_id_zero_based": 3,
            "word_raw": "—",
            "ob1_word": "x",
            "reason": "punctuation_only_ob1_empty_token",
        },
    ]
    assert audit["token_transformations"] == 2
    assert audit["input_manifest"]["token_transformations"] == 2
    assert audit["input_manifest"]["token_transformations_sha256"]


@pytest.mark.parametrize(
    "passage_ids",
    ([7], [0, 2], [0, 0], [0.5]),
)
def test_ob1_runtime_rejects_ids_that_upstream_cannot_preserve(
    passage_ids,
):
    """OB1 row-position text IDs require an exact contiguous canonical grid."""
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": passage_ids,
            "passage_text": ["word"] * len(passage_ids),
        }
    )

    with pytest.raises(ValueError, match="passage IDs must"):
        validate_ob1_passages(passages)


def test_ob1_runtime_sorts_contiguous_ids_into_upstream_row_order():
    """A shuffled canonical table is reordered without changing its IDs."""
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [1, 0],
            "passage_text": ["second", "first"],
        }
    )

    validated = validate_ob1_passages(passages)

    assert validated["passage_id_zero_based"].tolist() == [0, 1]
    assert validated["passage_text"].tolist() == ["first", "second"]


def test_ob1_runtime_uses_unambiguous_named_transformation_files(tmp_path):
    """Each named stimulus receives a deterministic transformation audit file."""
    subtlex = tmp_path / "SUBTLEX_UK.txt"
    subtlex.write_text("Spelling\tLogFreq(Zipf)\none\t4.0\n", encoding="utf-8")
    passages = pd.DataFrame(
        {
            "passage_id_zero_based": [0],
            "passage_text": ["One – passage."],
        }
    )
    runtime = tmp_path / "runtime"

    first = prepare_ob1_runtime(
        passages,
        runtime,
        subtlex,
        stimulus_name="First_Corpus",
    )
    second = prepare_ob1_runtime(
        passages,
        runtime,
        subtlex,
        stimulus_name="Second_Corpus",
    )

    first_path = Path(first["token_transformations_path"])
    second_path = Path(second["token_transformations_path"])
    assert first_path != second_path
    assert first["input_manifest"]["token_transformations_filename"] == (
        "First_Corpus_token_transformations.csv"
    )
    assert second["input_manifest"]["token_transformations_filename"] == (
        "Second_Corpus_token_transformations.csv"
    )
    assert first_path.is_file()
    assert second_path.is_file()


def test_transform_ob1_passage_keeps_lexical_tokens_unchanged():
    """Tokens with at least one OB1-recognized character remain byte-identical."""
    text = "can't  lexical... café 123"
    transformed, records = transform_ob1_passage(0, text)

    assert transformed == text
    assert records == []


def test_ob1_stimulus_name_rejects_path_components():
    """A corpus label cannot escape the isolated runtime directory."""
    with pytest.raises(ValueError, match="letters, digits"):
        validate_stimulus_name("../OneStop")


def test_seed_chunks_are_balanced_and_complete():
    """Parallel workers receive every seed exactly once."""
    chunks = split_seed_chunks(list(range(10)), workers=3)

    assert [len(chunk) for chunk in chunks] == [4, 3, 3]
    assert sorted(seed for chunk in chunks for seed in chunk) == list(range(10))


def test_parallel_worker_outputs_restore_global_simulation_order(tmp_path):
    """Chunk-local simulation IDs are remapped to the requested seed order."""
    chunks = [([11, 13], tmp_path / "worker0"), ([12], tmp_path / "worker1")]
    for chunk_seeds, chunk_dir in chunks:
        chunk_dir.mkdir()
        rows = []
        runtimes = []
        for local_id, seed in enumerate(chunk_seeds):
            rows.append(
                {
                    "simulation_id": local_id,
                    "seed": seed,
                    "text_id": 0,
                    "fixation_counter": 0,
                    "word_id": 1,
                    "word": "word",
                    "fixation_duration": 100.0,
                    "saccade_type": "forward",
                    "attentional_width": 5.0,
                    "eye_position": 1.0,
                    "saccade_distance": 1.0,
                    "saccade_error": 0.0,
                    "saccade_cause": "",
                }
            )
            runtimes.append(
                {"simulation_id": local_id, "seed": seed, "seconds": 1.0}
            )
        pd.DataFrame(rows).to_csv(chunk_dir / "ob1_fixations.csv", index=False)
        (chunk_dir / "ob1_worker_manifest.json").write_text(
            __import__("json").dumps(
                {
                    "parameters": {"attend_width": 5.0},
                    "runtimes": runtimes,
                }
            ),
            encoding="utf-8",
        )

    merge_ob1_worker_outputs(
        tmp_path,
        seeds=[11, 12, 13],
        chunks=chunks,
        workers_requested=2,
        python_hash_seed=20260725,
    )
    merged = pd.read_csv(tmp_path / "ob1_fixations.csv")

    assert merged["seed"].tolist() == [11, 12, 13]
    assert merged["simulation_id"].tolist() == [0, 1, 2]

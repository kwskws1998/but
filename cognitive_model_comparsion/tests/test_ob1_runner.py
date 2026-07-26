"""Tests for isolated OB1 runtime preparation and TVT aggregation."""

import pandas as pd
import pytest

from cognitive_model_comparsion.src.ob1_runner import (
    aggregate_ob1_tvt,
    merge_ob1_worker_outputs,
    prepare_ob1_runtime,
    split_seed_chunks,
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

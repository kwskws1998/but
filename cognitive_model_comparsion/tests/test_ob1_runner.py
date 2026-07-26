"""Tests for isolated OB1 runtime preparation and TVT aggregation."""

import pandas as pd
import pytest

from cognitive_model_comparsion.src.ob1_runner import (
    aggregate_ob1_tvt,
    prepare_ob1_runtime,
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

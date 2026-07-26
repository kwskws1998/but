"""Prepare, execute, and aggregate the pinned OB1 Provo baseline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "third_party/ob1_reader_provo_2024"
OB1_SOURCE_COMMIT = "56b8d6401d1c2c1886a9c6ff9df4a143c6f2c12d"
WORKER_PATH = ROOT / "src/ob1_worker.py"
DEFAULT_RUNTIME_DIR = ROOT / "data/ob1_runtime"
SUBTLEX_PATH = ROOT / "data/raw/SUBTLEX_UK.txt"
DERIVED_CACHE_FILENAMES = (
    "lexicon.pkl",
    "frequency_map_Provo_Corpus_continuous_reading_english.json",
    "prediction_map_Provo_Corpus__continuous_reading_english.json",
    "inhibition_matrix_previous.pkl",
    "inhibition_matrix_parameters_previous.pkl",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one runtime input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_ob1_runtime(
    passages: pd.DataFrame,
    runtime_dir: Path,
    subtlex_path: Path = SUBTLEX_PATH,
    python_hash_seed: int = 20260725,
) -> dict:
    """Create the exact file layout required by the unmodified OB1 code."""
    runtime_dir = runtime_dir.resolve()
    source_working_dir = runtime_dir / "src"
    raw_dir = runtime_dir / "data/raw"
    processed_dir = runtime_dir / "data/processed"
    source_working_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    subtlex_destination = raw_dir / "SUBTLEX_UK.txt"
    if (
        not subtlex_destination.is_file()
        or sha256_file(subtlex_destination) != sha256_file(subtlex_path)
    ):
        shutil.copyfile(subtlex_path, subtlex_destination)

    stimuli = passages.sort_values("passage_id_zero_based").copy()
    stimuli_frame = pd.DataFrame(
        {
            "id": stimuli["passage_id_zero_based"].astype(int),
            "all": stimuli["passage_text"],
            "words": [
                [item for item in str(text).split()]
                for text in stimuli["passage_text"]
            ],
            "word_ids": [
                list(range(len(str(text).split())))
                for text in stimuli["passage_text"]
            ],
        }
    )
    stimuli_path = processed_dir / "Provo_Corpus.csv"
    stimuli_frame.to_csv(stimuli_path, sep="\t", index=False)
    input_manifest_path = processed_dir / "runtime_input_manifest.json"
    input_manifest = {
        "stimuli_sha256": sha256_file(stimuli_path),
        "subtlex_sha256": sha256_file(subtlex_destination),
        "passages": len(stimuli_frame),
        "python_hash_seed": int(python_hash_seed),
        "vendor_commit": OB1_SOURCE_COMMIT,
    }
    previous_manifest = None
    if input_manifest_path.is_file():
        with input_manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
    removed_caches = []
    if previous_manifest != input_manifest:
        for filename in DERIVED_CACHE_FILENAMES:
            cache_path = processed_dir / filename
            if cache_path.is_file():
                cache_path.unlink()
                removed_caches.append(filename)
    with input_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(input_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "runtime_dir": str(runtime_dir),
        "source_working_dir": str(source_working_dir),
        "stimuli_path": str(stimuli_path),
        "subtlex_path": str(subtlex_destination),
        "passages": len(stimuli_frame),
        "vendor_source": str(VENDOR_ROOT.resolve()),
        "vendor_commit": OB1_SOURCE_COMMIT,
        "input_manifest": input_manifest,
        "removed_stale_caches": removed_caches,
    }


def run_ob1_subprocess(
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int = 55,
    python_hash_seed: int = 20260725,
) -> None:
    """Run the baseline worker in a fixed-hash Python subprocess."""
    if not seeds:
        raise ValueError("At least one OB1 virtual-reader seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("OB1 virtual-reader seeds must be unique")
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--vendor-src",
        str(VENDOR_ROOT / "src"),
        "--runtime-dir",
        str(runtime_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--seeds",
        ",".join(map(str, seeds)),
        "--n-trials",
        str(n_trials),
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(python_hash_seed)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    subprocess.run(command, check=True, env=environment)


def aggregate_ob1_tvt(
    fixations: pd.DataFrame,
    canonical_words: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sum all fixations per word, fill skips with zero, and average readers."""
    required = {
        "simulation_id",
        "seed",
        "text_id",
        "word_id",
        "fixation_duration",
    }
    missing = required.difference(fixations.columns)
    if missing:
        raise ValueError(f"OB1 fixation table is missing columns: {sorted(missing)}")
    if (fixations["fixation_duration"] < 0).any():
        raise ValueError("OB1 fixation durations must be nonnegative")

    simulation_ids = sorted(fixations["simulation_id"].unique().tolist())
    seed_map = (
        fixations[["simulation_id", "seed"]]
        .drop_duplicates()
        .set_index("simulation_id")["seed"]
        .to_dict()
    )
    if len(seed_map) != len(simulation_ids):
        raise ValueError("Every simulation ID must map to exactly one seed")

    grids = []
    canonical = canonical_words[
        [
            "passage_id_raw",
            "passage_id_zero_based",
            "word_id_zero_based",
            "word_raw",
        ]
    ]
    for simulation_id in simulation_ids:
        grid = canonical.copy()
        grid["simulation_id"] = simulation_id
        grid["seed"] = seed_map[simulation_id]
        grids.append(grid)
    complete = pd.concat(grids, ignore_index=True)
    summed = (
        fixations.groupby(
            ["simulation_id", "text_id", "word_id"],
            sort=True,
        )["fixation_duration"]
        .sum()
        .reset_index(name="ob1_tvt")
        .rename(
            columns={
                "text_id": "passage_id_zero_based",
                "word_id": "word_id_zero_based",
            }
        )
    )
    per_simulation = complete.merge(
        summed,
        on=[
            "simulation_id",
            "passage_id_zero_based",
            "word_id_zero_based",
        ],
        how="left",
        validate="one_to_one",
    )
    per_simulation["ob1_tvt"] = per_simulation["ob1_tvt"].fillna(0.0)
    mean_values = (
        per_simulation.groupby(
            [
                "passage_id_raw",
                "passage_id_zero_based",
                "word_id_zero_based",
                "word_raw",
            ],
            sort=True,
        )["ob1_tvt"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "ob1_tvt",
                "std": "ob1_tvt_reader_std",
            }
        )
    )
    audit = {
        "virtual_readers": len(simulation_ids),
        "seeds": [int(seed_map[item]) for item in simulation_ids],
        "fixations": len(fixations),
        "per_simulation_word_rows": len(per_simulation),
        "mean_word_rows": len(mean_values),
        "zero_tvt_rows": int((per_simulation["ob1_tvt"] == 0).sum()),
        "regressive_fixations": (
            int((fixations["saccade_type"] == "regression").sum())
            if "saccade_type" in fixations
            else None
        ),
    }
    expected_rows = len(canonical_words) * len(simulation_ids)
    if len(per_simulation) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} OB1 word rows, "
            f"found {len(per_simulation)}"
        )
    if len(mean_values) != len(canonical_words):
        raise ValueError("OB1 mean grid does not match canonical Provo grid")
    return per_simulation, mean_values, audit


def write_ob1_aggregation(
    output_dir: Path,
    per_simulation: pd.DataFrame,
    mean_values: pd.DataFrame,
    audit: dict,
) -> None:
    """Write word-level OB1 values and aggregation audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    per_simulation.to_csv(
        output_dir / "ob1_word_values_by_simulation.csv",
        index=False,
    )
    mean_values.to_csv(output_dir / "ob1_word_values.csv", index=False)
    with (output_dir / "ob1_aggregation_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

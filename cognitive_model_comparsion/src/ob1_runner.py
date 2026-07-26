"""Prepare, execute, and aggregate the pinned OB1 Provo baseline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
    workers: int = 1,
) -> None:
    """Run fixed-hash OB1 workers and merge their deterministic outputs."""
    if not seeds:
        raise ValueError("At least one OB1 virtual-reader seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("OB1 virtual-reader seeds must be unique")
    if workers < 1:
        raise ValueError("OB1 worker count must be positive")
    workers = min(workers, len(seeds))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if workers == 1:
        run_ob1_worker(
            runtime_dir,
            output_dir,
            seeds,
            n_trials,
            python_hash_seed,
        )
        return

    cache_ready = all(
        (runtime_dir / "data/processed" / filename).is_file()
        for filename in DERIVED_CACHE_FILENAMES
    )
    completed_chunks = []
    remaining_seeds = list(seeds)
    worker_root = output_dir / "_parallel_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    if not cache_ready:
        warmup_seed = remaining_seeds.pop(0)
        warmup_dir = worker_root / "warmup"
        run_ob1_worker(
            runtime_dir,
            warmup_dir,
            [warmup_seed],
            n_trials,
            python_hash_seed,
        )
        completed_chunks.append(([warmup_seed], warmup_dir))

    chunks = split_seed_chunks(remaining_seeds, workers)
    parallel_chunks = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_dir = worker_root / f"worker_{chunk_index:03d}"
        parallel_chunks.append((chunk, chunk_dir))

    with ThreadPoolExecutor(max_workers=len(parallel_chunks) or 1) as executor:
        futures = [
            executor.submit(
                run_ob1_worker,
                runtime_dir,
                chunk_dir,
                chunk,
                n_trials,
                python_hash_seed,
            )
            for chunk, chunk_dir in parallel_chunks
        ]
        for future in futures:
            future.result()

    completed_chunks.extend(parallel_chunks)
    merge_ob1_worker_outputs(
        output_dir,
        seeds,
        completed_chunks,
        workers_requested=workers,
        python_hash_seed=python_hash_seed,
    )


def split_seed_chunks(seeds: list[int], workers: int) -> list[list[int]]:
    """Distribute seeds evenly across a bounded number of worker processes."""
    if workers < 1:
        raise ValueError("OB1 worker count must be positive")
    worker_count = min(workers, len(seeds))
    if worker_count == 0:
        return []
    chunks = [[] for _ in range(worker_count)]
    for index, seed in enumerate(seeds):
        chunks[index % worker_count].append(seed)
    return [chunk for chunk in chunks if chunk]


def ob1_worker_environment(python_hash_seed: int) -> dict[str, str]:
    """Create a deterministic single-thread environment for one OB1 process."""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(python_hash_seed)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    return environment


def run_ob1_worker(
    runtime_dir: Path,
    output_dir: Path,
    seeds: list[int],
    n_trials: int,
    python_hash_seed: int,
) -> None:
    """Run one isolated OB1 subprocess for an explicit seed chunk."""
    if not seeds:
        raise ValueError("An OB1 worker cannot receive an empty seed chunk")
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
    subprocess.run(
        command,
        check=True,
        env=ob1_worker_environment(python_hash_seed),
    )


def merge_ob1_worker_outputs(
    output_dir: Path,
    seeds: list[int],
    chunks: list[tuple[list[int], Path]],
    workers_requested: int,
    python_hash_seed: int,
) -> None:
    """Merge worker CSVs and manifests into the serial output contract."""
    seed_to_simulation = {seed: index for index, seed in enumerate(seeds)}
    fixation_frames = []
    runtimes = []
    parameters = None
    chunk_records = []

    for chunk_seeds, chunk_dir in chunks:
        fixation_path = chunk_dir / "ob1_fixations.csv"
        manifest_path = chunk_dir / "ob1_worker_manifest.json"
        if not fixation_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing OB1 worker output under {chunk_dir}"
            )
        fixation_frame = pd.read_csv(fixation_path)
        observed_seeds = set(fixation_frame["seed"].astype(int).unique())
        if observed_seeds != set(chunk_seeds):
            raise ValueError(
                f"OB1 worker seed mismatch in {chunk_dir}: "
                f"{sorted(observed_seeds)} versus {sorted(chunk_seeds)}"
            )
        fixation_frame["simulation_id"] = fixation_frame["seed"].map(
            seed_to_simulation
        )
        if fixation_frame["simulation_id"].isna().any():
            raise ValueError(f"Unknown OB1 seed in {chunk_dir}")
        fixation_frame["simulation_id"] = fixation_frame[
            "simulation_id"
        ].astype(int)
        fixation_frames.append(fixation_frame)

        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if parameters is None:
            parameters = manifest["parameters"]
        elif manifest["parameters"] != parameters:
            raise ValueError("Parallel OB1 workers used different parameters")
        for runtime in manifest["runtimes"]:
            runtime = dict(runtime)
            runtime["simulation_id"] = seed_to_simulation[int(runtime["seed"])]
            runtimes.append(runtime)
        chunk_records.append(
            {
                "seeds": [int(seed) for seed in chunk_seeds],
                "output_dir": str(chunk_dir.resolve()),
            }
        )

    fixations = pd.concat(fixation_frames, ignore_index=True)
    fixations = fixations.sort_values(
        ["simulation_id", "text_id", "fixation_counter"]
    ).reset_index(drop=True)
    observed_all_seeds = fixations["seed"].astype(int).drop_duplicates().tolist()
    if observed_all_seeds != seeds:
        raise ValueError(
            f"Merged OB1 seed order mismatch: {observed_all_seeds} versus {seeds}"
        )
    fixations.to_csv(output_dir / "ob1_fixations.csv", index=False)
    runtimes = sorted(runtimes, key=lambda item: item["simulation_id"])
    with (output_dir / "ob1_worker_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "condition": "baseline_no_predictability",
                "python_hash_seed": str(python_hash_seed),
                "seeds": seeds,
                "parameters": parameters,
                "runtimes": runtimes,
                "fixation_rows": len(fixations),
                "parallel": True,
                "workers_requested": workers_requested,
                "worker_chunks": chunk_records,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


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

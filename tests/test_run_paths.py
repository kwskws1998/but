from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from utils.run_paths import (
    MAX_IDENTIFIER_COMPONENT_BYTES,
    MAX_RUN_COMPONENT_BYTES,
    allocate_unique_folder_name,
    create_folder_name,
    experiment_digest,
    get_unique_folder_name,
)


def hybrid_et2_configuration():
    return {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "dataset_name": "OpenAssistant/oasst1",
        "concat": True,
        "use_softprompt": True,
        "batch_size": 1,
        "train_epochs": 2,
        "gradient_acum_steps": 8,
        "logging_steps": 50,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine_with_min_lr",
        "min_lr_ratio": 0.7,
        "weight_decay": 0.1,
        "seed": 42,
        "fixations_model_version": 2,
        "features_used": [1, 1, 1, 1, 1],
        "use_asym_gaussian_redistributor": True,
        "et2_gaze_concat_condition": "trt_redistributed_gaze_concat",
        "fp_dropout": [0.1, 0.3],
    }


def test_hybrid_et2_run_component_fits_linux_name_limit():
    folder_name = create_folder_name(**hybrid_et2_configuration())
    components = Path(folder_name).parts

    assert len(components[-1].encode("utf-8")) <= MAX_RUN_COMPONENT_BYTES
    assert len(components[-1].encode("utf-8")) < 255
    assert "fmv2" in components[-1]
    assert "feat11111" in components[-1]
    assert "et2-trtred" in components[-1]


def test_long_identifiers_are_bounded_and_collision_resistant():
    configuration = hybrid_et2_configuration()
    configuration["model_name"] = "organization/" + "model-" * 100
    configuration["dataset_name"] = "데이터셋/" + "split-" * 100
    first_name = create_folder_name(**configuration)

    configuration["dataset_name"] += "different"
    second_name = create_folder_name(**configuration)

    for component in Path(first_name).parts[:2]:
        assert len(component.encode("utf-8")) <= MAX_IDENTIFIER_COMPONENT_BYTES
    assert first_name != second_name


def test_run_name_is_deterministic_and_changes_with_configuration():
    configuration = hybrid_et2_configuration()
    first_name = create_folder_name(**configuration)
    second_name = create_folder_name(**configuration)

    configuration["seed"] = 43
    different_seed_name = create_folder_name(**configuration)

    assert first_name == second_name
    assert first_name != different_seed_name


def test_resolved_config_digest_covers_non_readable_settings():
    configuration = hybrid_et2_configuration()
    resolved_config = dict(configuration)
    resolved_config["max_tokens"] = None
    resolved_config["sigma_lr"] = 0.05
    first_digest = experiment_digest(resolved_config)
    first_name = create_folder_name(
        **configuration,
        resolved_config=resolved_config,
    )

    resolved_config["sigma_lr"] = 0.01
    second_digest = experiment_digest(resolved_config)
    second_name = create_folder_name(
        **configuration,
        resolved_config=resolved_config,
    )

    assert first_name != second_name
    assert first_digest != second_digest


def test_unique_path_adds_version_without_exceeding_name_limit(tmp_path):
    configuration = hybrid_et2_configuration()
    canonical, available = get_unique_folder_name(
        base_folder=tmp_path,
        **configuration,
    )
    assert canonical == available

    Path(canonical).mkdir(parents=True)
    repeated_canonical, versioned = get_unique_folder_name(
        base_folder=tmp_path,
        **configuration,
    )

    assert repeated_canonical == canonical
    assert versioned.endswith("_v1")
    assert len(Path(versioned).name.encode("utf-8")) < 255


def test_atomic_allocation_creates_distinct_directories(tmp_path):
    configuration = hybrid_et2_configuration()
    first_canonical, first_run = allocate_unique_folder_name(
        base_folder=tmp_path,
        **configuration,
    )
    second_canonical, second_run = allocate_unique_folder_name(
        base_folder=tmp_path,
        **configuration,
    )

    assert first_canonical == second_canonical
    assert first_run != second_run
    assert Path(first_run).is_dir()
    assert Path(second_run).is_dir()
    assert second_run.endswith("_v1")


def test_concurrent_allocations_are_unique(tmp_path):
    configuration = hybrid_et2_configuration()

    def allocate(_):
        return allocate_unique_folder_name(
            base_folder=tmp_path,
            **configuration,
        )[1]

    with ThreadPoolExecutor(max_workers=8) as executor:
        allocated_paths = list(executor.map(allocate, range(8)))

    assert len(set(allocated_paths)) == 8
    assert all(Path(path).is_dir() for path in allocated_paths)


def test_path_like_identifiers_cannot_escape_output_hierarchy():
    configuration = hybrid_et2_configuration()
    configuration["model_name"] = "../../outside model"
    configuration["dataset_name"] = r"..\..\outside dataset"

    folder_name = create_folder_name(**configuration)

    assert len(Path(folder_name).parts) == 4
    assert all(component not in (".", "..") for component in Path(folder_name).parts)

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Sequence


MAX_IDENTIFIER_COMPONENT_BYTES = 96
MAX_RUN_COMPONENT_BYTES = 220
RUN_HASH_LENGTH = 16
RUN_NAMING_SCHEMA_VERSION = 1


def _boolean_bit(value) -> int:
    """Normalize a boolean-like value for a compact run identifier."""
    if isinstance(value, str):
        return int(value.strip().lower() == "true")
    return int(bool(value))


def _compact_value(value) -> str:
    """Format numeric configuration values without unnecessary characters."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _bounded_component(value, max_bytes: int) -> str:
    """Create a portable path component within a byte-length limit."""
    raw_value = str(value)
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:RUN_HASH_LENGTH]
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_value).strip("-._")
    if not component:
        component = f"item-{digest}"
    if len(component.encode("utf-8")) <= max_bytes:
        return component

    suffix = f"-{digest}"
    prefix_budget = max_bytes - len(suffix.encode("utf-8"))
    prefix = component.encode("utf-8")[:prefix_budget].decode(
        "utf-8",
        errors="ignore",
    )
    prefix = prefix.rstrip("-._")
    return f"{prefix}{suffix}"


def _scheduler_label(scheduler_type: str) -> str:
    """Return a concise, readable scheduler label."""
    aliases = {
        "cosine_with_min_lr": "cosmin",
        "constant_with_warmup": "constwarm",
        "cosine_with_restarts": "cosrestart",
    }
    return aliases.get(
        scheduler_type,
        _bounded_component(scheduler_type, max_bytes=20),
    )


def _et2_condition_label(condition) -> str:
    """Return a concise ET2 condition label."""
    aliases = {
        "raw_gaze_concat": "raw",
        "trt_redistributed_gaze_concat": "trtred",
    }
    return aliases.get(
        condition,
        _bounded_component(condition, max_bytes=24),
    )


def experiment_digest(configuration) -> str:
    """Return the full digest of a normalized experiment configuration."""
    serialized_configuration = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(serialized_configuration.encode("utf-8")).hexdigest()


def create_folder_name(
    model_name,
    dataset_name,
    concat,
    use_softprompt,
    batch_size,
    train_epochs,
    gradient_acum_steps,
    logging_steps,
    learning_rate,
    lr_scheduler_type,
    min_lr_ratio,
    weight_decay,
    seed,
    fixations_model_version,
    features_used: Sequence[int],
    use_asym_gaussian_redistributor,
    et2_gaze_concat_condition,
    fp_dropout: Sequence[float],
    resolved_config=None,
) -> str:
    """Build a readable, collision-resistant experiment path."""
    if resolved_config is None:
        resolved_config = {
            "batch_size": batch_size,
            "concat": concat,
            "dataset_name": dataset_name,
            "et2_gaze_concat_condition": et2_gaze_concat_condition,
            "features_used": list(features_used),
            "fixations_model_version": fixations_model_version,
            "fp_dropout": list(fp_dropout),
            "gradient_acum_steps": gradient_acum_steps,
            "learning_rate": learning_rate,
            "logging_steps": logging_steps,
            "lr_scheduler_type": lr_scheduler_type,
            "min_lr_ratio": min_lr_ratio,
            "model_name": model_name,
            "seed": seed,
            "train_epochs": train_epochs,
            "use_asym_gaussian_redistributor": (use_asym_gaussian_redistributor),
            "use_softprompt": use_softprompt,
            "weight_decay": weight_decay,
        }
    run_hash = experiment_digest(resolved_config)[:RUN_HASH_LENGTH]

    features_label = "".join(str(value) for value in features_used)
    dropout_label = "-".join(_compact_value(value) for value in fp_dropout)
    run_parts = [
        f"sp{_boolean_bit(use_softprompt)}",
        f"bs{_compact_value(batch_size)}",
        f"ep{_compact_value(train_epochs)}",
        f"ga{_compact_value(gradient_acum_steps)}",
        f"log{_compact_value(logging_steps)}",
        f"lr{_compact_value(learning_rate)}",
        f"sch{_scheduler_label(str(lr_scheduler_type))}",
        f"mlr{_compact_value(min_lr_ratio)}",
        f"wd{_compact_value(weight_decay)}",
        f"s{_compact_value(seed)}",
        f"fmv{_compact_value(fixations_model_version)}",
        f"feat{features_label}",
        f"red{_boolean_bit(use_asym_gaussian_redistributor)}",
    ]
    if et2_gaze_concat_condition is not None:
        run_parts.append(f"et2-{_et2_condition_label(et2_gaze_concat_condition)}")
    run_parts.extend(
        [
            f"fp{dropout_label}",
            f"id{run_hash}",
        ]
    )

    model_component = _bounded_component(
        model_name,
        max_bytes=MAX_IDENTIFIER_COMPONENT_BYTES,
    )
    dataset_component = _bounded_component(
        dataset_name,
        max_bytes=MAX_IDENTIFIER_COMPONENT_BYTES,
    )
    run_component = _bounded_component(
        "_".join(run_parts),
        max_bytes=MAX_RUN_COMPONENT_BYTES,
    )
    return os.path.join(
        model_component,
        dataset_component,
        f"c{_boolean_bit(concat)}",
        run_component,
    )


def get_unique_folder_name(
    base_folder,
    model_name,
    dataset_name,
    concat,
    use_softprompt,
    batch_size,
    train_epochs,
    gradient_acum_steps,
    logging_steps,
    learning_rate,
    lr_scheduler_type,
    min_lr_ratio,
    weight_decay,
    seed,
    fixations_model_version,
    features_used,
    use_asym_gaussian_redistributor,
    et2_gaze_concat_condition,
    fp_dropout,
    resolved_config=None,
) -> tuple[str, str]:
    """Return the canonical path and the first unused versioned path."""
    folder_name = create_folder_name(
        model_name,
        dataset_name,
        concat,
        use_softprompt,
        batch_size,
        train_epochs,
        gradient_acum_steps,
        logging_steps,
        learning_rate,
        lr_scheduler_type,
        min_lr_ratio,
        weight_decay,
        seed,
        fixations_model_version,
        features_used,
        use_asym_gaussian_redistributor,
        et2_gaze_concat_condition,
        fp_dropout,
        resolved_config,
    )
    canonical_path = Path(base_folder) / folder_name
    available_path = canonical_path
    version = 1
    while available_path.exists():
        available_path = canonical_path.with_name(f"{canonical_path.name}_v{version}")
        version += 1
    return str(canonical_path), str(available_path)


def allocate_unique_folder_name(
    base_folder,
    model_name,
    dataset_name,
    concat,
    use_softprompt,
    batch_size,
    train_epochs,
    gradient_acum_steps,
    logging_steps,
    learning_rate,
    lr_scheduler_type,
    min_lr_ratio,
    weight_decay,
    seed,
    fixations_model_version,
    features_used,
    use_asym_gaussian_redistributor,
    et2_gaze_concat_condition,
    fp_dropout,
    resolved_config=None,
) -> tuple[str, str]:
    """Atomically create and return the first unused experiment directory."""
    folder_name = create_folder_name(
        model_name,
        dataset_name,
        concat,
        use_softprompt,
        batch_size,
        train_epochs,
        gradient_acum_steps,
        logging_steps,
        learning_rate,
        lr_scheduler_type,
        min_lr_ratio,
        weight_decay,
        seed,
        fixations_model_version,
        features_used,
        use_asym_gaussian_redistributor,
        et2_gaze_concat_condition,
        fp_dropout,
        resolved_config,
    )
    canonical_path = Path(base_folder) / folder_name
    version = 0
    while True:
        available_path = (
            canonical_path
            if version == 0
            else canonical_path.with_name(f"{canonical_path.name}_v{version}")
        )
        try:
            available_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            version += 1
            continue
        return str(canonical_path), str(available_path)

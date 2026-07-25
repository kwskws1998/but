from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download


ET2_REPO_ID = "skboy/et_prediction_2"
ET2_REVISION = "5785e77309d9fce8b88e908a9db100c1a0a63456"
ET2_CHECKPOINT_FILENAME = "et_predictor2_seed123.safetensors"
ET2_CHECKPOINT_SIZE = 498_621_996
ET2_CHECKPOINT_SHA256 = (
    "1a70c01f6a37e897fec8cf0d39ccba8a50ad144f076545cc4f0d8b7d67bf2b40"
)
ET2_CHECKPOINT_ENV = "GAZE_REWARD_ET2_CHECKPOINT"
ET2_CACHE_ENV = "GAZE_REWARD_HF_CACHE"
ET2_TOKENIZER_ENV = "GAZE_REWARD_ET2_TOKENIZER"


class ET2CheckpointValidationError(RuntimeError):
    """Raised when the ET2 checkpoint fails size or digest validation."""


def default_et2_cache_dir() -> Path:
    """Return the configured Hugging Face cache directory."""
    configured_path = os.environ.get(ET2_CACHE_ENV)
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "cache" / "huggingface"


def checkpoint_sha256(path: Union[str, Path]) -> str:
    """Compute the SHA-256 digest of a checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_et2_checkpoint(
    path: Union[str, Path],
    expected_size: int = ET2_CHECKPOINT_SIZE,
    expected_sha256: str = ET2_CHECKPOINT_SHA256,
) -> Path:
    """Validate the pinned ET2 safetensors checkpoint."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ET2CheckpointValidationError(
            f"ET2 checkpoint does not exist: {checkpoint_path}"
        )

    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected_size:
        raise ET2CheckpointValidationError(
            "ET2 checkpoint size mismatch: "
            f"expected {expected_size}, got {actual_size}"
        )

    actual_sha256 = checkpoint_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise ET2CheckpointValidationError(
            "ET2 checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return checkpoint_path


def ensure_et2_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    expected_size: int = ET2_CHECKPOINT_SIZE,
    expected_sha256: str = ET2_CHECKPOINT_SHA256,
) -> Path:
    """Resolve an override or download the pinned ET2 checkpoint from the Hub."""
    configured_path = checkpoint_path or os.environ.get(ET2_CHECKPOINT_ENV)
    if configured_path:
        return validate_et2_checkpoint(
            configured_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

    resolved_cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_et2_cache_dir()
    )
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=ET2_REPO_ID,
        filename=ET2_CHECKPOINT_FILENAME,
        revision=ET2_REVISION,
        cache_dir=str(resolved_cache_dir),
    )
    try:
        return validate_et2_checkpoint(
            downloaded_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    except ET2CheckpointValidationError:
        downloaded_path = hf_hub_download(
            repo_id=ET2_REPO_ID,
            filename=ET2_CHECKPOINT_FILENAME,
            revision=ET2_REVISION,
            cache_dir=str(resolved_cache_dir),
            force_download=True,
        )
        return validate_et2_checkpoint(
            downloaded_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )

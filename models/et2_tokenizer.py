from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, Union

from filelock import FileLock
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizerFast

from models.et2_checkpoint import (
    ET2_CACHE_ENV,
    ET2_REPO_ID,
    ET2_REVISION,
    ET2_TOKENIZER_ENV,
    default_et2_cache_dir,
)


ET2_TOKENIZER_FILENAME = "tokenizer.json"
ET2_TOKENIZER_SIZE = 3_558_642
ET2_TOKENIZER_SHA256 = (
    "2bb1a22cfbe25b8e5a232b7fc4d7fc5073923b45724a5f813b00811bb6620f66"
)


class ET2TokenizerValidationError(RuntimeError):
    """Raised when the pinned ET2 tokenizer JSON fails validation."""


def _sha256(path: Path) -> str:
    """Compute a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_et2_tokenizer_json(path: Union[str, Path]) -> Path:
    """Validate the tokenizer JSON published in the pinned ET2 repository."""
    tokenizer_path = Path(path).expanduser().resolve()
    if not tokenizer_path.is_file():
        raise ET2TokenizerValidationError(
            f"ET2 tokenizer JSON does not exist: {tokenizer_path}"
        )
    if tokenizer_path.stat().st_size != ET2_TOKENIZER_SIZE:
        raise ET2TokenizerValidationError(
            "ET2 tokenizer size mismatch: "
            f"expected {ET2_TOKENIZER_SIZE}, "
            f"got {tokenizer_path.stat().st_size}"
        )
    actual_sha256 = _sha256(tokenizer_path)
    if actual_sha256 != ET2_TOKENIZER_SHA256:
        raise ET2TokenizerValidationError(
            "ET2 tokenizer SHA-256 mismatch: "
            f"expected {ET2_TOKENIZER_SHA256}, got {actual_sha256}"
        )
    return tokenizer_path


def _resolve_tokenizer_source(
    tokenizer_path: Optional[Union[str, Path]],
    cache_dir: Path,
) -> Path:
    """Resolve an offline tokenizer source or download the pinned Hub JSON."""
    configured_path = tokenizer_path or os.environ.get(ET2_TOKENIZER_ENV)
    if configured_path:
        source = Path(configured_path).expanduser().resolve()
        if source.is_dir():
            source = source / ET2_TOKENIZER_FILENAME
        return validate_et2_tokenizer_json(source)

    downloaded_path = hf_hub_download(
        repo_id=ET2_REPO_ID,
        filename=ET2_TOKENIZER_FILENAME,
        revision=ET2_REVISION,
        cache_dir=str(cache_dir),
    )
    try:
        return validate_et2_tokenizer_json(downloaded_path)
    except ET2TokenizerValidationError:
        downloaded_path = hf_hub_download(
            repo_id=ET2_REPO_ID,
            filename=ET2_TOKENIZER_FILENAME,
            revision=ET2_REVISION,
            cache_dir=str(cache_dir),
            force_download=True,
        )
        return validate_et2_tokenizer_json(downloaded_path)


def _write_atomic_text(path: Path, content: str):
    """Atomically write one derived tokenizer file."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_legacy_compatible_files(
    tokenizer_json_path: Path,
    cache_dir: Path,
) -> tuple[Path, Path]:
    """Derive vocab and merges files compatible with tokenizers 0.19."""
    output_dir = (
        cache_dir
        / "et2_tokenizer_compat"
        / ET2_TOKENIZER_SHA256
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    lock_path = output_dir / ".build.lock"

    with FileLock(str(lock_path), timeout=600):
        with tokenizer_json_path.open(encoding="utf-8") as stream:
            tokenizer_data = json.load(stream)
        model_data = tokenizer_data.get("model", {})
        vocab = model_data.get("vocab")
        merges = model_data.get("merges")
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise ET2TokenizerValidationError(
                "ET2 tokenizer JSON does not contain a BPE vocab and merges"
            )

        vocab_content = json.dumps(
            vocab,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        merge_lines = []
        for merge in merges:
            if isinstance(merge, list) and len(merge) == 2:
                merge_lines.append(f"{merge[0]} {merge[1]}")
            elif isinstance(merge, str):
                merge_lines.append(merge)
            else:
                raise ET2TokenizerValidationError(
                    f"Unsupported ET2 BPE merge entry: {merge!r}"
                )
        merges_content = "#version: 0.2\n" + "\n".join(merge_lines) + "\n"

        vocab_is_current = (
            vocab_path.is_file()
            and vocab_path.read_text(encoding="utf-8") == vocab_content
        )
        merges_are_current = (
            merges_path.is_file()
            and merges_path.read_text(encoding="utf-8") == merges_content
        )
        if not vocab_is_current:
            _write_atomic_text(vocab_path, vocab_content)
        if not merges_are_current:
            _write_atomic_text(merges_path, merges_content)
    return vocab_path, merges_path


def load_et2_tokenizer(
    tokenizer_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> RobertaTokenizerFast:
    """Load the pinned ET2 tokenizer across the supported dependency versions."""
    resolved_cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else default_et2_cache_dir()
    )
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = _resolve_tokenizer_source(
        tokenizer_path,
        resolved_cache_dir,
    )
    vocab_path, merges_path = _build_legacy_compatible_files(
        tokenizer_json_path,
        resolved_cache_dir,
    )
    return RobertaTokenizerFast(
        vocab_file=str(vocab_path),
        merges_file=str(merges_path),
        add_prefix_space=True,
        bos_token="<s>",
        cls_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
        model_max_length=512,
        pad_token="<pad>",
        sep_token="</s>",
        unk_token="<unk>",
    )

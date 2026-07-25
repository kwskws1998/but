import hashlib
from io import BytesIO
from urllib.error import URLError

import pytest

import models.et_checkpoint as et_checkpoint
from models.et_checkpoint import (
    CheckpointValidationError,
    ensure_et1_checkpoint,
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_valid_cached_checkpoint_is_reused(tmp_path):
    data = b"valid checkpoint"
    target = tmp_path / "checkpoint"
    target.write_bytes(data)

    result = ensure_et1_checkpoint(
        path=target,
        url=(tmp_path / "missing").as_uri(),
        expected_size=len(data),
        expected_sha256=_sha256(data),
    )

    assert result == target
    assert target.read_bytes() == data


def test_missing_checkpoint_is_downloaded_and_validated(tmp_path):
    data = b"downloaded checkpoint"
    source = tmp_path / "source"
    target = tmp_path / "checkpoint"
    source.write_bytes(data)

    result = ensure_et1_checkpoint(
        path=target,
        url=source.as_uri(),
        expected_size=len(data),
        expected_sha256=_sha256(data),
    )

    assert result == target
    assert target.read_bytes() == data


def test_corrupt_checkpoint_is_replaced_atomically(tmp_path):
    data = b"replacement checkpoint"
    source = tmp_path / "source"
    target = tmp_path / "checkpoint"
    source.write_bytes(data)
    target.write_bytes(b"corrupt")

    ensure_et1_checkpoint(
        path=target,
        url=source.as_uri(),
        expected_size=len(data),
        expected_sha256=_sha256(data),
    )

    assert target.read_bytes() == data
    assert list(tmp_path.glob("*.part")) == []


def test_invalid_download_is_removed(tmp_path):
    data = b"invalid checkpoint"
    source = tmp_path / "source"
    target = tmp_path / "checkpoint"
    source.write_bytes(data)

    with pytest.raises(CheckpointValidationError):
        ensure_et1_checkpoint(
            path=target,
            url=source.as_uri(),
            expected_size=len(data),
            expected_sha256="0" * 64,
            max_attempts=1,
        )

    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_transient_download_failure_is_retried(tmp_path, monkeypatch):
    data = b"checkpoint after retry"
    target = tmp_path / "checkpoint"
    calls = []

    def flaky_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) < 3:
            raise URLError("temporary failure")
        return BytesIO(data)

    monkeypatch.setattr(et_checkpoint, "urlopen", flaky_urlopen)

    result = ensure_et1_checkpoint(
        path=target,
        url="https://example.test/checkpoint",
        expected_size=len(data),
        expected_sha256=_sha256(data),
        timeout_seconds=7,
        max_attempts=3,
        retry_delay_seconds=0,
    )

    assert result == target
    assert target.read_bytes() == data
    assert calls == [
        ("https://example.test/checkpoint", 7),
        ("https://example.test/checkpoint", 7),
        ("https://example.test/checkpoint", 7),
    ]
    assert list(tmp_path.glob("*.part")) == []

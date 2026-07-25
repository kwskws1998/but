import hashlib

import pytest

import models.et2_checkpoint as et2_checkpoint


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_explicit_checkpoint_override_is_validated_without_hub(
    tmp_path,
    monkeypatch,
):
    data = b"local ET2 checkpoint"
    checkpoint = tmp_path / "et2.safetensors"
    checkpoint.write_bytes(data)

    def fail_if_called(**kwargs):
        raise AssertionError("hf_hub_download must not be called")

    monkeypatch.setattr(et2_checkpoint, "hf_hub_download", fail_if_called)
    result = et2_checkpoint.ensure_et2_checkpoint(
        checkpoint_path=checkpoint,
        expected_size=len(data),
        expected_sha256=_sha256(data),
    )

    assert result == checkpoint.resolve()


def test_hub_download_is_pinned_and_validated(tmp_path, monkeypatch):
    data = b"downloaded ET2 checkpoint"
    checkpoint = tmp_path / "et2.safetensors"
    checkpoint.write_bytes(data)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setattr(et2_checkpoint, "hf_hub_download", fake_download)
    result = et2_checkpoint.ensure_et2_checkpoint(
        cache_dir=tmp_path / "cache",
        expected_size=len(data),
        expected_sha256=_sha256(data),
    )

    assert result == checkpoint.resolve()
    assert calls == [
        {
            "repo_id": et2_checkpoint.ET2_REPO_ID,
            "filename": et2_checkpoint.ET2_CHECKPOINT_FILENAME,
            "revision": et2_checkpoint.ET2_REVISION,
            "cache_dir": str((tmp_path / "cache").resolve()),
        }
    ]


def test_corrupt_hub_cache_forces_one_redownload(tmp_path, monkeypatch):
    valid_data = b"valid ET2 checkpoint"
    corrupt = tmp_path / "corrupt.safetensors"
    valid = tmp_path / "valid.safetensors"
    corrupt.write_bytes(b"corrupt")
    valid.write_bytes(valid_data)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(valid if kwargs.get("force_download") else corrupt)

    monkeypatch.setattr(et2_checkpoint, "hf_hub_download", fake_download)
    result = et2_checkpoint.ensure_et2_checkpoint(
        cache_dir=tmp_path / "cache",
        expected_size=len(valid_data),
        expected_sha256=_sha256(valid_data),
    )

    assert result == valid.resolve()
    assert len(calls) == 2
    assert "force_download" not in calls[0]
    assert calls[1]["force_download"] is True


def test_invalid_explicit_checkpoint_is_not_replaced(tmp_path, monkeypatch):
    checkpoint = tmp_path / "invalid.safetensors"
    checkpoint.write_bytes(b"invalid")

    def fail_if_called(**kwargs):
        raise AssertionError("an explicit override must not be replaced")

    monkeypatch.setattr(et2_checkpoint, "hf_hub_download", fail_if_called)
    with pytest.raises(et2_checkpoint.ET2CheckpointValidationError):
        et2_checkpoint.ensure_et2_checkpoint(
            checkpoint_path=checkpoint,
            expected_size=7,
            expected_sha256="0" * 64,
        )

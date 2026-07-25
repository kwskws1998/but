import hashlib
import json
from pathlib import Path

import models.et2_tokenizer as et2_tokenizer
from models.et2_tokenizer import _build_legacy_compatible_files


def test_tokenizer_json_is_converted_to_legacy_bpe_files(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"<s>": 0, "</s>": 1, "a": 2, "b": 3},
                    "merges": [["a", "b"], "b a"],
                }
            }
        ),
        encoding="utf-8",
    )

    vocab_path, merges_path = _build_legacy_compatible_files(
        tokenizer_json,
        tmp_path / "cache",
    )

    assert json.loads(vocab_path.read_text(encoding="utf-8")) == {
        "<s>": 0,
        "</s>": 1,
        "a": 2,
        "b": 3,
    }
    assert merges_path.read_text(encoding="utf-8") == (
        "#version: 0.2\na b\nb a\n"
    )
    assert list(vocab_path.parent.glob("*.part")) == []


def test_corrupt_derived_tokenizer_file_is_rebuilt(tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {"a": 0, "b": 1},
                    "merges": [["a", "b"]],
                }
            }
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    first_vocab, first_merges = _build_legacy_compatible_files(
        tokenizer_json,
        cache_dir,
    )
    first_vocab.write_text('{"corrupt":1}', encoding="utf-8")
    first_merges.write_text("corrupt\n", encoding="utf-8")

    second_vocab, second_merges = _build_legacy_compatible_files(
        tokenizer_json,
        cache_dir,
    )

    assert second_vocab == first_vocab
    assert second_merges == first_merges
    assert json.loads(second_vocab.read_text(encoding="utf-8")) == {
        "a": 0,
        "b": 1,
    }
    assert second_merges.read_text(encoding="utf-8") == (
        "#version: 0.2\na b\n"
    )
    assert list(second_vocab.parent.glob("*.part")) == []


def test_hub_tokenizer_download_is_pinned_and_validated(
    tmp_path,
    monkeypatch,
):
    data = b"pinned tokenizer"
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_bytes(data)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(tokenizer_json)

    monkeypatch.delenv(et2_tokenizer.ET2_TOKENIZER_ENV, raising=False)
    monkeypatch.setattr(et2_tokenizer, "ET2_TOKENIZER_SIZE", len(data))
    monkeypatch.setattr(
        et2_tokenizer,
        "ET2_TOKENIZER_SHA256",
        hashlib.sha256(data).hexdigest(),
    )
    monkeypatch.setattr(
        et2_tokenizer,
        "hf_hub_download",
        fake_download,
    )

    result = et2_tokenizer._resolve_tokenizer_source(
        tokenizer_path=None,
        cache_dir=tmp_path / "cache",
    )

    assert result == tokenizer_json.resolve()
    assert calls == [
        {
            "repo_id": et2_tokenizer.ET2_REPO_ID,
            "filename": et2_tokenizer.ET2_TOKENIZER_FILENAME,
            "revision": et2_tokenizer.ET2_REVISION,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]


def test_corrupt_hub_tokenizer_cache_forces_redownload(
    tmp_path,
    monkeypatch,
):
    valid_data = b"valid tokenizer"
    corrupt = tmp_path / "corrupt.json"
    valid = tmp_path / "valid.json"
    corrupt.write_bytes(b"corrupt")
    valid.write_bytes(valid_data)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(valid if kwargs.get("force_download") else corrupt)

    monkeypatch.delenv(et2_tokenizer.ET2_TOKENIZER_ENV, raising=False)
    monkeypatch.setattr(
        et2_tokenizer,
        "ET2_TOKENIZER_SIZE",
        len(valid_data),
    )
    monkeypatch.setattr(
        et2_tokenizer,
        "ET2_TOKENIZER_SHA256",
        hashlib.sha256(valid_data).hexdigest(),
    )
    monkeypatch.setattr(
        et2_tokenizer,
        "hf_hub_download",
        fake_download,
    )

    result = et2_tokenizer._resolve_tokenizer_source(
        tokenizer_path=None,
        cache_dir=tmp_path / "cache",
    )

    assert result == valid.resolve()
    assert len(calls) == 2
    assert "force_download" not in calls[0]
    assert calls[1]["force_download"] is True


def test_default_et2_cache_is_project_local(monkeypatch):
    monkeypatch.delenv(et2_tokenizer.ET2_CACHE_ENV, raising=False)

    expected = (
        Path(et2_tokenizer.__file__).resolve().parents[1]
        / "cache"
        / "huggingface"
    )

    assert et2_tokenizer.default_et2_cache_dir() == expected

from pathlib import Path

import pytest

import models.et1_tokenizer as et1_tokenizer


def test_pinned_tokenizer_is_downloaded_into_created_cache(
    tmp_path,
    monkeypatch,
):
    cache_dir = tmp_path / "nested" / "cache"
    calls = []
    sentinel = object()

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.delenv(et1_tokenizer.ET1_TOKENIZER_ENV, raising=False)
    monkeypatch.setattr(
        et1_tokenizer.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    result = et1_tokenizer.load_et1_tokenizer(cache_dir=cache_dir)

    assert result is sentinel
    assert cache_dir.is_dir()
    assert calls == [
        (
            (et1_tokenizer.ET1_TOKENIZER_REPO_ID,),
            {
                "revision": et1_tokenizer.ET1_TOKENIZER_REVISION,
                "cache_dir": str(cache_dir.resolve()),
                "model_max_length": (
                    et1_tokenizer.ET1_TOKENIZER_MODEL_MAX_LENGTH
                ),
            },
        )
    ]


def test_local_tokenizer_environment_override_is_offline(
    tmp_path,
    monkeypatch,
):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    calls = []
    sentinel = object()

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setenv(
        et1_tokenizer.ET1_TOKENIZER_ENV,
        str(tokenizer_dir),
    )
    monkeypatch.setattr(
        et1_tokenizer.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )

    result = et1_tokenizer.load_et1_tokenizer(
        cache_dir=tmp_path / "cache"
    )

    assert result is sentinel
    assert calls == [
        (
            (str(tokenizer_dir.resolve()),),
            {
                "local_files_only": True,
                "model_max_length": (
                    et1_tokenizer.ET1_TOKENIZER_MODEL_MAX_LENGTH
                ),
            },
        )
    ]


def test_invalid_local_tokenizer_override_fails_before_loading(
    tmp_path,
    monkeypatch,
):
    missing_path = tmp_path / "missing"
    monkeypatch.setenv(
        et1_tokenizer.ET1_TOKENIZER_ENV,
        str(missing_path),
    )

    with pytest.raises(FileNotFoundError, match="tokenizer directory"):
        et1_tokenizer.load_et1_tokenizer(cache_dir=tmp_path / "cache")


def test_default_cache_is_project_local(monkeypatch):
    monkeypatch.delenv(
        et1_tokenizer.ET1_TOKENIZER_CACHE_ENV,
        raising=False,
    )

    expected = Path(et1_tokenizer.__file__).resolve().parents[1] / "cache" / "models"

    assert et1_tokenizer.default_et1_tokenizer_cache_dir() == expected

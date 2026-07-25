import argparse

import pytest

from utils.training_limits import (
    DEFAULT_MAX_LENGTH,
    add_input_limit_arguments,
    resolve_input_limits,
)


def build_parser():
    parser = argparse.ArgumentParser()
    return add_input_limit_arguments(parser)


def test_cli_defaults_to_5000_characters_without_token_truncation():
    args = build_parser().parse_args([])

    assert args.max_length == DEFAULT_MAX_LENGTH == 5000
    assert args.max_tokens is None


def test_cli_values_override_both_limits_independently():
    args = build_parser().parse_args(
        ["--max_length", "7000", "--max_tokens", "1350"]
    )

    assert args.max_length == 7000
    assert args.max_tokens == 1350


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max_length", "0"],
        ["--max_length", "-1"],
        ["--max_tokens", "0"],
        ["--max_tokens", "-1"],
    ],
)
def test_cli_rejects_non_positive_limits(arguments):
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_programmatic_limits_accept_none_for_max_tokens():
    assert resolve_input_limits(5000, None) == (5000, None)


@pytest.mark.parametrize(
    ("max_length", "max_tokens", "exception"),
    [
        (0, None, ValueError),
        (5000, 0, ValueError),
        (5000, "1350", TypeError),
    ],
)
def test_programmatic_limits_reject_invalid_values(
    max_length,
    max_tokens,
    exception,
):
    with pytest.raises(exception):
        resolve_input_limits(max_length, max_tokens)

import argparse
from typing import Optional


DEFAULT_MAX_LENGTH = 5000
DEFAULT_MAX_TOKENS = None


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for an argparse option."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed_value


def add_input_limit_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the character-filter and optional token-truncation arguments."""
    parser.add_argument(
        "--max_length",
        type=positive_int,
        default=DEFAULT_MAX_LENGTH,
        help=(
            "maximum character length used by the dataset filter "
            f"(default: {DEFAULT_MAX_LENGTH})"
        ),
    )
    parser.add_argument(
        "--max_tokens",
        type=positive_int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "optional per-sequence tokenizer truncation limit "
            "(default: no explicit token truncation)"
        ),
    )
    return parser


def resolve_input_limits(
    max_length: int = DEFAULT_MAX_LENGTH,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
) -> tuple[int, Optional[int]]:
    """Validate input limits supplied outside argparse."""
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    if max_tokens is None:
        return max_length, None
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer or None")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer or None")
    return max_length, max_tokens

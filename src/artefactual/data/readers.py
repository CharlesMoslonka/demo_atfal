"""Utilities for reading data files."""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import orjson as json
import toolz as tlz
from beartype import beartype
from etils import epath

T = TypeVar("T", bound=dict[str, Any])
DEFAULT_GET_ID = tlz.curried.get("id")


@beartype
def read_file(path: epath.Path) -> list[dict[str, Any]]:
    """Read a file containing JSON lines.

    Args:
        path: Path to the file

    Returns:
        List of dictionaries parsed from the file
    """
    with path.open("r") as src:
        lines = src.readlines()
        samples = map(json.loads, lines)
        return list(samples)


@beartype
def join_samples(
    ratings: Sequence[T],
    responses: Sequence[T],
    key_fn: Callable[[T], str] = DEFAULT_GET_ID,
) -> list[T]:
    """Join two sequences of dictionaries on a common key.

    Args:
        ratings: First sequence of dictionaries
        responses: Second sequence of dictionaries
        key_fn: Function to extract the key from each dictionary

    Returns:
        List of joined dictionaries
    """
    joined = tlz.join(key_fn, responses, key_fn, ratings)
    return [{"id": left["id"], **left, **right} for left, right in joined]

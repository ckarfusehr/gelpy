"""The type column of the generated reference is rendered by a fixed mapping.

It must not change with the Python version (``Union`` vs ``X | None``) or the click version
(``text`` vs ``str``), so these pin the strings, not the environment.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import pytest
from click import types as ct

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.docs import annotation_name, cli_type_name  # noqa: E402


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (float, "float"),
        (Optional[float], "float | None"),  # noqa: UP045 - the typing spelling is the point
        (float | None, "float | None"),
        (Union[int, str], "int | str"),  # noqa: UP007
        (Union[str, None, int], "str | int | None"),  # noqa: UP007
        (list[str], "list[str]"),
        (Optional[list[int]], "list[int] | None"),  # noqa: UP045
        (dict[str, Any], "dict[str, Any]"),
        (tuple[int, ...], "tuple[int, ...]"),
        (Literal["a", "b"], "Literal['a', 'b']"),
        (Annotated[int, "meta"], "int"),
        (Union[Literal["all"], int, list[Union[int, str]]], "Literal['all'] | int | list[int | str]"),  # noqa: UP007
    ],
)
def test_annotation_name(annotation, expected):
    assert annotation_name(annotation) == expected


def test_typing_union_and_pipe_union_render_the_same():
    assert annotation_name(typing.Union[int, None]) == annotation_name(int | None)  # noqa: UP007


@pytest.mark.parametrize(
    ("click_type", "expected"),
    [
        (ct.STRING, "text"),
        (ct.INT, "integer"),
        (ct.FLOAT, "float"),
        (ct.BOOL, "boolean"),
        (ct.UUID, "uuid"),
        (ct.Path(), "path"),
        (ct.File(), "file"),
        (ct.Choice(["a", "b"]), "choice"),
        (ct.IntRange(0, 5), "integer"),
        (ct.FloatRange(0, 1), "float"),
        (ct.Tuple([ct.INT, ct.INT]), "<integer integer>"),
        (ct.Tuple([ct.FLOAT, ct.FLOAT, ct.INT]), "<float float integer>"),
    ],
)
def test_cli_type_name(click_type, expected):
    assert cli_type_name(click_type) == expected

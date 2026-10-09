"""CLI I/O helpers: output formatting, argv lane-term grammar, error handling."""

from __future__ import annotations

import functools
import json
from typing import Any

import typer
from pydantic import BaseModel, ValidationError

from ..core.errors import GelpyError, to_error_response
from ..schemas.lanes import LaneSelector, LaneSelectorTerm


def parse_lane_term(token: str) -> LaneSelectorTerm:
    """Translate one argv token into a selector term.

    - bare digits            -> integer index
    - ``re:<pattern>``       -> regex (kept as a prefixed string)
    - ``label:<text>``       -> forced-literal label (kept as a prefixed string)
    - anything else          -> literal label
    """
    if token.lstrip("-").isdigit():
        return int(token)
    return token


def parse_lane_selector(tokens: list[str] | None) -> LaneSelector:
    """Turn repeated ``--lane`` tokens into a selector ("all" if none given).

    ``--lane all`` (the schema's own documented literal, case-insensitive) is
    special-cased when it's the sole token, so passing it explicitly behaves the
    same as omitting ``--lane`` entirely, rather than being treated as a literal
    label that matches nothing.
    """
    if not tokens:
        return "all"
    if len(tokens) == 1 and tokens[0].lower() == "all":
        return "all"
    return [parse_lane_term(t) for t in tokens]


def run_relative(run_dir: str, path) -> str:
    """A produced file's path, relative to its run folder.

    Every entry in a result's ``files`` map must use the same base, or a caller cannot join
    them. ``runfolder.commands`` relativizes to the run root, so the CLI-side render paths
    (which arrive absolute or CWD-relative) have to be normalized the same way -- otherwise
    one JSON object carries e.g. "results.csv" beside "r/report.md" and only one of them can
    be resolved against the run directory.
    """
    from pathlib import Path

    root = Path(run_dir).resolve()
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def emit(value: Any, fmt: str = "json") -> None:
    """Print a result to stdout. ``json`` dumps models; ``text`` prints a terse form."""
    if isinstance(value, BaseModel):
        if fmt == "json":
            typer.echo(value.model_dump_json(indent=2))
        else:
            typer.echo(_terse(value.model_dump(mode="json")))
    elif fmt == "json":
        typer.echo(json.dumps(value, indent=2, default=str))
    else:
        typer.echo(str(value))


def _terse(payload: Any) -> str:
    return json.dumps(payload, default=str)


def handle_errors(func):
    """Decorator: convert domain/validation errors into an ``ErrorResponse`` + exit 1.

    The raise→response mapping lives in ``core.errors.to_error_response`` so the CLI
    and a future MCP wrapper share one translation (and one ``code`` taxonomy).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (GelpyError, FileNotFoundError, ValidationError) as exc:
            resp = to_error_response(exc)
            typer.echo(resp.model_dump_json(indent=2), err=True)
            raise typer.Exit(code=1)

    return wrapper

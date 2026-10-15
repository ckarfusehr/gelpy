"""CLI I/O helpers: output formatting, argv lane-term grammar, error handling."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel

from ..core.errors import to_error_response
from ..schemas.common import ErrorResponse
from ..schemas.lanes import LaneSelector, LaneSelectorTerm


def _click_class(name: str) -> type[Exception]:
    """One of click's exception classes, found through typer's own exports.

    Recent typer versions vendor click (``typer._click``), older ones use click itself;
    walking ``typer.BadParameter``'s MRO finds the class in either case.
    """
    return next(cls for cls in typer.BadParameter.__mro__ if cls.__name__ == name)


#: A malformed command line: an unknown option or command, a missing argument.
USAGE_ERROR = _click_class("UsageError")
#: click's control flow (exits, aborts, its own errors): never mapped, always passed on.
CLICK_CONTROL: tuple[type[BaseException], ...] = (
    typer.Exit,
    typer.Abort,
    _click_class("ClickException"),
)


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
    root = Path(run_dir).resolve()
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def run_path(run_dir: str, path) -> str:
    """A produced file's path: relative to its run folder when it is inside it, else absolute.

    A path outside the run, relative to the working directory of the call, would mean nothing to a
    caller that changes directory between calls: it is made absolute, whatever spelling it was given.
    """
    root = Path(run_dir).resolve()
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return os.path.abspath(path)


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


def emit_error(response: ErrorResponse) -> None:
    """Print one ``ErrorResponse`` to stderr: the only output of a failing command."""
    typer.echo(response.model_dump_json(indent=2), err=True)


def handle_errors(func):
    """Decorator: answer any failure inside a command with an ``ErrorResponse`` and exit 1.

    Every exception goes through ``core.errors.to_error_response``, the one mapping every
    front-end shares, so a command never prints a traceback. Two kinds pass on untouched: click's
    own control flow, and ``BrokenPipeError``. A reader that leaves early (``gelpy ... | head``)
    is not a failure of the command: typer turns the broken pipe into a quiet exit 1, whereas
    mapping it to ``os_error`` printed an error object and left Python's own flush error and
    exit code 120 behind.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (*CLICK_CONTROL, BrokenPipeError):
            raise
        except Exception as exc:  # every failure becomes one structured error
            emit_error(to_error_response(exc))
            raise typer.Exit(code=1) from None

    return wrapper


def _command_path(ctx: Any) -> str:
    """``gelpy`` and the subcommand names leading to ``ctx``, whatever the program was called."""
    names: list[str] = []
    while ctx is not None and ctx.parent is not None:
        names.append(ctx.info_name)
        ctx = ctx.parent
    return " ".join(["gelpy", *reversed(names)])


def usage_error_response(exc: Exception) -> ErrorResponse:
    """A malformed command line as an ``ErrorResponse``: ``available`` lists the group's
    subcommands, or the command's long options.

    A group called without a subcommand is click's ``no_args_is_help`` case: typer has already
    printed the help, and the exception's own message is empty, so ``error`` says what is missing.
    """
    ctx = getattr(exc, "ctx", None)
    command = getattr(ctx, "command", None)
    subcommands = getattr(command, "commands", None)
    if subcommands:
        available: list[str] | None = sorted(subcommands)
    else:
        options = {
            opt
            for param in getattr(command, "params", [])
            for opt in (*param.opts, *getattr(param, "secondary_opts", []))
            if opt.startswith("--")
        }
        available = sorted(options) or None
    return ErrorResponse(
        error=exc.format_message() or "Missing command.",
        code="usage_error",
        detail=f"Run `{_command_path(ctx)} --help` to see its arguments.",
        available=available,
    )

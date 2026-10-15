"""An advisory lock on a run folder, so two processes never write one run at the same time.

Every command that writes a run folder holds it (the writers in ``runfolder.commands`` and the
renderers in ``cli/_render.py``); reads (``status``, ``results show``) do not, so a long fit
never blocks a status check. The lock is re-entrant within one thread: a step that calls other
locked steps (a fit that then writes its report) holds it once. Another process, or another
thread of this one, gets ``run_locked`` at once instead of waiting.

It is an OS file lock (``flock`` on POSIX, ``msvcrt.locking`` on Windows) on
``<run>/.gelpy.lock``, released by the operating system when its holder exits, even after a
crash, so a stale lock file never needs deleting.
"""

from __future__ import annotations

import errno
import functools
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from ..core.errors import RunLockedError, shown
from .layout import RunPaths, open_no_follow

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _acquire(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


#: What a non-blocking lock attempt answers when another holder has the lock.
_HELD = {errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK}

_local = threading.local()


def _held() -> set[str]:
    """Run folders this thread holds the lock of."""
    if not hasattr(_local, "held"):
        _local.held = set()
    return _local.held


@contextmanager
def run_lock(run_dir: str | Path, *, create: bool = False) -> Iterator[None]:
    """Hold ``run_dir``'s lock for the duration of the block.

    A folder that is not a run (no ``recipe.json``: it does not exist, or it is an ordinary
    directory) is not locked (``create=False``): the command inside then fails with ``not_found``,
    and nothing is created in the folder. ``init`` passes ``create=True``, because it is the
    command that makes the folder a run, and checks the run's folders itself. Raises
    ``RunLockedError`` when another process or thread holds the lock.

    Every writer holds the lock, so this is where a run whose own folders are links is refused,
    before the lock file is made or anything is written (``RunPaths.require_real_folders``). The
    lock file is opened without following a link: one there is refused too.
    """
    paths = RunPaths(run_dir)
    if create:
        paths.root.mkdir(parents=True, exist_ok=True)
    elif not paths.exists():
        yield
        return
    key = os.path.realpath(paths.root)
    held = _held()
    if key in held:  # nested: this thread already holds it
        yield
        return
    if not create:
        paths.require_real_folders()
    fd = open_no_follow(paths.lock, os.O_RDWR | os.O_CREAT, 0o644, name=paths.lock.name)
    try:
        _acquire(fd)
    except OSError as exc:
        os.close(fd)
        if exc.errno not in _HELD:  # a filesystem without lock support is not a held lock
            raise OSError(
                exc.errno,
                "cannot lock the run folder (does its file system support file locks?)",
                str(paths.lock),
            ) from exc
        raise RunLockedError(f"Another gelpy command is writing {shown(str(run_dir))}.") from None
    held.add(key)
    try:
        yield
    finally:
        held.discard(key)
        try:
            _release(fd)
        finally:
            os.close(fd)


F = TypeVar("F", bound=Callable[..., Any])

#: Set on what ``locked`` and ``locked_run_dir`` return; ``takes_run_lock`` reads it.
_MARK = "_takes_run_lock"


def takes_run_lock(func: Callable[..., Any]) -> bool:
    """True when ``func`` was wrapped by ``locked`` or ``locked_run_dir``, also when another
    ``functools.wraps`` decorator sits on top. The tests use it to pin that every writer is locked."""
    return bool(getattr(func, _MARK, False))


def locked(func: F) -> F:
    """Run ``func(request, ...)`` holding the lock on ``request.run_dir``."""

    @functools.wraps(func)
    def wrapper(request: Any, *args: Any, **kwargs: Any) -> Any:
        with run_lock(request.run_dir):
            return func(request, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper  # type: ignore[return-value]


def locked_run_dir(func: F) -> F:
    """Run ``func(run_dir, ...)`` holding the lock on ``run_dir``."""

    @functools.wraps(func)
    def wrapper(run_dir: Any, *args: Any, **kwargs: Any) -> Any:
        with run_lock(run_dir):
            return func(run_dir, *args, **kwargs)

    setattr(wrapper, _MARK, True)
    return wrapper  # type: ignore[return-value]

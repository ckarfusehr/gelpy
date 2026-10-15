"""Atomic file writes: the one way gelpy puts a file into a run folder or a report folder.

A reader never sees half a file, a write that fails leaves the old file and no temp file, and
the write replaces what is at the path: a link there is replaced, never written through. A write
that is killed (SIGKILL, a power cut) can leave its temp file behind; ``is_temp_name`` says which
names those are, so that a folder gelpy owns can sweep them.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

#: Names a write tries before it gives up. A clash needs a 48-bit coincidence or a hostile folder.
_ATTEMPTS = 10
#: Bytes of randomness in a temp name: twelve hexadecimal characters.
_TOKEN_BYTES = 6
_TEMP_NAME = re.compile(r"\.(?P<final>.+)\.[0-9a-f]{12}\.tmp")


def is_temp_name(name: str) -> bool:
    """Whether ``name`` is the name of a temp file of ``atomic_write``: ``.<final name>.<random>.tmp``.

    ``<random>`` is twelve lower-case hexadecimal characters, so that nothing else is taken for one.
    """
    return _TEMP_NAME.fullmatch(name) is not None


def atomic_write(path: str | Path, content: bytes | Callable[[BinaryIO], object]) -> None:
    """Write ``path`` by way of a temp file in its folder, then rename the temp file over it.

    ``content`` is the bytes to write, or a function that writes them to the open binary temp file
    (``np.save`` and ``Figure.savefig`` take one). The folder is made when it is missing.

    The temp file is ``.<name>.<random>.tmp``, created with ``O_CREAT | O_EXCL`` (and
    ``O_NOFOLLOW`` where the system has it) and the mode ``0o666``: the file does not exist before,
    so a link planted at a name gelpy would have used is never followed, and the kernel takes the
    process's umask off the mode, which is the mode ``open`` gives a new file. A name that is taken
    is not used: another is tried. ``os.replace`` puts the temp file over ``path`` in one step.
    Whatever fails, the temp file is removed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    for _ in range(_ATTEMPTS):
        temp = path.with_name(f".{path.name}.{secrets.token_hex(_TOKEN_BYTES)}.tmp")
        try:
            fd = os.open(temp, flags, 0o666)
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "no free name for a temp file", str(path.parent))
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            if callable(content):
                content(handle)
            else:
                handle.write(content)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise

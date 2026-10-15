"""The HTML report folder: what marks it as the report's, what it holds, and whether it is current.

``gelpy report html`` leaves ``.gelpy-report.json`` (``schemas.report.ReportMarker``) in the folder
it builds. That marker says a folder is a report of gelpy's, so:

- a build writes only into a folder that is new, empty or carries a valid marker, and a rebuild
  replaces, and drops, only the files its marker lists;
- ``init --force`` removes only those files;
- ``status`` compares the marker's fingerprint with the run's.

A marker is a plain file that anyone can write, so it is never enough on its own. It may list only
names a build writes (``schemas.report.is_report_file``), and a folder is no target, whatever marker
it carries, when it is the run folder, a folder inside it other than the run's own ``report/``, a
folder that holds the run folder, or a folder that holds a ``recipe.json`` or a ``manifest.json``. A
build writes over no file its marker does not list (``check_overwrites``), and a file is the same
file by what it is, not by how its name is spelled (case, hard links).

Nothing here follows a link: a listed file that is a link is unlinked and never what it points at,
a link where the report needs a folder is left alone or refused, a link typed as a path inside the
run is refused as the folder (``claim_folder``), and a marker that is a link, is not a regular file
or lists a name a build does not write is no marker. The temp files of a write that was killed
(``utils.atomic.is_temp_name``) are the report folder's debris, and a ``.DS_Store`` is what a file
manager leaves: neither makes a folder non-empty, and the next build and ``init --force`` remove the
debris, best effort (one that cannot be removed stays and stops nothing).
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..core.errors import InvalidValueError, shown
from ..schemas.fit_config import FitConfigStore
from ..schemas.recipe import Recipe
from ..schemas.report import REPORT_INDEX, REPORT_MARKER, ReportMarker, ReportStatus
from ..schemas.run import RunManifest
from ..utils.atomic import is_temp_name
from .cache import write_json
from .hashing import content_hash
from .layout import RunPaths
from .stages import StageName, code_version, stage_key

#: What a marker may weigh: it lists a few dozen file names.
_MARKER_LIMIT = 1 << 20

NEXT_STEP = "Pass an empty or new folder, or one an earlier report html wrote."
RUN_NEXT_STEP = "Pass a new or empty folder outside the run folder, or leave out --out to use report/."


# --- reading a file that must be a plain file ------------------------------------------------
def read_regular_file(
    path: str | Path,
    *,
    what: str,
    next_step: str,
    limit: int | None = None,
    follow_links: bool = True,
) -> bytes:
    """The bytes of a regular file; anything else is an ``InvalidValueError`` that names the file.

    A pipe, a device and a folder are refused from their type, before a byte is read: opening is
    non-blocking, so a pipe nobody writes to cannot hold the caller. ``follow_links=False`` also
    refuses a link: the file is opened without following one. A file of more than ``limit`` bytes
    is refused after ``limit`` + 1 have been read, however it grows.
    ``what`` names the file in the message ("plot", "stylesheet"), ``next_step`` is its detail.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    if not follow_links:
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if os.path.islink(path):  # where the system has no O_NOFOLLOW
            raise InvalidValueError(f"The {what} {shown(str(path))} is a link.", detail=next_step)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if not follow_links and exc.errno in (errno.ELOOP, errno.EMLINK):  # a link, with O_NOFOLLOW
            raise InvalidValueError(f"The {what} {shown(str(path))} is a link.", detail=next_step) from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise InvalidValueError(
                f"The {what} {shown(str(path))} is not a regular file.", detail=next_step
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read() if limit is None else handle.read(limit + 1)
    finally:
        os.close(fd)
    if limit is not None and len(data) > limit:
        size = f"{limit >> 20} MiB" if limit % (1 << 20) == 0 else f"{limit} bytes"
        raise InvalidValueError(f"The {what} {shown(str(path))} is larger than {size}.", detail=next_step)
    return data


# --- the marker ----------------------------------------------------------------------------------
def read_marker(folder: str | Path) -> ReportMarker | None:
    """The valid marker of ``folder``, or None: there is none, or it is no regular file, too big,
    damaged, or lists a path that is not a plain relative one."""
    try:
        data = read_regular_file(
            Path(folder) / REPORT_MARKER, what="marker", next_step="", limit=_MARKER_LIMIT,
            follow_links=False,
        )
        return ReportMarker.model_validate_json(data)
    except (OSError, ValueError):  # InvalidValueError and pydantic's ValidationError are ValueErrors
        return None


def write_marker(folder: str | Path, marker: ReportMarker) -> None:
    write_json(Path(folder) / REPORT_MARKER, marker)


def _same(a: str, b: str) -> bool:
    """Whether two paths are one file or folder, however they are spelled (case, links)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _below(inner: str, outer: str) -> bool:
    """Whether ``inner`` is ``outer`` or lies below it. ``inner`` need not exist: its nearest
    folder that does is what counts."""
    here = inner
    while True:
        if _same(here, outer):
            return True
        parent = os.path.dirname(here)
        if parent == here:
            return False
        here = parent


def _is_own_report(real: str, run: str, paths: RunPaths) -> bool:
    """Whether ``real`` is the run's own ``report/``: a real folder of that name in the run."""
    own = os.path.join(run, paths.html_report.name)
    if os.path.normcase(real) == os.path.normcase(own):
        return True
    return os.path.isdir(own) and not os.path.islink(own) and _same(real, own)


def _refuse_the_run(folder: Path, paths: RunPaths) -> None:
    """Refuse the run folder, a folder inside it (but its own ``report/``) and a folder around it.

    Compared by what the folders are (``os.path.realpath`` and ``samefile``), not by how the path
    was typed: a link to the run folder is the run folder."""
    run = os.path.realpath(paths.root)
    real = os.path.realpath(folder)
    where = shown(str(folder))
    if _same(real, run):
        raise InvalidValueError(
            f"The folder {where} is the run folder: a report is not written there.", detail=RUN_NEXT_STEP
        )
    if _below(real, run) and not _is_own_report(real, run, paths):
        raise InvalidValueError(
            f"The folder {where} is inside the run folder, and only its own report/ may be: "
            "a report is not written there.",
            detail=RUN_NEXT_STEP,
        )
    if _below(run, real):
        raise InvalidValueError(
            f"The folder {where} holds the run folder: a report is not written there.",
            detail=RUN_NEXT_STEP,
        )


#: Files a file manager leaves in a folder it has shown. They are nobody's content.
_FILE_MANAGER_FILES = frozenset({".DS_Store"})


def _is_debris(entry: os.DirEntry) -> bool:
    return is_temp_name(entry.name) and entry.is_file(follow_symlinks=False)


def _is_nobodys(entry: os.DirEntry) -> bool:
    """A regular file that is no content: the temp file of a killed write, or a file manager's own."""
    return _is_debris(entry) or (entry.name in _FILE_MANAGER_FILES and entry.is_file(follow_symlinks=False))


def is_empty(folder: Path) -> bool:
    """Whether ``folder`` holds nothing a person put there: no entry, or only the temp files of a
    killed write and the files a file manager leaves (``.DS_Store``)."""
    with os.scandir(folder) as entries:
        return all(_is_nobodys(entry) for entry in entries)


def _typed_inside_the_run(folder: Path, paths: RunPaths) -> bool:
    """Whether the path as typed lies inside the run folder: the folder it is in, links followed, is
    the run folder or a folder in it. The last name is not followed: it is what is being judged."""
    parent = os.path.dirname(os.path.abspath(folder))
    return _below(os.path.realpath(parent), os.path.realpath(paths.root))


def claim_folder(folder: Path, paths: RunPaths, *, chosen: bool = True) -> ReportMarker | None:
    """Check that a report of the run ``paths`` may be written into ``folder``; the marker it holds.

    ``chosen`` says that the person named the folder (``--out``); False is the run's own
    ``report/`` by default. A link there is refused, and so is a link typed as a path inside the run
    (``--out RUN/report``): the run's own report/ must be a real folder or not exist, and ``status``
    reads it as the run's report. A folder the person passes from outside the run may be a link, and
    is checked as what it points to.

    A marker is never enough. The run folder, a folder inside it other than its own ``report/``,
    and a folder that holds it are refused, and so is any folder that holds a ``recipe.json`` or
    a ``manifest.json``. Of the rest, a folder that does not exist, is empty, or holds nothing but
    the temp files of a killed write and a ``.DS_Store`` is free (None), and one that carries a valid
    marker holds a report of gelpy's, which the build replaces (that marker). Any other folder is
    refused with ``invalid_value``: the build never writes among files it did not write.
    """
    if os.path.islink(folder) and (not chosen or _typed_inside_the_run(folder, paths)):
        raise InvalidValueError(
            f"{shown(str(folder))} is a link, and report html builds into a real folder there.",
            detail="Delete the link (what it points to is not touched) and repeat the command, "
            + (
                "or pass a folder outside the run folder with --out."
                if chosen
                else "or pass --out FOLDER."
            ),
        )
    _refuse_the_run(folder, paths)
    try:
        info = os.stat(folder)  # a link given as the folder is followed once, on purpose
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        raise InvalidValueError(f"The report folder {shown(str(folder))} is not a folder.", detail=NEXT_STEP)
    for name in (paths.recipe.name, paths.manifest.name):
        if os.path.lexists(os.path.join(folder, name)):
            raise InvalidValueError(
                f"The folder {shown(str(folder))} holds a {name}, which makes it a run folder: "
                "a report is not written there.",
                detail=RUN_NEXT_STEP,
            )
    marker = read_marker(folder)
    if marker is not None:
        return marker
    if is_empty(folder):
        return None
    raise InvalidValueError(
        f"The folder {shown(str(folder))} is not empty and no earlier report html wrote it "
        f"(it has no valid {REPORT_MARKER}).",
        detail=NEXT_STEP,
    )


def check_destinations(folder: Path, names: Iterable[str]) -> None:
    """Refuse to write below a link, or a file, where the report needs a folder."""
    seen: set[Path] = set()
    for name in names:
        here = folder
        for part in name.split("/")[:-1]:
            here = here / part
            if here in seen:
                continue
            try:
                mode = os.lstat(here).st_mode
            except FileNotFoundError:
                break  # not there yet: it is made
            if not stat.S_ISDIR(mode):
                raise InvalidValueError(
                    f"{shown(str(here))} is a link or a file, and the report needs a folder there.",
                    detail="Remove it and repeat the command.",
                )
            seen.add(here)


def _identity(path: Path) -> tuple[int, int] | None:
    """What a file is, not what it is called: (device, inode) of ``path``, a link not followed.

    None when nothing is there, and where the system gives no inode number (0): then names are all
    there is to compare. Two names of one file give one identity (a hard link, or a case alias on a
    file system that does not tell names apart by case)."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino) if info.st_ino else None


def check_overwrites(folder: Path, names: Iterable[str], earlier: ReportMarker | None) -> None:
    """Refuse a build that would write over a file its marker does not list.

    A marker says which files an earlier build wrote, and those are the report's to replace. A name
    the build writes that already exists in ``folder`` and that the marker does not list is a file
    someone else put there (a marker that lists nothing, or only some of the files, is as good as
    any: a marker is a plain file anyone can write): the build is refused before it writes anything,
    with ``invalid_value`` that names the files. A name is listed when the marker lists it, or lists
    the same file under another spelling: files are compared by identity (``_identity``).
    """
    listed = set(earlier.files) if earlier is not None else set()
    known = {i for i in (_identity(folder / name) for name in listed) if i is not None}
    unlisted = []
    for name in names:
        if name in listed or not os.path.lexists(folder / name):
            continue
        if _identity(folder / name) in known:
            continue
        unlisted.append(name)
    if unlisted:
        more = f" and {len(unlisted) - 5} more" if len(unlisted) > 5 else ""
        raise InvalidValueError(
            f"The folder {shown(str(folder))} already holds {', '.join(unlisted[:5])}{more}, which its "
            f"{REPORT_MARKER} does not list: report html does not write over a file it did not write.",
            detail="Move or delete the file, or pass another folder with --out, and repeat the command.",
        )


# --- removing what a build wrote ------------------------------------------------------------
@dataclass
class Removal:
    removed: list[str] = field(default_factory=list)
    #: Listed, but a link or a folder stands where the file was written: not removed.
    left: list[str] = field(default_factory=list)


def _mode_at(path: Path) -> int | None:
    """The mode of what is at ``path``, a link not followed; None when nothing is there.

    A name the file system finds too long (ENAMETOOLONG: a marker may list 255 bytes, and some file
    systems take fewer) is nothing there either: it cannot exist."""
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ENAMETOOLONG:
            return None
        raise


def _unlink_below(folder: Path, name: str) -> str:
    """Remove the file ``name`` below ``folder``: 'removed', 'absent' or 'left'.

    Every folder on the way must be a real folder (a link is not followed), and what is removed is
    the file or the link at ``name`` and never what a link points at. A failure to remove is raised."""
    *folders, leaf = name.split("/")
    here = folder
    for part in folders:
        here = here / part
        mode = _mode_at(here)
        if mode is None:
            return "absent"
        if not stat.S_ISDIR(mode):
            return "left"
    path = here / leaf
    mode = _mode_at(path)
    if mode is None:
        return "absent"
    if stat.S_ISDIR(mode):
        return "left"
    os.unlink(path)
    return "removed"


def _prune(folder: Path, names: Iterable[str]) -> None:
    """Remove the folders below ``folder`` that the removal of ``names`` left empty, deepest first."""
    folders: set[Path] = set()
    for name in names:
        parent = Path(name).parent
        while parent != Path("."):
            folders.add(parent)
            parent = parent.parent
    for relative in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        try:
            os.rmdir(folder / relative)
        except OSError:  # it holds something that is not the report's, or it is not a folder
            pass


def remove_files(folder: Path, names: Iterable[str], *, best_effort: bool = False) -> Removal:
    """Remove the listed files, then the folders that became empty. Nothing else is touched.

    A file that cannot be removed is an error (the report would stay reachable while it is called
    gone), unless ``best_effort``: that is for the leftovers of a killed write, which are no part of
    the report, and one that an immutable flag or a read-only file keeps in place is skipped."""
    result = Removal()
    for name in names:
        try:
            outcome = _unlink_below(folder, name)
        except OSError:
            if best_effort:
                continue
            raise
        if outcome == "removed":
            result.removed.append(name)
        elif outcome == "left":
            result.left.append(name)
    _prune(folder, result.removed)
    return result


def drop_stale(folder: Path, earlier: ReportMarker | None, keep: Iterable[str]) -> list[str]:
    """After a rebuild: remove what the earlier build wrote and this one did not.

    Files are compared by identity, not by name: on a file system that does not tell names apart by
    case, an earlier marker that lists ``img/LINEAR.png`` names the file this build wrote as
    ``img/linear.png``, and removing the old name would delete the new image."""
    if earlier is None:
        return []
    kept = set(keep)
    fresh = {i for i in (_identity(folder / name) for name in kept) if i is not None}
    stale = [
        name for name in earlier.files if name not in kept and _identity(folder / name) not in fresh
    ]
    return remove_files(folder, stale).removed


# --- the leftovers of a killed write ------------------------------------------------------------------
#: Where a build writes, so where a write that is killed can leave its temp file.
_DEBRIS_PLACES = ("", "img", "img/fits")


def _is_real_folder(folder: Path, relative: str) -> bool:
    """Whether ``relative`` is a folder below ``folder`` by way of real folders only (no link)."""
    here = folder
    for part in relative.split("/"):
        here = here / part
        try:
            if not stat.S_ISDIR(os.lstat(here).st_mode):
                return False
        except OSError:
            return False
    return True


def _debris_in(folder: Path) -> list[str]:
    """The temp files a killed write left in ``folder``, in ``img/`` and in ``img/fits/``: regular
    files named as ``utils.atomic.is_temp_name`` says (relative names, sorted). A link or a folder
    of such a name is not debris, and no link is entered."""
    found: list[str] = []
    for place in _DEBRIS_PLACES:
        if place and not _is_real_folder(folder, place):
            continue
        try:
            with os.scandir(folder / place if place else folder) as entries:
                found += [f"{place}/{e.name}" if place else e.name for e in entries if _is_debris(e)]
        except OSError:
            continue
    return sorted(found)


def sweep_debris(folder: Path) -> list[str]:
    """Remove the temp files a killed write left in a report folder; their names, sorted.

    Only regular files of the exact name ``.<file>.<12 hex>.tmp`` in the folder, ``img/`` and
    ``img/fits/`` go, and the folders that became empty with them. Best effort: a leftover that
    cannot be removed stays, and does not stop the build that finished."""
    return remove_files(folder, _debris_in(folder), best_effort=True).removed


@dataclass
class Cleared:
    removed: list[str]
    #: What is still below the folder afterwards (files and links, relative, sorted).
    left: list[str]


def _entries_below(folder: Path) -> list[str]:
    found = []
    for here, folders, names in os.walk(folder):  # a link to a folder is listed, not entered
        base = Path(here)
        found += [(base / name).relative_to(folder).as_posix() for name in names]
        found += [(base / d).relative_to(folder).as_posix() for d in folders if os.path.islink(base / d)]
    return sorted(found)


def _clear_leftovers(folder: Path) -> Cleared | None:
    """A folder with no valid marker that holds nothing but the leftovers of a killed first build
    (the temp file of its marker, say) is gelpy's debris: it goes, and so does the folder. Any other
    folder is not the report's: None."""
    leftovers = _debris_in(folder)
    try:
        with os.scandir(folder) as entries:
            only_leftovers = bool(leftovers) and all(_is_nobodys(entry) for entry in entries)
    except OSError:
        return None
    if not only_leftovers:
        return None
    removed = remove_files(folder, leftovers, best_effort=True).removed
    try:
        os.rmdir(folder)
    except OSError:  # a leftover that stayed, or a .DS_Store
        pass
    return Cleared(removed=removed, left=_entries_below(folder) if folder.exists() else [])


def clear_report(folder: Path) -> Cleared | None:
    """``init --force``: remove the report in ``folder``; None when it is not the report's.

    Only the files the marker lists go, and the leftovers of killed writes, then the folders that
    became empty, then the marker, then the folder itself when nothing else is in it. A failure to
    remove a listed file is raised; a leftover that cannot be removed stays, and is among the files
    left. A folder with no valid marker is left alone, unless it holds nothing but the leftovers of
    a killed first build.
    """
    marker = read_marker(folder)
    if marker is None:
        return _clear_leftovers(folder)
    removal = remove_files(folder, marker.files)
    swept = remove_files(folder, _debris_in(folder), best_effort=True)
    removal.removed.extend(swept.removed)
    removal.left.extend(swept.left)
    os.unlink(folder / REPORT_MARKER)
    try:
        os.rmdir(folder)
    except OSError:  # there is more in it
        pass
    return Cleared(removed=removal.removed, left=_entries_below(folder) if folder.exists() else [])


# --- whether a report is current ------------------------------------------------------------------
def report_fingerprint(
    recipe: Recipe, manifest: RunManifest, current_lanes: Iterable[int], store: FitConfigStore
) -> str:
    """A short hash of what a report shows: the gelpy version, the recipe, the stored fit
    configuration and the fit of each lane in ``current_lanes`` (the lanes whose fit is current).

    The recipe is hashed as it is now, not read from the manifest: a hand edit of recipe.json,
    which no command recorded, is seen at once. The whole ``fit_config.json`` store is hashed, not
    only the lanes that have a current fit: the page lists the configuration of every lane in it.
    A relabel, a refit, a change of any other setting and a new gelpy version each give another
    fingerprint.
    """
    fits = {str(i): manifest.stages[stage_key(StageName.FIT, i)].hash for i in sorted(current_lanes)}
    return content_hash(
        {"gelpy": code_version(), "recipe": content_hash(recipe), "fit_config": content_hash(store), "fits": fits}
    )[:16]


def report_status(paths: RunPaths, fingerprint: str) -> ReportStatus | None:
    """The run's own ``report/`` as ``status`` shows it, or None when there is no report there."""
    marker = read_marker(paths.html_report)
    if marker is None:
        return None
    index = f"{paths.html_report.relative_to(paths.root).as_posix()}/{REPORT_INDEX}"
    return ReportStatus(path=index, current=bool(marker.fingerprint) and marker.fingerprint == fingerprint)

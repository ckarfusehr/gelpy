"""On-disk layout of a run folder."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..core.errors import InvalidValueError

#: The folders of a run that gelpy writes into, as names relative to the run folder, each one after
#: the folder it lies in. ``report/`` is not among them: the HTML report folder has rules of its own
#: (``runfolder.report_folder``).
WRITTEN_FOLDERS = (
    "cache",
    "cache/image",
    "cache/background",
    "cache/profiles",
    "cache/fits",
    "render",
    "render/fits",
    "log",
)

_LINK_RULE = "gelpy never writes through a link inside a run folder."


def _linked(name: str, *, folder: bool = True) -> InvalidValueError:
    """The refusal of a link where gelpy writes: ``name`` is relative to the run folder."""
    shown_name = f"{name}/" if folder else name
    return InvalidValueError(
        f"{shown_name} is a link, and {_LINK_RULE}",
        detail=f"Delete the link {shown_name} (what it points to is not touched) and repeat the "
        f"command: gelpy makes it again.",
    )


def open_no_follow(path: str | Path, flags: int, mode: int = 0o666, *, name: str) -> int:
    """``os.open`` for a file gelpy opens in place (the lock, the history log): a link there is refused.

    The file is opened with ``O_NOFOLLOW`` where the system has it, so that a link at ``path`` is
    never opened or created through; where the system has none, a link found beforehand is refused
    instead. The refusal is an ``InvalidValueError``; ``name``, relative to the run folder, is how it
    calls the file. Any other failure is the ``OSError`` it was.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and os.path.islink(path):
        raise _linked(name, folder=False)
    try:
        return os.open(path, flags | nofollow, mode)
    except OSError as exc:
        # ELOOP (EMLINK on FreeBSD) is what a link at the last name gives with O_NOFOLLOW; a loop in a
        # folder on the way gives it too, and is not a link at ``path``
        if nofollow and exc.errno in (errno.ELOOP, errno.EMLINK) and os.path.islink(path):
            raise _linked(name, folder=False) from exc
        raise


class RunPaths:
    """Every path within a run directory.

    ::

        <run_dir>/
          recipe.json            # editable intent — precious
          fit_config.json        # per-lane fit intent — precious, editable
          manifest.json          # cache ledger — disposable
          cache/
            image/<hash>.npy
            background/<hash>.npz
            profiles/<lane>__<hash>.json
            fits/<lane>__<hash>.json
          render/ ...
          results.csv / results.json / report.md
          report/                # gelpy report html: index.html, report.css, img/
          log/history.jsonl
          .gelpy.lock            # advisory lock, held while a command writes
    """

    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir)

    # --- top-level files ---
    @property
    def recipe(self) -> Path:
        return self.root / "recipe.json"

    @property
    def fit_config(self) -> Path:
        return self.root / "fit_config.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results_csv(self) -> Path:
        return self.root / "results.csv"

    @property
    def results_json(self) -> Path:
        return self.root / "results.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    # --- directories ---
    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def render(self) -> Path:
        return self.root / "render"

    @property
    def render_fits(self) -> Path:
        return self.render / "fits"

    @property
    def log(self) -> Path:
        return self.root / "log"

    @property
    def history(self) -> Path:
        return self.log / "history.jsonl"

    @property
    def lock(self) -> Path:
        """The advisory lock file every writing command holds (``runfolder/lock.py``)."""
        return self.root / ".gelpy.lock"

    # --- cache artifacts ---
    def image_cache(self, image_hash: str) -> Path:
        return self.cache / "image" / f"{image_hash}.npy"

    def background_cache(self, background_hash: str) -> Path:
        return self.cache / "background" / f"{background_hash}.npz"

    def profile_cache(self, lane_index: int, profile_hash: str) -> Path:
        return self.cache / "profiles" / f"lane_{lane_index:03d}__{profile_hash}.json"

    def fit_cache(self, lane_index: int, fit_hash: str) -> Path:
        return self.cache / "fits" / f"lane_{lane_index:03d}__{fit_hash}.json"

    def fit_render(self, lane_index: int) -> Path:
        return self.render_fits / f"lane_{lane_index:03d}.png"

    @property
    def fit_plot_index(self) -> Path:
        """What each ``render/fits/lane_NNN.png`` shows: its fit (by hash) and the lane label."""
        return self.render_fits / "index.json"

    @property
    def html_report(self) -> Path:
        """Default folder of the static HTML report (``gelpy report html``)."""
        return self.root / "report"

    def require_real_folders(self, *, replaced: Iterable[str] = ()) -> None:
        """Refuse a run folder whose own folders are not real folders, before anything is written.

        ``WRITTEN_FOLDERS`` must each be a real folder or absent (an absent one is made as a real
        folder when it is needed), and so must ``log/history.jsonl``, which is appended to in place:
        a link is refused (``invalid_value``, naming it and the next step), and so is a file where
        a folder belongs. Only the last name is looked at, never followed, so the run folder itself,
        and any folder above it, may be a link: the person passes it. ``replaced`` names folders the
        calling command removes whole anyway (``init --force`` and ``render``): they, and what lies
        below them, are not looked at.
        """
        gone = tuple(replaced)
        for name in WRITTEN_FOLDERS:
            if any(name == old or name.startswith(f"{old}/") for old in gone):
                continue
            try:
                mode = os.lstat(self.root / name).st_mode
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(mode):
                raise _linked(name)
            if not stat.S_ISDIR(mode):
                raise InvalidValueError(
                    f"{name}/ is not a folder, and gelpy keeps files in it.",
                    detail=f"Move or delete {name} and repeat the command: gelpy makes the folder.",
                )
        if os.path.islink(self.history):
            raise _linked("log/history.jsonl", folder=False)

    def ensure_dirs(self) -> None:
        """Create the full directory skeleton (idempotent). A folder that is a link is refused."""
        self.require_real_folders()
        for d in (
            self.cache / "image",
            self.cache / "background",
            self.cache / "profiles",
            self.cache / "fits",
            self.render_fits,
            self.log,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        # os.path.exists, not Path.exists: Python 3.10's raises on a name too long for the file system
        return os.path.exists(self.recipe)

    def store_path(self, target: str | Path) -> str:
        """How this run folder records ``target``.

        Relative to the run folder when the two share a directory below the filesystem root,
        so the record reads the same from any working directory and survives moving the run
        folder together with what it points to. Absolute otherwise: a relative path that climbs
        to the filesystem root would break as soon as either side moved.

        The operating system follows ``..`` from the physical directory, so a path worked out
        from the names alone is wrong when the run folder, or a directory above it, is a symlink,
        and the names of one directory can share nothing (``/tmp`` and ``/private/tmp``). The same
        holds for the target: ``analysis/../data/gel.tif`` opens ``/disk/data/gel.tif`` when
        ``analysis`` is a link to ``/disk/analysis``, whatever ``data/gel.tif`` stands for by name. So
        ``target`` is judged as typed (made absolute from the working directory, its ``..`` not
        collapsed), and a relative path is kept only if it opens that file from the run folder as it
        is now (``os.path.samefile``). The plainest one that does wins: from the names as given (``..``
        collapsed), then from the physical run folder, then from the physical directory of ``target``
        as well. A ``target`` that does not exist yet cannot be compared, and gets the plainest. The
        record separates names with ``/`` on every system, so it reads the same on any machine.
        """
        typed = os.path.join(os.getcwd(), os.fspath(target))
        collapsed = os.path.abspath(typed)
        exists = os.path.exists(typed)
        for root, candidate in self._spellings(typed, collapsed):
            relative = _relative_below_root(candidate, root)
            if relative is not None and (not exists or self._opens(relative, typed)):
                return Path(relative).as_posix()
        if exists and not _same_file(collapsed, typed):  # the names lead to another file: say where it is
            return _physical(typed)
        return collapsed

    def _spellings(self, typed: str, collapsed: str) -> Iterator[tuple[str, str]]:
        """The (run folder, target) pairs to take a relative path between, the plainest first.

        Lazy: the physical ones cost a stat per path component, and are needed only when the
        plainest path does not open the target."""
        yield os.path.abspath(self.root), collapsed
        root_real = os.path.realpath(self.root)
        yield root_real, collapsed
        yield root_real, _physical(typed)

    def _opens(self, relative: str, typed: str) -> bool:
        """True when ``relative``, followed from the run folder as it is now, is the file ``typed`` opens."""
        return _same_file(os.path.join(self.root, relative), typed)

    def resolve(self, stored: str | Path) -> Path:
        """Where a path this run folder recorded points.

        Absolute: as is. Relative: against the run folder, except a path that exists only
        relative to the working directory, which is how gelpy 1.0 recorded paths; it is honoured
        so such run folders keep opening from the directory they were made in.
        """
        path = Path(stored)
        if path.is_absolute():
            return path
        inside = self.root / path
        if inside.exists() or not path.exists():
            return inside
        return path


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:  # one of them does not exist
        return False


def _physical(typed: str) -> str:
    """``typed`` with the directory it lies in resolved the way the system follows it (links, then ``..``)."""
    return os.path.join(os.path.realpath(os.path.dirname(typed)), os.path.basename(typed))


def _relative_below_root(target: str, root: str) -> str | None:
    """``target`` relative to ``root`` when the two share a directory below the filesystem root."""
    try:
        common = os.path.commonpath([target, root])
    except ValueError:  # different drives (Windows)
        return None
    if common == os.path.dirname(common):  # only the filesystem root is shared
        return None
    return os.path.relpath(target, root)

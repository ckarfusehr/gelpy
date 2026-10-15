"""Contracts of the HTML report folder: the marker it carries and what ``status`` says about it."""

from __future__ import annotations

from pydantic import Field, field_validator

from .common import CanonicalModel

#: The file that marks a folder as an HTML report of gelpy's (``gelpy report html``).
REPORT_MARKER = ".gelpy-report.json"
#: The page of a report folder.
REPORT_INDEX = "index.html"
#: What a build writes next to its images (the marker, which lists none of them, is the one more).
REPORT_FILES = (REPORT_INDEX, "report.css", "theme.css")
#: The folders the images go in, relative to the report folder.
REPORT_IMAGE_FOLDERS = ("img", "img/fits")
#: The longest file name, in bytes of UTF-8, that a marker may list: no file system takes a longer one,
#: so a marker that lists one could never be removed from (the system answers "File name too long").
REPORT_MAX_NAME_BYTES = 255


def _is_png_name(leaf: str) -> bool:
    return (
        len(leaf) > len(".png")
        and leaf.endswith(".png")
        and not leaf.startswith(".")
        and ".." not in leaf
        and not any(c in "/\\:" for c in leaf)
        and leaf.isprintable()  # no control character, and none that reorders or hides text
        and len(leaf.encode("utf-8")) <= REPORT_MAX_NAME_BYTES  # after isprintable: no lone surrogate
    )


def is_report_file(name: str) -> bool:
    """Whether ``name`` (relative to the report folder, separated by '/') is a file a build writes.

    That is the page and the two stylesheets, and a PNG directly in ``img/`` or in ``img/fits/``
    whose name has no separator, no ``..``, no leading dot and at most 255 bytes. Nothing else: so
    that a marker, which anyone can write, can never name a file a build or ``init --force`` may
    remove.
    """
    if name in REPORT_FILES:
        return True
    folder, _, leaf = name.rpartition("/")
    return folder in REPORT_IMAGE_FOLDERS and _is_png_name(leaf)


class ReportMarker(CanonicalModel):
    """``.gelpy-report.json``: the one thing that says a folder is a report of gelpy's.

    ``report html`` writes it into the folder it builds, and only into a folder that is new, empty
    or already carries one (the run folder, the folders around it and inside it, and any folder
    that holds a recipe.json or a manifest.json, are no target whatever they carry). A rebuild
    replaces the files it lists, ``init --force`` removes them, and ``status`` compares its
    fingerprint with the run's: nothing else in the folder is ever touched. A marker is a plain
    file anyone can write, so it may list only names a build writes (``is_report_file``): a marker
    that lists anything else is no marker.
    """

    gelpy_version: str = Field(description="The gelpy version that wrote the report.")
    fingerprint: str = Field(
        description="A short hash of the recipe, the stored fit configuration, the gelpy version "
        "and the fit of every lane the page shows. Empty while a build is running, or when one "
        "stopped half way."
    )
    files: list[str] = Field(
        description="Every file the build wrote, relative to the report folder, separated by '/'. "
        "Only names a build writes are accepted: index.html, report.css, theme.css, and PNGs "
        "directly in img/ and img/fits/, each file name at most 255 bytes. The marker itself is "
        "not listed."
    )

    @field_validator("files")
    @classmethod
    def _only_what_a_build_writes(cls, files: list[str]) -> list[str]:
        for name in files:
            if not is_report_file(name):
                raise ValueError(f"{name!r} is not a file a report build writes")
        return files


class ReportStatus(CanonicalModel):
    """The HTML report in a run's own ``report/`` folder, as ``status`` reports it."""

    path: str = Field(description="Run-relative path of the report's index.html.")
    current: bool = Field(
        description="True when the report was built from the recipe, the stored fit configuration, "
        "the fits and the gelpy version the run has now. False after a relabel, a refit, any other "
        "change of the recipe or of a lane's stored fit configuration, or a new gelpy version, and "
        "for a build that stopped half way: build it again with `gelpy report html`."
    )

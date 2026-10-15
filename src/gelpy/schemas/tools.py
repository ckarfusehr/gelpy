"""Contracts of the tool layer: what `gelpy tool list` describes, and what render tools return."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from .common import CanonicalModel

#: How a tool uses a path argument: a run folder, a file it reads, a file or folder it writes.
PathRole = Literal["run", "input", "output"]


class ToolAnnotations(CanonicalModel):
    """Behaviour hints, named as in the Model Context Protocol."""

    read_only_hint: bool = Field(description="The tool changes nothing on disk.")
    destructive_hint: bool = Field(
        description="The tool may overwrite intent a person or an agent wrote (recipe.json, "
        "fit_config.json, an existing run). Rendering and caching are not destructive."
    )
    idempotent_hint: bool = Field(
        description="Repeating the call with the same arguments changes nothing further, the "
        "append-only history log aside."
    )
    open_world_hint: bool = Field(
        default=False, description="Always false: gelpy reads and writes local files only."
    )


class ToolSpec(CanonicalModel):
    """One tool: its name (public API, stable within a major version), schemas and hints."""

    name: str = Field(description="Tool name, e.g. 'lanes_detect'.")
    title: str = Field(description="Short human-readable title.")
    description: str = Field(description="What the tool does and what to check afterwards.")
    input_schema: dict[str, Any] = Field(
        description="JSON Schema of the one JSON object the tool reads; registry names are enums."
    )
    output_schema: dict[str, Any] = Field(
        description="JSON Schema of the one JSON object it writes on success."
    )
    annotations: ToolAnnotations = Field(
        description="Behaviour hints: read-only, destructive, idempotent, open-world."
    )
    paths: dict[str, PathRole] = Field(
        default_factory=dict,
        description="Arguments that are file system paths: 'run' (a run folder), 'input' (a "
        "file the tool reads), 'output' (a file or folder it writes). Only `gelpy tool list` "
        "carries them: the MCP server does not advertise them.",
    )


class ToolList(CanonicalModel):
    """What `gelpy tool list --json` prints."""

    gelpy_version: str = Field(description="The gelpy version serving these tools.")
    tools: list[ToolSpec] = Field(description="Every tool, in workflow order.")


class ImageRenderResult(CanonicalModel):
    """Where `image show` (tool `image_render`) wrote its PNG."""

    image: str = Field(description="Run-relative path of the rendered PNG.")


class HtmlReportResult(CanonicalModel):
    """Where `report html` (tool `report_html`) wrote the report."""

    index: str = Field(
        description="Path of index.html: relative to the run folder when the report is inside it, "
        "absolute otherwise."
    )
    files: list[str] = Field(
        description="The files of the report, relative to the report folder: the page, the "
        "stylesheets and the images. .gelpy-report.json, which marks the folder as the report's, is "
        "not listed."
    )

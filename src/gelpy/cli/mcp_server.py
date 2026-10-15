"""`gelpy mcp`: the tool layer served over the Model Context Protocol on stdin/stdout.

The tools, schemas and annotations are exactly ``cli.tools``'s; the ``paths`` roles of
``gelpy tool list`` are not advertised, MCP has no field for them. A call answers with the JSON the
CLI prints, as text and as structured content; a failure with the same ``ErrorResponse``,
flagged as an error result. Calls run one at a time, reads included, each in a worker thread:
the protocol keeps answering while a fit runs, and two calls never interleave their file writes
or their renders (matplotlib is not thread-safe). A call waits for its turn in the event loop,
first come first served, so it holds no thread while it waits and one that is cancelled meanwhile
never runs. One that is cancelled while it runs finishes first, because a thread cannot be
interrupted, and keeps its turn until then. Needs the optional dependency:
``pip install "gelpy[mcp]"``.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import MissingDependencyError
from ..runfolder.stages import code_version
from . import tools

INSTRUCTIONS = (
    "gelpy analyses gel electrophoresis images. Every tool works on a run folder (run_dir) on "
    "this machine. Pass absolute paths (run_dir, image_path): a relative path resolves against "
    "the directory the server was started in. Order: init (the lane count and labels come from "
    "the user, never guessed) -> lanes_detect, then open the overlay PNG it names and check that "
    "every line sits on a lane before anything else -> background_remove -> image_render, "
    "profiles_extract, fit_run -> status and results_get. A fit's status 'ok' only means it "
    "converged; its confidence and reasons are the quality signal. Every error carries a stable "
    "code, and its detail names the next step."
)


def _require_mcp_2() -> None:
    """Refuse an installed ``mcp`` of another major version than the extra declares (``>=2.2,<3``):
    the 1.x series imports fine but has a different server API, and would fail with a TypeError."""
    from importlib import metadata

    try:
        found = metadata.version("mcp")
    except metadata.PackageNotFoundError:
        return
    if found.split(".")[0] != "2":
        raise MissingDependencyError(
            f"`gelpy mcp` needs mcp 2.x, and mcp {found} is installed. "
            'Upgrade it with `pip install -U "gelpy[mcp]"`.'
        )


def build_server() -> Any:
    """The MCP server object (``mcp.server.lowlevel.Server``) over the tool registry."""
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
    except ImportError as exc:
        raise MissingDependencyError(
            f"`gelpy mcp` needs the optional dependency `mcp` ({exc}). "
            'Install it with `pip install "gelpy[mcp]"`.'
        ) from None
    _require_mcp_2()

    advertised = [
        types.Tool(
            name=spec.name,
            title=spec.title,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            annotations=types.ToolAnnotations(title=spec.title, **spec.annotations.model_dump()),
        )
        for spec in tools.tool_list().tools
    ]
    # One call at a time. The queue is in the event loop, not in worker threads: an anyio.Lock
    # serves its waiters first come first served and a waiter can be cancelled, while a call that
    # waited on a threading.Lock inside its thread could not be, and held one of anyio's 40
    # threads, which the stdio transport needs to read and write.
    gate = anyio.Lock()

    async def list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=advertised)

    async def call_tool(ctx: Any, params: Any) -> Any:
        async with gate:
            # abandon_on_cancel stays False: a call cancelled while it runs keeps the gate until
            # its thread has returned, so the next call never starts beside it.
            ok, payload = await anyio.to_thread.run_sync(
                tools.call_tool_safely, params.name, params.arguments
            )
        text = types.TextContent(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))
        if ok:
            return types.CallToolResult(content=[text], structured_content=payload)
        return types.CallToolResult(content=[text], is_error=True)

    return Server(
        "gelpy",
        version=code_version(),
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def serve() -> None:
    """Serve on stdin/stdout until the client disconnects."""
    server = build_server()
    import os
    import signal

    import anyio
    from mcp.server.stdio import stdio_server

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    # Ctrl-C stops the server at once, with the shell's code for it. The stdio transport reads
    # stdin in a worker thread that nothing can interrupt, so a normal shutdown would wait for
    # a line that never comes; a call that is running is abandoned (its files are written
    # atomically, and its lock goes with the process).
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))
    anyio.run(run)

"""`gelpy mcp` serves the tool layer over MCP: the same tools, answers and errors."""

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
anyio = pytest.importorskip("anyio")

from gelpy.cli import tools  # noqa: E402
from gelpy.cli.mcp_server import build_server  # noqa: E402
from gelpy.schemas.run import RunStatus  # noqa: E402


def _with_client(work):
    async def main():
        async with mcp.Client(build_server()) as client:
            return await work(client)

    return anyio.run(main)


def test_the_server_lists_the_tool_layer():
    listing = _with_client(lambda client: client.list_tools())
    served = {tool.name: tool for tool in listing.tools}
    assert sorted(served) == sorted(tools.TOOLS)
    for name, tool in served.items():
        published = tools.spec(tools.TOOLS[name])
        assert tool.input_schema == published.input_schema
        assert tool.output_schema == published.output_schema
        assert tool.annotations.read_only_hint is published.annotations.read_only_hint
        assert tool.annotations.destructive_hint is published.annotations.destructive_hint


def test_a_workflow_through_mcp(gel_image, run_dir):
    steps = [
        ("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4, "width": 30}),
        ("lanes_set", {"run_dir": run_dir, "linspace": [40, 250, 4]}),
        ("background_remove", {"run_dir": run_dir, "model": "plane_ransac"}),
        ("fit_run", {"run_dir": run_dir, "lanes": [0]}),
        ("results_get", {"run_dir": run_dir}),
    ]

    async def work(client):
        answers = []
        for name, arguments in steps:
            answer = await client.call_tool(name, arguments)
            assert not answer.is_error, answer.content[0].text
            answers.append(answer)
        return answers

    results = _with_client(work)[-1]
    assert [lane["lane_index"] for lane in results.structured_content["lanes"]] == [0]
    assert json.loads(results.content[0].text) == results.structured_content


def test_a_failure_is_an_error_result_carrying_the_error_response(tmp_path):
    missing = str(tmp_path / "missing")
    answer = _with_client(lambda client: client.call_tool("status", {"run_dir": missing}))
    assert answer.is_error
    error = json.loads(answer.content[0].text)
    assert error["code"] == "not_found" and error["detail"]


def test_an_unknown_tool_is_a_registry_error():
    answer = _with_client(lambda client: client.call_tool("nope", {}))
    assert answer.is_error
    error = json.loads(answer.content[0].text)
    assert error["code"] == "registry" and "status" in error["available"]


def test_the_text_keeps_non_ascii_characters_as_the_cli_prints_them(tmp_path):
    """The text content is the JSON the CLI prints: a name with a micro sign is not escaped."""
    missing = str(tmp_path / "\u00b5-run")
    answer = _with_client(lambda client: client.call_tool("status", {"run_dir": missing}))
    assert "\u00b5-run" in answer.content[0].text
    assert "\\u00b5" not in answer.content[0].text


def test_an_installed_mcp_of_another_major_version_is_refused(monkeypatch):
    from importlib import metadata

    from gelpy.core.errors import MissingDependencyError

    monkeypatch.setattr(metadata, "version", lambda name: "1.12.0")
    with pytest.raises(MissingDependencyError, match=r"mcp 2\.x.*1\.12\.0.*gelpy\[mcp\]"):
        build_server()


def test_ctrl_c_stops_the_server_with_exit_130():
    """The stdio transport reads stdin in a thread nothing interrupts: without a handler of its
    own a SIGINT left the server running."""
    process = subprocess.Popen(
        [sys.executable, "-m", "gelpy", "mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        time.sleep(3)  # imports done, serving
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 130
    finally:
        process.kill()
        process.communicate()


def test_mcp_calls_run_one_at_a_time(monkeypatch, run_dir):
    spans = []

    def slow(request):
        start = time.monotonic()
        time.sleep(0.3)
        spans.append((start, time.monotonic()))
        return RunStatus(
            run_dir=request.run_dir, source_image="x.tif", image_width=1, image_height=1
        )

    monkeypatch.setitem(
        tools.TOOLS, "status", dataclasses.replace(tools.TOOLS["status"], handler=slow)
    )

    async def work(client):
        async with anyio.create_task_group() as group:
            for _ in range(2):
                group.start_soon(client.call_tool, "status", {"run_dir": run_dir})

    _with_client(work)
    (_, first_end), (second_start, _) = sorted(spans)
    assert second_start >= first_end


def test_gelpy_mcp_serves_over_stdio(tmp_path):
    params = mcp.StdioServerParameters(
        command=sys.executable, args=["-m", "gelpy", "mcp"], env=dict(os.environ)
    )

    async def main():
        async with mcp.Client(params) as client:
            listing = await client.list_tools()
            answer = await client.call_tool("status", {"run_dir": str(tmp_path / "missing")})
            return listing, answer

    listing, answer = anyio.run(main)
    assert "fit_run" in {tool.name for tool in listing.tools}
    assert answer.is_error
    assert json.loads(answer.content[0].text)["code"] == "not_found"


# --- the queue: a call waits for its turn in the event loop, not in a worker thread -------------
def _slow_status(monkeypatch, seconds, spans=None):
    """Make the `status` tool sleep; `spans` collects (run_dir, start, end) of each call, in the order they ran."""

    def slow(request):
        start = time.monotonic()
        time.sleep(seconds)
        if spans is not None:
            spans.append((request.run_dir, start, time.monotonic()))
        return RunStatus(
            run_dir=request.run_dir, source_image="x.tif", image_width=1, image_height=1
        )

    monkeypatch.setitem(
        tools.TOOLS, "status", dataclasses.replace(tools.TOOLS["status"], handler=slow)
    )


def _files(folder):
    """Every file of a run folder with its bytes (the lock file only marks a writer)."""
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(Path(folder).rglob("*"))
        if path.is_file() and path.name != ".gelpy.lock"
    }


def test_a_call_cancelled_while_it_waits_never_runs(monkeypatch, gel_image, run_dir):
    """`init` with force replaces a run: one that the client gave up on while it waited behind
    another call must not run when its turn comes."""
    prepare = [
        ("init", {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 4, "width": 30}),
        ("lanes_set", {"run_dir": run_dir, "linspace": [40, 250, 4], "labels": ["a", "b", "c", "d"]}),
    ]
    for name, arguments in prepare:
        ok, payload = tools.call_tool_safely(name, arguments)
        assert ok, payload
    before = _files(run_dir)
    _slow_status(monkeypatch, 0.6)
    replace = {"run_dir": run_dir, "image_path": gel_image, "n_lanes": 2, "force": True}

    async def work(client):
        async with anyio.create_task_group() as group:
            group.start_soon(client.call_tool, "status", {"run_dir": run_dir})  # holds its turn
            await anyio.sleep(0.1)
            with anyio.move_on_after(0.15) as waiting:  # the client gives up on the init
                await client.call_tool("init", replace)
        return waiting.cancelled_caught

    cancelled = _with_client(work)
    assert _files(run_dir) == before, "the call that was cancelled while it waited ran"
    assert cancelled


def test_a_call_cancelled_while_it_runs_still_holds_its_turn(monkeypatch):
    """A running tool cannot be interrupted, so the next call waits until its thread has returned:
    letting it in earlier would put two writers on one run folder."""
    spans = []
    _slow_status(monkeypatch, 0.4, spans)

    async def work(client):
        async def give_up():
            with anyio.move_on_after(0.1):
                await client.call_tool("status", {"run_dir": "first"})

        async with anyio.create_task_group() as group:
            group.start_soon(give_up)
            await anyio.sleep(0.05)
            group.start_soon(client.call_tool, "status", {"run_dir": "second"})

    _with_client(work)
    ran = {run: (start, end) for run, start, end in spans}
    assert ran["second"][0] >= ran["first"][1]


def test_waiting_calls_run_in_the_order_they_arrived(monkeypatch):
    spans = []
    _slow_status(monkeypatch, 0.02, spans)

    async def work(client):
        async with anyio.create_task_group() as group:
            for number in range(10):
                group.start_soon(client.call_tool, "status", {"run_dir": f"call-{number}"})
                await anyio.sleep(0.005)

    _with_client(work)
    assert [run for run, _, _ in spans] == [f"call-{number}" for number in range(10)]


#: `gelpy mcp` with a `status` tool that sleeps. The real tools take seconds, and the test needs
#: dozens of calls in flight, so it starts the server as a process that makes one tool slow.
SLOW_SERVER = """
import dataclasses
import time

from gelpy.cli import mcp_server, tools
from gelpy.schemas.run import RunStatus


def slow(request):
    time.sleep(0.03)
    return RunStatus(run_dir=request.run_dir, source_image="x.tif", image_width=1, image_height=1)


tools.TOOLS["status"] = dataclasses.replace(tools.TOOLS["status"], handler=slow)
mcp_server.serve()
"""


def test_the_protocol_answers_while_many_calls_wait():
    """Over stdio the SDK reads and writes through anyio's 40 worker threads. A call that waited for
    its turn in a thread would hold one of them, and with 40 waiting the server could neither read
    its input nor answer until one of them finished."""
    params = mcp.StdioServerParameters(
        command=sys.executable, args=["-c", SLOW_SERVER], env=dict(os.environ)
    )

    async def main():
        async with mcp.Client(params) as client:
            await client.list_tools()
            async with anyio.create_task_group() as group:
                for number in range(80):
                    group.start_soon(client.call_tool, "status", {"run_dir": f"call-{number}"})
                await anyio.sleep(0.3)  # all of them are sent, the first is running
                start = time.monotonic()
                await client.list_tools()
                return time.monotonic() - start

    assert anyio.run(main) < 0.75

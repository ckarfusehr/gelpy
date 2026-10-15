"""Without the optional `mcp` package, `gelpy mcp` names the extra to install."""

import json
import sys

from typer.testing import CliRunner

from gelpy.cli.app import app


def test_without_the_extra_gelpy_mcp_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)  # `import mcp...` now raises ImportError
    result = CliRunner().invoke(app, ["mcp"])
    assert result.exit_code == 1
    error = json.loads(result.stderr)
    assert error["code"] == "missing_dependency"
    assert "gelpy[mcp]" in error["detail"]

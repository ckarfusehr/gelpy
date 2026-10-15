"""gelpy — gel electrophoresis image analysis, rebuilt as an agent-native tool.

The public interface is the CLI (``python -m gelpy`` / the ``gelpy`` entry point).
The importable surface is intentionally small; use ``gelpy.schemas`` for the data
contracts and ``gelpy.runfolder.commands`` for the programmatic (request -> result)
API.
"""

__version__ = "1.1.0"

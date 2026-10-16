"""Expose a source checkout to vLLM's normal multi-process plugin discovery."""

from __future__ import annotations

import atexit
import os
import sys
import tempfile
from importlib.metadata import entry_points
from pathlib import Path

_runtime_metadata = None


def ensure_source_plugin() -> None:
    """No pip/site-packages changes; workers inherit the source and metadata paths."""
    global _runtime_metadata
    source_root = str(Path(__file__).resolve().parent.parent)
    paths = [source_root]
    plugins = entry_points(group="vllm.general_plugins", name="seam")
    if any(plugin.value != "seam_vllm.plugin:register" for plugin in plugins):
        raise RuntimeError("Another plugin named 'seam' points to a different implementation")
    if not plugins:
        if _runtime_metadata is None:
            _runtime_metadata = tempfile.TemporaryDirectory(prefix="seam_vllm_source_")
            atexit.register(_runtime_metadata.cleanup)
            metadata = Path(_runtime_metadata.name) / "seam_vllm_source_runtime-0.0.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: seam-vllm-source-runtime\nVersion: 0.0.0\n", encoding="utf-8"
            )
            (metadata / "entry_points.txt").write_text(
                "[vllm.general_plugins]\nseam = seam_vllm.plugin:register\n", encoding="utf-8"
            )
    if _runtime_metadata is not None:
        paths.append(_runtime_metadata.name)
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*paths, *(path for path in inherited if path)]))
    selected = os.environ.get("VLLM_PLUGINS")
    if selected is not None:
        names = [name.strip() for name in selected.split(",") if name.strip()]
        os.environ["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*names, "seam"]))

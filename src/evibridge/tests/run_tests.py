"""Dependency-free runner for the small assert-based test suite."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent
    count = 0
    for path in sorted(directory.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                getattr(module, name)()
                count += 1
    print(f"{count} tests passed")


if __name__ == "__main__":
    main()

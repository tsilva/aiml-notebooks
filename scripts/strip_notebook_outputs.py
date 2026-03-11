#!/usr/bin/env python3
"""Strip outputs and execution counts from Jupyter notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def iter_paths(argv: list[str]) -> list[str]:
    if len(argv) >= 2 and argv[1] == "--read-null-stdin":
        return [path for path in sys.stdin.buffer.read().decode("utf-8").split("\0") if path]
    return argv[1:]


def strip_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if not changed:
        return False

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return True


def main(argv: list[str]) -> int:
    paths = iter_paths(argv)
    if not paths:
        return 0

    changed_paths: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or path.suffix != ".ipynb":
            continue
        if strip_notebook(path):
            changed_paths.append(raw_path)

    if changed_paths:
        print("Stripped notebook outputs:")
        for path in changed_paths:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

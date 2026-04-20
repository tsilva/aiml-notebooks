#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "modal_notebook_runner.py"


def default_output_path(notebook_path: pathlib.Path) -> pathlib.Path:
    return notebook_path.with_name(f"{notebook_path.stem}.modal{notebook_path.suffix}")


def relative_to_root(path: pathlib.Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local notebook remotely on Modal and save the executed notebook.",
    )
    parser.add_argument("notebook", help="Notebook path, for example notebooks/foo.ipynb")
    parser.add_argument(
        "-o",
        "--output",
        help="Output notebook path. Defaults to notebook-name.modal.ipynb beside the source.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Per-cell execution timeout passed to nbconvert. Default: 3600.",
    )
    parser.add_argument("--gpu", action="store_true", help="Run on the Modal A10G GPU function.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    notebook = (ROOT / args.notebook).resolve()
    try:
        notebook_rel = relative_to_root(notebook)
    except ValueError:
        print(f"Notebook must be inside this repo: {notebook}", file=sys.stderr)
        return 2

    if not notebook.exists():
        print(f"Notebook does not exist: {notebook}", file=sys.stderr)
        return 2
    if notebook.suffix != ".ipynb":
        print(f"Expected a .ipynb file: {notebook}", file=sys.stderr)
        return 2
    if not notebook_rel.startswith("notebooks/"):
        print(f"Expected a notebook under notebooks/: {notebook_rel}", file=sys.stderr)
        return 2

    output = (ROOT / args.output).resolve() if args.output else default_output_path(notebook)
    try:
        output_rel = relative_to_root(output)
    except ValueError:
        print(f"Output must be inside this repo: {output}", file=sys.stderr)
        return 2

    command = [
        "uv",
        "run",
        "--with",
        "modal",
        "modal",
        "run",
        str(RUNNER.relative_to(ROOT)),
        "--notebook-path",
        notebook_rel,
        "--output-path",
        output_rel,
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.gpu:
        command.append("--gpu")

    print("Running on Modal:")
    print(" ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())

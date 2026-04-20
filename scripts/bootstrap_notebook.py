"""Standalone bootstrap entry point for remote notebook kernels."""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def _safe_cwd() -> Path:
    try:
        return Path.cwd().resolve()
    except FileNotFoundError:
        for candidate in (Path("/content"), Path("/workspace"), Path.home(), Path("/tmp")):
            if candidate.exists() and os.access(candidate, os.W_OK):
                os.chdir(candidate)
                return candidate.resolve()
        raise


def _find_repo_root(start: Path) -> Path | None:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "aiml_notebooks").is_dir():
            return path
    return None


def _default_remote_repo_dir() -> Path:
    configured = os.environ.get("AIML_NOTEBOOKS_REPO")
    if configured:
        return Path(configured).expanduser()
    for candidate in (Path("/workspace"), Path("/content")):
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate / "aiml-notebooks"
    return _safe_cwd() / "aiml-notebooks"


root = _find_repo_root(_safe_cwd())
if root is None:
    root = _default_remote_repo_dir().resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                os.environ.get("AIML_NOTEBOOKS_BRANCH", "main"),
                os.environ.get("AIML_NOTEBOOKS_REPO_URL", "https://github.com/tsilva/aiml-notebooks.git"),
                str(root),
            ],
            check=True,
        )

bootstrap_module = root / "src" / "aiml_notebooks" / "bootstrap.py"
if not bootstrap_module.exists() and (root / ".git").exists():
    subprocess.run(["git", "-C", str(root), "pull", "--ff-only"], check=True)

src_dir = root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from aiml_notebooks.bootstrap import bootstrap_notebook  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "aiml_notebooks.bootstrap":
        raise

    branch = os.environ.get("AIML_NOTEBOOKS_BRANCH", "main")
    bootstrap_url = (
        f"https://raw.githubusercontent.com/tsilva/aiml-notebooks/{branch}/"
        "src/aiml_notebooks/bootstrap.py"
    )
    exec(urllib.request.urlopen(bootstrap_url).read().decode("utf-8"), globals())


bootstrap_notebook(namespace=globals())

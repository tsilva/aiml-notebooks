"""Bootstrap notebooks that run outside a checked-out repo.

This file is intentionally standalone: a remote kernel may have only a notebook,
not the local `aiml_notebooks` package. Running this script finds or clones the
repo, adds `src/` to `sys.path`, and exposes the common notebook helpers.

Typical remote-kernel one-liner after this file is on the default branch:

    import urllib.request; exec(urllib.request.urlopen("https://raw.githubusercontent.com/tsilva/aiml-notebooks/main/scripts/bootstrap_notebook.py").read().decode("utf-8"), globals())
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_URL = os.environ.get(
    "AIML_NOTEBOOKS_REPO_URL",
    "https://github.com/tsilva/aiml-notebooks.git",
)
REPO_BRANCH = os.environ.get("AIML_NOTEBOOKS_BRANCH", "main")


def _is_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "src" / "aiml_notebooks").is_dir()


def _find_repo_root(start: Path) -> Path | None:
    for path in (start, *start.parents):
        if _is_repo_root(path):
            return path
    return None


def _default_remote_repo_dir() -> Path:
    configured = os.environ.get("AIML_NOTEBOOKS_REPO")
    if configured:
        return Path(configured).expanduser()

    for candidate in (Path("/workspace"), Path("/content")):
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate / "aiml-notebooks"

    return Path.cwd() / "aiml-notebooks"


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_notebook(
    *,
    repo_dir: str | os.PathLike[str] | None = None,
    update: bool | None = None,
    install_editable: bool | None = None,
    chdir: bool | None = None,
    namespace: dict[str, Any] | None = None,
) -> Path:
    """Find or clone this repo and expose shared notebook helpers.

    Args:
        repo_dir: Explicit checkout path. Defaults to `AIML_NOTEBOOKS_REPO`,
            then `/workspace/aiml-notebooks`, `/content/aiml-notebooks`, or a
            local `aiml-notebooks` folder.
        update: If true, run `git pull --ff-only` for an existing checkout.
            Defaults to `AIML_NOTEBOOKS_UPDATE=1`.
        install_editable: If true, run `python -m pip install -e <repo>`.
            Defaults to `AIML_NOTEBOOKS_INSTALL=1`.
        chdir: If true, make the repo root the working directory. Defaults to
            true unless `AIML_NOTEBOOKS_CHDIR=0`.
        namespace: Namespace to populate with `PROJECT_ROOT`, `SRC_DIR`,
            `repo_dir`, `get_device`, `set_seed`, and `load_repo_module`.

    Returns:
        The repository root path.
    """

    start = Path.cwd().resolve()
    root = Path(repo_dir).expanduser().resolve() if repo_dir else _find_repo_root(start)

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
                    REPO_BRANCH,
                    REPO_URL,
                    str(root),
                ],
                check=True,
            )

    if not _is_repo_root(root):
        raise RuntimeError(f"{root} is not an aiml-notebooks checkout")

    if update is None:
        update = os.environ.get("AIML_NOTEBOOKS_UPDATE") == "1"
    if update:
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only"], check=True)

    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if install_editable is None:
        install_editable = os.environ.get("AIML_NOTEBOOKS_INSTALL") == "1"
    if install_editable:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root)], check=True)

    if chdir is None:
        chdir = os.environ.get("AIML_NOTEBOOKS_CHDIR") != "0"
    if chdir:
        os.chdir(root)

    def load_repo_module(module_name: str) -> ModuleType:
        return _load_module_from_path(
            f"aiml_notebooks_{module_name.replace('.', '_')}",
            root / "src" / "aiml_notebooks" / f"{module_name.replace('.', '/')}.py",
        )

    utils = load_repo_module("utils")

    exported = {
        "PROJECT_ROOT": root,
        "SRC_DIR": src_dir,
        "repo_dir": root,
        "load_repo_module": load_repo_module,
        "get_device": utils.get_device,
        "set_seed": utils.set_seed,
    }
    if namespace is not None:
        namespace.update(exported)

    print(f"aiml-notebooks root: {root}")
    print(f"aiml-notebooks src:  {src_dir}")
    print(f"working directory:    {Path.cwd()}")
    return root


bootstrap_notebook(namespace=globals())

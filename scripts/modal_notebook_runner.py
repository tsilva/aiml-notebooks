from __future__ import annotations

import pathlib
import subprocess
import sys

import modal


LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_ROOT = pathlib.Path("/workspace/aiml-notebooks")

app = modal.App("aiml-notebook-runner")
wandb_secret = modal.Secret.from_name("wandb-secret")


def default_output_path(notebook_path: str) -> str:
    path = pathlib.PurePosixPath(notebook_path)
    return str(path.with_name(f"{path.stem}.modal{path.suffix}"))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "ipykernel==7.2.0",
        "jupytext==1.19.1",
        "matplotlib==3.10.8",
        "nbconvert==7.17.1",
        "numpy==2.4.4",
        "torch==2.11.0",
        "torchvision==0.26.0",
        "unsloth",
        "wandb",
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(REMOTE_ROOT / "src"),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    .workdir(str(REMOTE_ROOT))
    .add_local_file(
        LOCAL_ROOT / "pyproject.toml",
        remote_path=str(REMOTE_ROOT / "pyproject.toml"),
    )
    .add_local_file(LOCAL_ROOT / "uv.lock", remote_path=str(REMOTE_ROOT / "uv.lock"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(LOCAL_ROOT / "notebooks", remote_path=str(REMOTE_ROOT / "notebooks"))
    .add_local_file(LOCAL_ROOT / "names.txt", remote_path=str(REMOTE_ROOT / "names.txt"))
)


def _execute_notebook(notebook_path: str, timeout_seconds: int) -> bytes:
    notebook = (REMOTE_ROOT / notebook_path).resolve()
    if notebook.suffix != ".ipynb" or not notebook.is_relative_to(REMOTE_ROOT / "notebooks"):
        raise ValueError(f"Expected a notebook path under notebooks/: {notebook_path}")

    output_dir = pathlib.Path("/tmp/aiml-notebook-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={timeout_seconds}",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output-dir",
            str(output_dir),
            str(notebook),
        ],
        cwd=REMOTE_ROOT,
        check=True,
    )

    return (output_dir / notebook.name).read_bytes()


@app.function(image=image, secrets=[wandb_secret], timeout=4 * 60 * 60, cpu=4.0, memory=8192)
def run_notebook_cpu(notebook_path: str, timeout_seconds: int = 3600) -> bytes:
    return _execute_notebook(notebook_path, timeout_seconds)


@app.function(image=image, secrets=[wandb_secret], gpu="A10G", timeout=4 * 60 * 60, cpu=4.0, memory=16384)
def run_notebook_gpu(notebook_path: str, timeout_seconds: int = 3600) -> bytes:
    return _execute_notebook(notebook_path, timeout_seconds)


@app.local_entrypoint()
def main(
    notebook_path: str,
    output_path: str = "",
    timeout_seconds: int = 3600,
    gpu: bool = False,
):
    runner = run_notebook_gpu if gpu else run_notebook_cpu
    executed_notebook = runner.remote(notebook_path, timeout_seconds)

    destination = LOCAL_ROOT / (output_path or default_output_path(notebook_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(executed_notebook)
    print(f"Wrote executed notebook to {destination}")

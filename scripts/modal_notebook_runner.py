from __future__ import annotations

import pathlib
import subprocess
import sys

import modal


LOCAL_ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE_ROOT = pathlib.Path("/workspace/aiml-notebooks")

app = modal.App("aiml-notebook-runner")
wandb_secret = modal.Secret.from_name("wandb-secret")

SUPPORTED_GPU_TYPES = (
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "RTX-PRO-6000",
    "H100",
    "H100!",
    "H200",
    "B200",
    "B200+",
)
DEFAULT_GPU_TYPE = "A10"
GPU_TYPE_ALIASES = {
    "A10G": "A10",
}


def default_output_path(notebook_path: str) -> str:
    path = pathlib.PurePosixPath(notebook_path)
    return str(path.with_name(f"{path.stem}.modal{path.suffix}"))


def normalize_gpu_type(gpu_type: str) -> str:
    normalized = gpu_type.strip().upper()
    return GPU_TYPE_ALIASES.get(normalized, normalized)


def gpu_function_name(gpu_type: str) -> str:
    safe_name = (
        gpu_type.lower()
        .replace("-", "_")
        .replace("+", "_plus")
        .replace("!", "_pinned")
    )
    return f"run_notebook_gpu_{safe_name}"


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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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


def run_notebook_gpu_base(notebook_path: str, timeout_seconds: int = 3600) -> bytes:
    return _execute_notebook(notebook_path, timeout_seconds)


def create_gpu_runner(gpu_type: str) -> modal.Function:
    return app.function(
        image=image,
        secrets=[wandb_secret],
        gpu=gpu_type,
        timeout=4 * 60 * 60,
        cpu=4.0,
        memory=16384,
        name=gpu_function_name(gpu_type),
    )(run_notebook_gpu_base)


GPU_RUNNERS = {gpu_type: create_gpu_runner(gpu_type) for gpu_type in SUPPORTED_GPU_TYPES}


@app.local_entrypoint()
def main(
    notebook_path: str,
    output_path: str = "",
    timeout_seconds: int = 3600,
    gpu: bool = False,
    gpu_type: str = DEFAULT_GPU_TYPE,
):
    selected_gpu_type = normalize_gpu_type(gpu_type)
    if gpu and selected_gpu_type not in GPU_RUNNERS:
        supported = ", ".join(SUPPORTED_GPU_TYPES)
        raise ValueError(f"Unsupported GPU type {gpu_type!r}. Supported GPU types: {supported}")

    runner = GPU_RUNNERS[selected_gpu_type] if gpu else run_notebook_cpu
    if gpu:
        print(f"Running notebook on Modal GPU: {selected_gpu_type}")
    executed_notebook = runner.remote(notebook_path, timeout_seconds)

    destination = LOCAL_ROOT / (output_path or default_output_path(notebook_path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(executed_notebook)
    print(f"Wrote executed notebook to {destination}")

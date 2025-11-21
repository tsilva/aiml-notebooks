"""
Modal runner for Jupyter notebooks with GPU support.

Usage:
    modal run modal_notebook_runner.py
    modal run modal_notebook_runner.py --notebook notebooks/karpathy-build-gpt.ipynb
"""

import modal

app = modal.App("notebook-runner")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("papermill", "ipykernel")
    # Install all dependencies from pyproject.toml
    .pip_install_from_pyproject("pyproject.toml")
)

# Persistent volume for checkpoints
volume = modal.Volume.from_name("gpt-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-40GB",  # Change to "A10G", "T4", etc. as needed
    timeout=3600 * 8,  # 8 hours max
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],  # Optional: remove if not using W&B
)
def run_notebook(notebook_content: bytes, notebook_name: str, package_tarball: bytes, parameters: dict = None):
    """Execute a Jupyter notebook with optional parameter injection."""
    import papermill as pm
    import os
    import sys
    import tarfile
    import subprocess
    from pathlib import Path

    # Extract and install package
    package_tar_path = "/tmp/package.tar.gz"
    Path(package_tar_path).write_bytes(package_tarball)

    with tarfile.open(package_tar_path, 'r:gz') as tar:
        tar.extractall('/tmp/aiml_package')

    # Install the package
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-e", "/tmp/aiml_package"
    ], check=True)

    # Setup paths
    input_path = f"/tmp/{notebook_name}"
    output_path = f"/checkpoints/{notebook_name.replace('.ipynb', '_executed.ipynb')}"

    # Write notebook to container
    Path(input_path).write_bytes(notebook_content)

    # Set checkpoint directory
    os.environ['CHECKPOINT_DIR'] = '/checkpoints'

    print(f"📓 Executing notebook: {notebook_name}")
    print(f"📊 GPU: {os.environ.get('MODAL_GPU_TYPE', 'unknown')}")
    print(f"💾 Checkpoints: /checkpoints")

    if parameters:
        print(f"⚙️  Parameters: {parameters}")

    try:
        # Execute notebook with live output streaming
        pm.execute_notebook(
            input_path,
            output_path,
            parameters=parameters or {},
            kernel_name='python3',
            progress_bar=True,  # Show cell-by-cell progress
            log_output=True,    # Stream outputs to stdout in real-time
            stdout_file=None,   # Output to console (not file)
            stderr_file=None    # Errors to console (not file)
        )

        print(f"✅ Notebook executed successfully!")
        print(f"📁 Output saved to: {output_path}")

        # Commit volume to persist checkpoints
        volume.commit()

        return {
            "status": "success",
            "output_notebook": output_path,
            "parameters": parameters
        }

    except Exception as e:
        print(f"❌ Notebook execution failed: {e}")
        # Still commit volume in case checkpoints were created
        volume.commit()
        raise

@app.local_entrypoint()
def main(notebook: str = "notebooks/karpathy-build-gpt.ipynb"):
    """
    Run a Jupyter notebook on Modal with GPU acceleration.

    Args:
        notebook: Path to notebook file

    Examples:
        modal run modal_notebook_runner.py
        modal run modal_notebook_runner.py --notebook notebooks/karpathy-build-gpt.ipynb
    """
    from pathlib import Path
    import tarfile
    import io

    # Read notebook
    notebook_path = Path(notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    notebook_content = notebook_path.read_bytes()

    # Create tarball of the package
    print("📦 Packaging aiml_notebooks...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        # Add pyproject.toml
        tar.add("pyproject.toml", arcname="pyproject.toml")
        # Add README.md (required by pyproject.toml)
        tar.add("README.md", arcname="README.md")
        # Add src directory
        tar.add("src", arcname="src")

    package_tarball = tar_buffer.getvalue()

    print(f"🚀 Launching notebook on Modal...")
    print(f"📓 Notebook: {notebook}")

    # Run remotely
    result = run_notebook.remote(
        notebook_content=notebook_content,
        notebook_name=notebook_path.name,
        package_tarball=package_tarball,
        parameters=None
    )

    print(f"\n✅ Complete!")
    print(f"   Output: {result['output_notebook']}")
    print(f"   Checkpoints saved to Modal volume 'gpt-checkpoints'")

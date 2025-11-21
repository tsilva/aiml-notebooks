"""
Modal runner for Jupyter notebooks with GPU support.

Usage:
    modal run modal_notebook_runner.py --notebook notebooks/karpathy-build-gpt.ipynb

Or with custom parameters:
    modal run modal_notebook_runner.py --notebook notebooks/karpathy-build-gpt.ipynb \
        --param batch_size 128 --param max_steps 10000
"""

import modal

app = modal.App("notebook-runner")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("papermill", "ipykernel")
    # Copy source code for aiml_notebooks imports
    .copy_local_dir("src/aiml_notebooks", "/root/aiml_notebooks")
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
def run_notebook(notebook_content: bytes, notebook_name: str, parameters: dict = None):
    """Execute a Jupyter notebook with optional parameter injection."""
    import papermill as pm
    import sys
    import os
    from pathlib import Path

    # Setup paths
    input_path = f"/tmp/{notebook_name}"
    output_path = f"/checkpoints/{notebook_name.replace('.ipynb', '_executed.ipynb')}"

    # Write notebook to container
    Path(input_path).write_bytes(notebook_content)

    # Add aiml_notebooks to path
    sys.path.insert(0, '/root')

    # Set checkpoint directory
    os.environ['CHECKPOINT_DIR'] = '/checkpoints'

    print(f"📓 Executing notebook: {notebook_name}")
    print(f"📊 GPU: {os.environ.get('MODAL_GPU_TYPE', 'unknown')}")
    print(f"💾 Checkpoints: /checkpoints")

    if parameters:
        print(f"⚙️  Parameters: {parameters}")

    try:
        # Execute notebook
        pm.execute_notebook(
            input_path,
            output_path,
            parameters=parameters or {},
            kernel_name='python3',
            progress_bar=False
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
def main(
    notebook: str = "notebooks/karpathy-build-gpt.ipynb",
    param: list[str] = [],  # Format: ["key=value", "key2=value2"]
):
    """
    Run a Jupyter notebook on Modal with GPU acceleration.

    Args:
        notebook: Path to notebook file
        param: Parameters to inject (format: key=value)

    Examples:
        modal run modal_notebook_runner.py
        modal run modal_notebook_runner.py --notebook notebooks/karpathy-build-gpt.ipynb
        modal run modal_notebook_runner.py --param batch_size=128 --param max_steps=10000
    """
    from pathlib import Path

    # Read notebook
    notebook_path = Path(notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    notebook_content = notebook_path.read_bytes()

    # Parse parameters
    parameters = {}
    for p in param:
        if "=" not in p:
            raise ValueError(f"Invalid parameter format: {p}. Use key=value")
        key, value = p.split("=", 1)
        # Try to parse as int/float/bool, otherwise keep as string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
        parameters[key] = value

    print(f"🚀 Launching notebook on Modal...")
    print(f"📓 Notebook: {notebook}")
    if parameters:
        print(f"⚙️  Parameters: {parameters}")

    # Run remotely
    result = run_notebook.remote(
        notebook_content=notebook_content,
        notebook_name=notebook_path.name,
        parameters=parameters
    )

    print(f"\n✅ Complete!")
    print(f"   Output: {result['output_notebook']}")
    print(f"   Checkpoints saved to Modal volume 'gpt-checkpoints'")

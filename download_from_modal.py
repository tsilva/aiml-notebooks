"""
Download executed notebooks and checkpoints from Modal volume.

Usage:
    modal run download_from_modal.py
    modal run download_from_modal.py --output-dir ./modal_outputs
"""

import modal

app = modal.App("download-results")

# Reference the same volume
volume = modal.Volume.from_name("gpt-checkpoints", create_if_missing=False)

@app.function(volumes={"/checkpoints": volume})
def list_files():
    """List all files in the checkpoints volume."""
    import os
    from pathlib import Path

    files = []
    for root, dirs, filenames in os.walk("/checkpoints"):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            size = os.path.getsize(filepath)
            files.append({
                "path": filepath,
                "size": size,
                "size_mb": size / (1024 * 1024)
            })

    return files

@app.function(volumes={"/checkpoints": volume})
def download_file(remote_path: str) -> bytes:
    """Download a file from the volume."""
    from pathlib import Path
    return Path(remote_path).read_bytes()

@app.local_entrypoint()
def main(output_dir: str = "./modal_outputs"):
    """Download all files from Modal volume."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("📋 Listing files in Modal volume...")
    files = list_files.remote()

    if not files:
        print("❌ No files found in volume")
        return

    print(f"\n📁 Found {len(files)} file(s):")
    for file in files:
        print(f"   {file['path']} ({file['size_mb']:.2f} MB)")

    print(f"\n⬇️  Downloading to {output_dir}/...")
    for file in files:
        remote_path = file['path']
        # Convert /checkpoints/file.ipynb to ./modal_outputs/file.ipynb
        relative_path = remote_path.replace("/checkpoints/", "")
        local_path = output_path / relative_path

        print(f"   Downloading {relative_path}...", end=" ")
        content = download_file.remote(remote_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(content)
        print(f"✅ ({len(content) / (1024 * 1024):.2f} MB)")

    print(f"\n✅ Download complete!")
    print(f"   Files saved to: {output_dir}/")
    print(f"\n💡 To view the executed notebook:")
    print(f"   jupyter notebook {output_dir}/karpathy-build-gpt_executed.ipynb")

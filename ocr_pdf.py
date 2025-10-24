#!/usr/bin/env python3
"""
Simple PDF OCR script using AllenAI OLM-OCR
Usage: python ocr_pdf.py document.pdf
"""

import argparse
import sys
import tempfile
import shutil
from pathlib import Path


def ocr_pdf(pdf_path: str, output_path: str = None, show_output: bool = True):
    """
    Extract text from a PDF using OLM-OCR.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the output YAML file
        show_output: Whether to print the output to console

    Returns:
        Path to the output YAML file
    """
    import subprocess

    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return None

    print(f"\n{'='*80}")
    print(f"OLM-OCR: Extracting text from PDF")
    print(f"{'='*80}")
    print(f"Input:  {pdf_path}")
    print(f"Size:   {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as tmp_workspace:
        workspace = Path(tmp_workspace)

        print(f"\n[1/2] Running OCR pipeline...")
        print(f"Note: First run will download the model (~13GB)")
        print(f"{'-'*80}")

        # Run olmocr pipeline
        try:
            cmd = [
                sys.executable, "-m", "olmocr.pipeline",
                str(workspace),
                "--pdfs", str(pdf_path),
                "--workers", "1"
            ]

            result = subprocess.run(
                cmd,
                capture_output=False,
                check=True
            )

            print(f"{'-'*80}")
            print(f"✓ OCR completed")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ OCR pipeline failed with exit code {e.returncode}")
            return None
        except Exception as e:
            print(f"\n✗ Error: {e}")
            return None

        # Find the output file
        print(f"\n[2/2] Collecting results...")

        # The pipeline creates output in workspace/output/ directory
        output_files = list(workspace.glob("**/output/**/*.yaml"))
        if not output_files:
            # Try alternative locations
            output_files = list(workspace.glob("**/*.yaml"))

        if not output_files:
            print("⚠ No output files found in workspace")
            print(f"Workspace contents:")
            for item in workspace.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(workspace)}")
            return None

        # Use the first output file
        result_file = output_files[0]

        # Read the content
        with open(result_file, 'r') as f:
            content = f.read()

        # Save to output file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"✓ Saved to: {output_path}")

        # Show output if requested
        if show_output:
            print(f"\n{'='*80}")
            print(f"EXTRACTED TEXT")
            print(f"{'='*80}")
            print(content)
            print(f"{'='*80}")

        # If output path specified, copy the file before temp dir is deleted
        if output_path:
            return Path(output_path)
        else:
            # Copy to current directory with same name + .yaml
            default_output = Path(pdf_path.stem + "_ocr.yaml")
            shutil.copy(result_file, default_output)
            print(f"✓ Saved to: {default_output}")
            return default_output


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF using OLM-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OCR a PDF and show results
  %(prog)s document.pdf

  # Save to specific output file
  %(prog)s document.pdf -o output.yaml

  # Quiet mode (only save to file)
  %(prog)s document.pdf -o output.yaml --quiet
"""
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file to OCR"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output YAML file path (default: <pdf_name>_ocr.yaml)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Don't print output to console"
    )

    args = parser.parse_args()

    # Run OCR
    result = ocr_pdf(
        args.pdf,
        output_path=args.output,
        show_output=not args.quiet
    )

    if result:
        print(f"\n✓ Done! Output saved to: {result}")
        return 0
    else:
        print(f"\n✗ OCR failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

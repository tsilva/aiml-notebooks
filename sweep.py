#!/usr/bin/env python
"""
Generic W&B Sweep Runner

This script runs W&B hyperparameter sweeps on Jupyter notebooks by:
1. Converting the notebook to a Python script (saved to tmp/sweeps/scripts/)
2. Loading sweep configuration from a YAML file
3. Running the sweep with the specified configuration

Usage:
    # Run sweep with config file and notebook
    uv run python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb

    # Initialize sweep only (don't run agent)
    uv run python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb --init-only

    # Run specific number of trials
    uv run python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb --count 5
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import wandb


def convert_notebook_to_script(notebook_path: Path, output_path: Path = None) -> Path:
    """Convert a Jupyter notebook to a Python script.

    Args:
        notebook_path: Path to the .ipynb file
        output_path: Optional output path for the .py file

    Returns:
        Path to the generated Python script
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    if output_path is None:
        # Generate output path in tmp/sweeps/scripts/ directory
        scripts_dir = Path('tmp/sweeps/scripts')
        scripts_dir.mkdir(parents=True, exist_ok=True)
        output_path = scripts_dir / f"{notebook_path.stem}.py"

    print(f"Converting notebook to script...")
    print(f"  Input:  {notebook_path}")
    print(f"  Output: {output_path}")

    cmd = [
        'jupyter', 'nbconvert',
        '--to', 'script',
        str(notebook_path),
        '--output', str(output_path.absolute())
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error converting notebook: {result.stderr}")
        sys.exit(1)

    # Clean up the double .py.py extension if it exists
    double_ext_path = Path(str(output_path) + '.py')
    if double_ext_path.exists():
        double_ext_path.rename(output_path)

    # Post-process the script to guard IPython-specific commands
    _guard_ipython_commands(output_path)

    print(f"✓ Conversion successful: {output_path.name}\n")
    return output_path


def _guard_ipython_commands(script_path: Path):
    """Guard IPython-specific commands and wrap execution in if __name__ == '__main__'.

    This function:
    1. Wraps get_ipython() calls in try-except blocks
    2. Adds if __name__ == '__main__': guard to prevent code execution during imports
    """
    with open(script_path, 'r') as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    # First pass: guard IPython commands
    modified_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line contains a get_ipython() call
        if 'get_ipython()' in line and not line.strip().startswith('#'):
            # Add a try-except wrapper
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent

            modified_lines.append(f"{indent_str}try:\n")
            # Add extra indentation to the original line
            modified_lines.append(f"{indent_str}    {line.lstrip()}")
            modified_lines.append(f"{indent_str}except NameError:\n")
            modified_lines.append(f"{indent_str}    pass  # Not in IPython environment\n")
        else:
            modified_lines.append(line)

        i += 1

    # Second pass: Find where executable code starts (after imports and function definitions)
    # Look for the first line that's not an import, comment, blank line, or function/class definition
    execution_start_idx = None
    in_multiline_string = False
    in_multiline_statement = False  # Track parentheses for multi-line statements
    paren_depth = 0

    for i, line in enumerate(modified_lines):
        stripped = line.strip()

        # Track multiline strings
        if '"""' in stripped or "'''" in stripped:
            in_multiline_string = not in_multiline_string
            continue

        if in_multiline_string:
            continue

        # Track parentheses depth for multi-line statements
        paren_depth += stripped.count('(') - stripped.count(')')
        if paren_depth > 0:
            in_multiline_statement = True
            continue
        elif in_multiline_statement:
            # Just finished a multi-line statement
            in_multiline_statement = False
            continue

        # Skip empty lines, comments, imports, function/class definitions
        if (not stripped or
            stripped.startswith('#') or
            stripped.startswith('import ') or
            stripped.startswith('from ') or
            stripped.startswith('def ') or
            stripped.startswith('class ') or
            stripped.startswith('@')):
            continue

        # Found first executable code
        execution_start_idx = i
        break

    # If we found executable code, wrap it in if __name__ == '__main__':
    if execution_start_idx is not None:
        # Split into imports/definitions and executable code
        header = modified_lines[:execution_start_idx]
        executable = modified_lines[execution_start_idx:]

        # Indent all executable code
        indented_executable = []
        for line in executable:
            if line.strip():  # Don't indent empty lines at start
                indented_executable.append('    ' + line)
            else:
                indented_executable.append(line)

        # Combine with if __name__ guard
        modified_lines = (
            header +
            ['if __name__ == \'__main__\':\n'] +
            indented_executable
        )

    # Write the modified content back
    with open(script_path, 'w') as f:
        f.writelines(modified_lines)


def load_sweep_config(config_path: Path) -> dict:
    """Load sweep configuration from YAML file.

    Args:
        config_path: Path to the sweep config YAML file

    Returns:
        Sweep configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    print(f"Loading sweep config: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'name' not in config:
        raise ValueError("Sweep config must have a 'name' field")
    if 'metric' not in config:
        raise ValueError("Sweep config must have a 'metric' field")
    if 'parameters' not in config:
        raise ValueError("Sweep config must have a 'parameters' field")

    print(f"  Name: {config['name']}")
    print(f"  Method: {config.get('method', 'grid')}")
    print(f"  Metric: {config['metric']['name']} ({config['metric']['goal']})")
    print(f"  Parameters: {len(config['parameters'])} hyperparameters\n")

    return config


def run_sweep(config: dict, script_path: Path, count: int = None,
              project: str = None, init_only: bool = False):
    """Initialize and run a W&B sweep.

    Args:
        config: Sweep configuration dictionary
        script_path: Path to the training script
        count: Number of runs to execute (None = unlimited)
        project: W&B project name (uses config default if not specified)
        init_only: If True, only initialize the sweep without running agents
    """
    # Ensure script path is in the config (use relative path from repo root)
    if 'program' not in config:
        # Make path relative to current directory
        try:
            config['program'] = str(script_path.relative_to(Path.cwd()))
        except ValueError:
            # If not relative, use absolute path
            config['program'] = str(script_path)

    # Use project from config or argument
    if project is None:
        project = config.get('project', 'ml-sweeps')

    print("="*60)
    print(f"INITIALIZING W&B SWEEP: {config['name']}")
    print("="*60)
    print(f"Program: {config['program']}")
    print(f"Project: {project}")
    print("="*60 + "\n")

    # Initialize sweep
    sweep_id = wandb.sweep(config, project=project)

    print(f"✨ Sweep ID: {sweep_id}")
    print(f"📊 View at: https://wandb.ai//{project}/sweeps/{sweep_id.split('/')[-1]}\n")

    if init_only:
        print("Sweep initialized. To run agents:")
        print(f"  uv run wandb agent {sweep_id}")
        print()
        return

    if count:
        print(f"Starting {count} sweep trials...")
    else:
        print("Starting sweep (Ctrl+C to stop)...")

    print("\nTo run additional agents in parallel, execute in another terminal:")
    print(f"  uv run wandb agent {sweep_id}\n")

    # Run sweep agent using subprocess
    cmd = ['wandb', 'agent', sweep_id]
    if count:
        cmd.extend(['--count', str(count)])

    try:
        # Run from the repo root directory (where program paths are relative to)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        return
    except subprocess.CalledProcessError as e:
        print(f"\n\nSweep failed with error: {e}")
        return

    print("\n" + "="*60)
    print("✅ SWEEP COMPLETE!")
    print("="*60)
    print(f"View results: https://wandb.ai//{project}/sweeps/{sweep_id.split('/')[-1]}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run W&B hyperparameter sweep on a Jupyter notebook',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sweep with config and notebook
  python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb

  # Initialize only (don't run agents)
  python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb --init-only

  # Run 5 trials
  python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb --count 5

  # Use existing Python script instead of notebook
  python sweep.py path/to/config.yaml tmp/sweeps/scripts/model.py --no-convert
        """
    )

    parser.add_argument('config', type=Path,
                       help='Path to sweep configuration YAML file')
    parser.add_argument('notebook', type=Path,
                       help='Path to Jupyter notebook (.ipynb) or Python script (.py)')
    parser.add_argument('--count', type=int, default=None,
                       help='Number of trials to run (default: unlimited for bayes, all for grid)')
    parser.add_argument('--project', type=str, default=None,
                       help='W&B project name (default: from config or "ml-sweeps")')
    parser.add_argument('--init-only', action='store_true',
                       help='Only initialize sweep, don\'t run agent')
    parser.add_argument('--no-convert', action='store_true',
                       help='Skip notebook conversion (use if input is already a .py file)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output path for converted script (default: tmp/sweeps/scripts/notebook_name.py)')

    args = parser.parse_args()

    # Validate inputs
    if not args.config.exists():
        print(f"Error: Sweep config not found: {args.config}")
        sys.exit(1)

    if not args.notebook.exists():
        print(f"Error: Notebook/script not found: {args.notebook}")
        sys.exit(1)

    # Load sweep configuration
    try:
        config = load_sweep_config(args.config)
    except Exception as e:
        print(f"Error loading sweep config: {e}")
        sys.exit(1)

    # Convert notebook to script if needed
    if args.notebook.suffix == '.ipynb' and not args.no_convert:
        try:
            script_path = convert_notebook_to_script(args.notebook, args.output)
        except Exception as e:
            print(f"Error converting notebook: {e}")
            sys.exit(1)
    elif args.notebook.suffix == '.py':
        script_path = args.notebook
        print(f"Using existing Python script: {script_path}\n")
    else:
        print(f"Error: Unsupported file type: {args.notebook.suffix}")
        print("Supported types: .ipynb, .py")
        sys.exit(1)

    # Run sweep
    try:
        run_sweep(
            config=config,
            script_path=script_path,
            count=args.count,
            project=args.project,
            init_only=args.init_only
        )
    except Exception as e:
        print(f"Error running sweep: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Check notebooks for markdown cell before every code cell."""

import json
import sys
from pathlib import Path


def check_notebook(notebook_path):
    """Check if notebook has markdown cell before every code cell.

    Returns list of violations (code cell indices without markdown before them).
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    violations = []
    cells = nb.get('cells', [])

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', '')

        # Check if this is a code cell
        if cell_type == 'code':
            # Check if previous cell is markdown (or if this is first cell)
            if i == 0:
                violations.append((i, "First cell is code (no markdown before it)"))
            else:
                prev_cell_type = cells[i-1].get('cell_type', '')
                if prev_cell_type != 'markdown':
                    violations.append((i, f"Code cell after {prev_cell_type} cell (no markdown before it)"))

    return violations


def main():
    notebooks_dir = Path('notebooks')
    all_violations = {}

    for notebook_path in sorted(notebooks_dir.glob('*.ipynb')):
        violations = check_notebook(notebook_path)
        if violations:
            all_violations[notebook_path.name] = violations

    if all_violations:
        print(f"Found {len(all_violations)} notebooks with violations:\n")
        for notebook_name, violations in all_violations.items():
            print(f"\n{notebook_name}:")
            for cell_idx, reason in violations:
                print(f"  - Cell {cell_idx}: {reason}")
    else:
        print("All notebooks follow the markdown-before-code rule!")

    return 0 if not all_violations else 1


if __name__ == '__main__':
    sys.exit(main())

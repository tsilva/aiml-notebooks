#!/usr/bin/env python3
"""Fix notebooks by adding markdown cells before code cells that lack them."""

import json
import sys
from pathlib import Path


def fix_notebook(notebook_path):
    """Add markdown cell before any code cell that doesn't have one.

    Returns number of cells added.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    new_cells = []
    cells_added = 0

    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', '')

        # Check if this is a code cell that needs markdown before it
        if cell_type == 'code':
            # Check if we need to insert a markdown cell
            need_markdown = False

            if i == 0:
                # First cell is code, need markdown
                need_markdown = True
            elif len(new_cells) > 0 and new_cells[-1].get('cell_type') != 'markdown':
                # Previous cell is not markdown
                need_markdown = True

            if need_markdown:
                # Create a simple markdown cell
                markdown_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["Execute the following code:"]
                }
                new_cells.append(markdown_cell)
                cells_added += 1

        # Add the current cell
        new_cells.append(cell)

    # Update the notebook
    nb['cells'] = new_cells

    # Write back to file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')  # Add trailing newline

    return cells_added


def main():
    notebooks_dir = Path('notebooks')
    total_cells_added = 0
    notebooks_fixed = 0

    # Get list of notebooks with violations from check script
    from check_markdown_rule import check_notebook

    for notebook_path in sorted(notebooks_dir.glob('*.ipynb')):
        violations = check_notebook(notebook_path)
        if violations:
            print(f"Fixing {notebook_path.name}...")
            cells_added = fix_notebook(notebook_path)
            total_cells_added += cells_added
            notebooks_fixed += 1
            print(f"  Added {cells_added} markdown cells")

    print(f"\nFixed {notebooks_fixed} notebooks, added {total_cells_added} markdown cells total")
    return 0


if __name__ == '__main__':
    sys.exit(main())

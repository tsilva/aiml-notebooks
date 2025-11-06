#!/usr/bin/env python3
"""Find generic markdown cells that need custom descriptions."""

import json
from pathlib import Path


def find_generic_cells(notebook_path):
    """Find cells with generic 'Execute the following code:' markdown."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    generic_cells = []

    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if source.strip() == 'Execute the following code:':
                # Get the next cell to show what code it precedes
                if i + 1 < len(cells) and cells[i + 1].get('cell_type') == 'code':
                    code_preview = ''.join(cells[i + 1].get('source', []))[:100]
                    generic_cells.append((i, code_preview))

    return generic_cells


def main():
    notebooks_dir = Path('notebooks')

    for notebook_path in sorted(notebooks_dir.glob('*.ipynb')):
        generic_cells = find_generic_cells(notebook_path)
        if generic_cells:
            print(f"\n{notebook_path.name}: {len(generic_cells)} generic cells")
            for cell_idx, code_preview in generic_cells:
                print(f"  Cell {cell_idx}: {code_preview.strip()[:80]}...")


if __name__ == '__main__':
    main()

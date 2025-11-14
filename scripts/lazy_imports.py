#!/usr/bin/env python3
"""
Refactor notebook imports to use lazy loading pattern.
Moves imports from the top of the notebook to right before first use.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_import_statement(line: str) -> Set[str]:
    """Extract symbols that will be available after this import."""
    line = line.strip()
    symbols = set()

    # Handle 'import X' or 'import X as Y'
    if line.startswith('import ') and ' from ' not in line:
        parts = line.replace('import ', '').split(',')
        for part in parts:
            if ' as ' in part:
                # import X as Y -> use Y
                symbols.add(part.split(' as ')[1].strip())
            else:
                # import X -> use X or X.something
                symbol = part.strip().split('.')[0]
                symbols.add(symbol)

    # Handle 'from X import Y' or 'from X import Y as Z'
    elif line.startswith('from '):
        # from X import Y, Z
        if ' import ' in line:
            import_part = line.split(' import ', 1)[1]
            if import_part.strip() == '*':
                # Can't track star imports easily, skip
                return symbols

            items = import_part.split(',')
            for item in items:
                if ' as ' in item:
                    symbols.add(item.split(' as ')[1].strip())
                else:
                    # Handle parenthesized imports
                    item = item.strip().strip('(').strip(')')
                    if item:
                        symbols.add(item.strip())

    return symbols


def find_symbol_usage(code: str, symbol: str) -> bool:
    """Check if a symbol is used in the code."""
    # Simple heuristic: look for the symbol as a word boundary
    # This might have false positives but should work for most cases
    pattern = r'\b' + re.escape(symbol) + r'\b'
    return bool(re.search(pattern, code))


def extract_imports_from_cell(cell: dict) -> List[Tuple[str, Set[str]]]:
    """Extract import statements and their symbols from a code cell."""
    if cell['cell_type'] != 'code':
        return []

    source = cell.get('source', [])
    if isinstance(source, str):
        source = source.split('\n')

    imports = []
    for line in source:
        line = line.strip()
        # Skip IPython magics and non-import lines
        if line.startswith('%') or line.startswith('!'):
            continue
        if line.startswith('import ') or line.startswith('from '):
            symbols = parse_import_statement(line)
            imports.append((line, symbols))

    return imports


def is_setup_cell(cell: dict, cell_index: int) -> bool:
    """Check if this is a setup/import cell that should be refactored."""
    if cell['cell_type'] != 'code':
        return False

    # Typically setup cells are in the first 5 cells
    if cell_index > 10:
        return False

    source = cell.get('source', [])
    if isinstance(source, str):
        source = source.split('\n')

    # Check if cell is mostly imports
    import_lines = 0
    code_lines = 0

    for line in source:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('%') or line.startswith('!'):
            continue
        if line.startswith('import ') or line.startswith('from '):
            import_lines += 1
        else:
            code_lines += 1

    # Cell is a setup cell if it has imports and little other code
    # Allow some setup code (like set_seed, device config)
    return import_lines > 0 and import_lines >= code_lines * 0.5


def refactor_notebook(notebook_path: Path, dry_run: bool = False) -> bool:
    """Refactor a notebook to use lazy imports."""
    print(f"\nProcessing: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    # Step 1: Find setup cells with imports
    all_imports: List[Tuple[int, str, Set[str]]] = []  # (cell_idx, import_line, symbols)
    setup_cell_indices = []

    for idx, cell in enumerate(cells):
        if is_setup_cell(cell, idx):
            setup_cell_indices.append(idx)
            imports = extract_imports_from_cell(cell)
            for import_line, symbols in imports:
                all_imports.append((idx, import_line, symbols))

    if not all_imports:
        print(f"  ✓ No stacked imports found (already lazy or no imports)")
        return False

    print(f"  Found {len(all_imports)} imports in {len(setup_cell_indices)} setup cells")

    # Step 2: For each import, find first usage
    import_placements: Dict[str, int] = {}  # import_line -> target_cell_idx

    for cell_idx, import_line, symbols in all_imports:
        # Find first cell after the setup cells that uses any of these symbols
        first_usage_idx = None

        for idx in range(max(setup_cell_indices) + 1, len(cells)):
            cell = cells[idx]
            if cell['cell_type'] != 'code':
                continue

            source = cell.get('source', [])
            if isinstance(source, str):
                code = source
            else:
                code = ''.join(source)

            # Check if any symbol is used
            for symbol in symbols:
                if find_symbol_usage(code, symbol):
                    first_usage_idx = idx
                    break

            if first_usage_idx is not None:
                break

        # If we found usage, record it; otherwise skip this import (might be unused)
        if first_usage_idx is not None:
            import_placements[import_line] = first_usage_idx

    if not import_placements:
        print(f"  ⚠ No usage found for any imports (might be unused or pattern not detected)")
        return False

    print(f"  Will move {len(import_placements)} imports to their first usage locations")

    if dry_run:
        print(f"  [DRY RUN] Would refactor this notebook")
        for import_line, target_idx in sorted(import_placements.items(), key=lambda x: x[1]):
            print(f"    '{import_line[:50]}...' -> cell {target_idx}")
        return False

    # Step 3: Create new cells with imports before first usage
    # We'll mark cells to modify and then apply all changes
    cells_to_insert = []  # (insert_before_idx, new_cells)

    for import_line, target_idx in sorted(import_placements.items(), key=lambda x: x[1]):
        # Check if there's already a markdown cell before this code cell
        needs_markdown = True
        if target_idx > 0 and cells[target_idx - 1]['cell_type'] == 'markdown':
            needs_markdown = False

        new_cells = []

        # Add markdown cell if needed
        if needs_markdown:
            # Create a simple descriptive markdown
            markdown_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': ['Import necessary libraries for this section.']
            }
            new_cells.append(markdown_cell)

        # Create code cell with the import
        import_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [import_line]
        }
        new_cells.append(import_cell)

        cells_to_insert.append((target_idx, new_cells))

    # Step 4: Remove imports from original setup cells
    # Remove entire cell if it only has imports, otherwise remove just the import lines
    cells_to_remove = []

    for idx in setup_cell_indices:
        cell = cells[idx]
        source = cell.get('source', [])
        if isinstance(source, str):
            source = source.split('\n')

        # Filter out import lines that we moved
        new_source = []
        for line in source:
            line_stripped = line.strip()
            # Keep non-import lines and imports we didn't move
            if not (line_stripped.startswith('import ') or line_stripped.startswith('from ')):
                new_source.append(line)
            elif line_stripped not in import_placements:
                # Keep imports we didn't move (e.g., IPython magics handled separately)
                new_source.append(line)

        # If cell is now empty or only has comments/whitespace, mark for removal
        has_content = False
        for line in new_source:
            line = line.strip()
            if line and not line.startswith('#'):
                # Keep IPython magics like %load_ext
                if line.startswith('%') or line.startswith('!'):
                    has_content = True
                elif not (line.startswith('import ') or line.startswith('from ')):
                    has_content = True

        if has_content:
            # Update cell with filtered source
            if isinstance(cell['source'], str):
                cell['source'] = '\n'.join(new_source)
            else:
                cell['source'] = new_source
        else:
            # Mark cell for removal
            cells_to_remove.append(idx)

    # Step 5: Apply changes - remove cells first (in reverse order to maintain indices)
    for idx in reversed(cells_to_remove):
        del cells[idx]
        # Adjust insertion indices
        cells_to_insert = [(i - 1 if i > idx else i, c) for i, c in cells_to_insert]

    # Then insert new cells (in reverse order to maintain indices)
    for target_idx, new_cells in reversed(cells_to_insert):
        for cell in reversed(new_cells):
            cells.insert(target_idx, cell)

    # Step 6: Save the modified notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print(f"  ✅ Refactored successfully")
    return True


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    # Find all notebooks
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    print(f"Found {len(notebooks)} notebooks to process\n")

    modified_count = 0
    for notebook_path in notebooks:
        try:
            if refactor_notebook(notebook_path, dry_run):
                modified_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'Would modify' if dry_run else 'Modified'} {modified_count}/{len(notebooks)} notebooks")


if __name__ == '__main__':
    main()

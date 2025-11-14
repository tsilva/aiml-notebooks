#!/usr/bin/env python3
"""
Simpler approach: Refactor notebooks to use lazy imports by adding import statements
right before their first use in the notebook.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def extract_imports(source_lines: List[str]) -> List[str]:
    """Extract all import lines from source."""
    imports = []
    for line in source_lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')) and ' import ' in stripped:
            if not stripped.startswith('%'):  # Skip IPython magics
                imports.append(line.rstrip())
    return imports


def get_imported_names(import_line: str) -> Set[str]:
    """Extract the names that will be available after this import."""
    line = import_line.strip()
    names = set()

    if line.startswith('from '):
        # from X import Y, Z or from X import Y as Z
        if ' import ' in line:
            import_part = line.split(' import ', 1)[1]
            # Handle parentheses and star imports
            if import_part.strip() == '*':
                return names  # Can't track easily
            import_part = import_part.replace('(', '').replace(')', '')
            items = [item.strip() for item in import_part.split(',')]
            for item in items:
                if ' as ' in item:
                    names.add(item.split(' as ')[1].strip())
                else:
                    names.add(item.strip())
    elif line.startswith('import '):
        # import X, Y or import X as Y
        import_part = line.replace('import ', '', 1)
        items = [item.strip() for item in import_part.split(',')]
        for item in items:
            if ' as ' in item:
                names.add(item.split(' as ')[1].strip())
            else:
                # For import x.y.z, the name is just 'x'
                names.add(item.split('.')[0].strip())

    return names


def find_first_usage(cells: List[dict], names: Set[str], start_from: int) -> int:
    """Find the first cell that uses any of the given names."""
    for idx in range(start_from, len(cells)):
        cell = cells[idx]
        if cell['cell_type'] != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source

        # Check if any name is used
        for name in names:
            # Simple regex to find word boundaries
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, code):
                return idx

    return -1  # Not found


def refactor_notebook(notebook_path: Path) -> bool:
    """Refactor a notebook to use lazy imports."""
    print(f"\n{notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    cells = notebook['cells']

    # Find first code cell with many imports (likely a setup cell)
    setup_cell_idx = -1
    all_imports = []

    for idx, cell in enumerate(cells[:5]):  # Check first 5 cells
        if cell['cell_type'] != 'code':
            continue

        source = cell.get('source', [])
        if isinstance(source, list):
            source_lines = source
        else:
            source_lines = source.split('\n')

        imports = extract_imports(source_lines)
        if len(imports) >= 3:  # Has multiple imports
            setup_cell_idx = idx
            all_imports = imports
            break

    if setup_cell_idx == -1 or not all_imports:
        print("  ✓ No setup cell with stacked imports found")
        return False

    print(f"  Found {len(all_imports)} imports in cell {setup_cell_idx}")

    # Map each import to where it should go
    import_placements: Dict[str, int] = {}

    for import_line in all_imports:
        names = get_imported_names(import_line)
        if not names:
            continue

        # Find first usage after setup cell
        first_use_idx = find_first_usage(cells, names, setup_cell_idx + 1)

        if first_use_idx > 0:
            import_placements[import_line] = first_use_idx

    if not import_placements:
        print("  ⚠ No usages found")
        return False

    print(f"  Will add {len(import_placements)} lazy imports")

    # Group imports by target cell
    imports_by_cell = defaultdict(list)
    for import_line, target_idx in import_placements.items():
        imports_by_cell[target_idx].append(import_line)

    # Add imports before each target cell
    # Work backwards to preserve indices
    for target_idx in sorted(imports_by_cell.keys(), reverse=True):
        import_lines = imports_by_cell[target_idx]

        # Create a code cell with these imports
        import_cell = {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': import_lines
        }

        # Check if previous cell is markdown
        if target_idx > 0 and cells[target_idx - 1]['cell_type'] == 'markdown':
            # Insert after markdown
            cells.insert(target_idx, import_cell)
        else:
            # Add markdown + import cell
            markdown_cell = {
                'cell_type': 'markdown',
                'metadata': {},
                'source': ['Import required libraries.']
            }
            cells.insert(target_idx, markdown_cell)
            cells.insert(target_idx + 1, import_cell)

    # Remove imports from original setup cell and mark it
    setup_cell = cells[setup_cell_idx]
    source = setup_cell.get('source', [])
    if isinstance(source, list):
        source_lines = source
    else:
        source_lines = source.split('\n')

    # Keep only IPython magics and non-import lines
    new_source = []
    has_content = False
    for line in source_lines:
        stripped = line.strip()
        # Keep IPython magics
        if stripped.startswith('%') or stripped.startswith('!'):
            new_source.append(line)
            has_content = True
        # Keep non-import lines
        elif not (stripped.startswith('import ') or stripped.startswith('from ')):
            if stripped and not stripped.startswith('#'):
                has_content = True
            new_source.append(line)

    # Update or remove the setup cell
    if has_content:
        if isinstance(setup_cell['source'], list):
            setup_cell['source'] = new_source
        else:
            setup_cell['source'] = '\n'.join(new_source)
        # Update metadata to clear execution count
        setup_cell['execution_count'] = None
        setup_cell['outputs'] = []
    else:
        # Remove empty setup cell
        del cells[setup_cell_idx]

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

    print(f"  ✅ Refactored")
    return True


def main():
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    print(f"Refactoring {len(notebooks)} notebooks...\n")

    modified = 0
    for nb_path in notebooks:
        try:
            if refactor_notebook(nb_path):
                modified += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Modified {modified}/{len(notebooks)} notebooks")


if __name__ == '__main__':
    main()

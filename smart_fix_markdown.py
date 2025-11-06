#!/usr/bin/env python3
"""Intelligently fix generic markdown cells based on code content."""

import json
import re
from pathlib import Path


def infer_description(code_source):
    """Infer a good description from code content."""
    code = ''.join(code_source).strip()

    # Check for visualization/plotting
    if 'plt.' in code or 'ax.' in code or 'fig,' in code:
        if 'subplot' in code:
            return 'Visualize the results with multiple plots for comparison.'
        elif 'scatter' in code:
            return 'Visualize the data distribution with a scatter plot.'
        elif 'hist' in code:
            return 'Plot a histogram to understand the distribution.'
        elif 'bar' in code:
            return 'Create a bar chart to compare values.'
        else:
            return 'Visualize the results.'

    # Check for training loops
    if 'for epoch in' in code or 'train_model' in code or 'fit(' in code:
        if 'print(' in code and 'Training' in code:
            return 'Train the model and monitor progress.'
        return 'Execute the training loop.'

    # Check for model evaluation/testing
    if 'model.eval()' in code or 'with torch.no_grad()' in code:
        return 'Evaluate the model on the test set.'

    # Check for data loading/preparation
    if 'DataLoader' in code or 'load_data' in code or 'Dataset' in code:
        return 'Prepare the data loaders for training and validation.'

    # Check for model definition
    if 'class ' in code and 'nn.Module' in code:
        class_match = re.search(r'class\s+(\w+)', code)
        if class_match:
            return f'Define the {class_match.group(1)} architecture.'
        return 'Define the model architecture.'

    # Check for results/metrics display
    if 'print(' in code:
        if 'accuracy' in code.lower() or 'loss' in code.lower():
            return 'Display training results and metrics.'
        if 'Final' in code or 'Results' in code:
            return 'Show the final results.'
        if '=' * 40 in code or '=' * 50 in code or '=' * 60 in code:
            return 'Print a summary of the results.'
        return 'Display the output.'

    # Check for comparisons
    if 'compare' in code.lower() or 'vs' in code.lower():
        return 'Compare the different approaches.'

    # Check for experiments
    if 'experiment' in code.lower() or 'test_' in code:
        return 'Run the experiment with different configurations.'

    # Check for specific keywords
    if 'initialize' in code.lower() or 'setup' in code.lower():
        return 'Initialize the experiment setup.'

    # Default fallback
    if len(code) < 50:
        return 'Execute a quick computation.'
    else:
        return 'Run the following analysis.'


def fix_notebook(notebook_path):
    """Fix generic markdown cells in a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    fixes_made = 0

    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if source.strip() == 'Execute the following code:':
                # Get the next code cell
                if i + 1 < len(cells) and cells[i + 1].get('cell_type') == 'code':
                    code_source = cells[i + 1].get('source', [])
                    new_description = infer_description(code_source)
                    cell['source'] = [new_description]
                    fixes_made += 1

    if fixes_made > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')

    return fixes_made


def main():
    notebooks_dir = Path('notebooks')
    total_fixes = 0
    notebooks_fixed = 0

    for notebook_path in sorted(notebooks_dir.glob('*.ipynb')):
        fixes = fix_notebook(notebook_path)
        if fixes > 0:
            total_fixes += fixes
            notebooks_fixed += 1
            print(f'Fixed {notebook_path.name}: {fixes} cells')

    print(f'\nTotal: Fixed {notebooks_fixed} notebooks, {total_fixes} cells')


if __name__ == '__main__':
    main()

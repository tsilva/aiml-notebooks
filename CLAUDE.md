# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of AI/ML Jupyter notebooks for learning and experimentation. The project contains detailed recreations of lessons from Andrej Karpathy's "Neural Networks: Zero to Hero" course, along with miscellaneous standalone experiments exploring various AI/ML concepts. Notebooks are deliberately more detailed than source material to serve as comprehensive references.

## Development Environment

This project uses **uv** for fast, reliable Python dependency management (NOT conda or pip). The project is configured as a non-package project (notebooks only) in `pyproject.toml`.

### Setup Commands

```bash
# Install dependencies (creates/syncs .venv)
uv sync

# Run Jupyter Lab
uv run jupyter lab

# Run Python scripts (e.g., fix_notebooks.py)
uv run python fix_notebooks.py
```

**Important**: Always use `uv run` to execute commands in the project's virtual environment. Do not manually activate the venv unless specifically requested.

### Key Dependencies

- Python 3.11+
- PyTorch (with MPS support on macOS for GPU acceleration)
- Jupyter/JupyterLab for notebook development
- transformers, timm for deep learning
- numpy, matplotlib, pillow for data/visualization
- graphviz for network visualization

## Repository Structure

```
notebooks/          # All Jupyter notebooks
├── zero2hero-XXX-*.ipynb    # Karpathy course recreations (numbered)
├── wip-*.ipynb              # Work-in-progress experiments
├── *.ipynb                  # Completed standalone notebooks
└── *.todo.md                # Todo lists for specific WIP notebooks

fix_notebooks.py    # Utility to fix Colab links and widget metadata
pyproject.toml      # Project dependencies (uv configuration)
uv.lock            # Locked dependencies for reproducibility
environment.yml     # Legacy conda config (not used; use uv instead)
```

## Notebook Conventions

### Naming Patterns

- `zero2hero-NNN-description.ipynb` - Numbered notebooks from Karpathy's course (in sequence)
- `wip-description.ipynb` - Work-in-progress experiments (incomplete/draft)
- `description.ipynb` - Completed standalone experiments

### Colab Integration

All notebooks include a "Open In Colab" badge/link at the top pointing to:
`https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/notebooks/filename.ipynb`

Use `fix_notebooks.py` to automatically update Colab links and fix widget metadata issues across all notebooks:

```bash
uv run python fix_notebooks.py
```

This script:
1. Fixes widget metadata structure (adds empty state if missing)
2. Updates Colab badge links to match correct repository paths
3. Processes all `.ipynb` files recursively (skips `.ipynb_checkpoints`)

## Working with Notebooks

### Creating New Notebooks

1. Place notebooks in the `notebooks/` directory
2. Use appropriate naming prefix (`wip-` for incomplete, `zero2hero-NNN-` for course recreations)
3. Add Colab badge at the top (markdown or HTML format)
4. Run `fix_notebooks.py` to ensure proper metadata and links

### Editing Notebooks

- Notebooks are meant to be detailed with extensive explanations
- Include mathematical formulas, visualizations, and step-by-step commentary
- Use markdown cells liberally to explain concepts
- For WIP notebooks, consider creating a `.todo.md` file to track progress

### GPU Acceleration

- **macOS**: PyTorch uses MPS (Metal Performance Shaders) for GPU acceleration
- **CUDA**: Only available on Linux/Windows (not macOS)
- Check device availability in notebooks: `torch.backends.mps.is_available()` or `torch.cuda.is_available()`

## Git Workflow

- Main branch: `main`
- `.gitignore` excludes: `.venv/`, `.ipynb_checkpoints`, `notebooks/tmp`, `notebooks/output`, `notebooks/wandb`, `notebooks/lightning_logs`, `.env`
- The `uv.lock` file is committed for reproducible installations

## Common Tasks

### Run a notebook
```bash
uv run jupyter lab notebooks/zero2hero-001-backprop-from-scratch.ipynb
```

### Fix all notebook metadata and Colab links
```bash
uv run python fix_notebooks.py
```

### Add a new dependency
```bash
# Add to pyproject.toml dependencies list, then:
uv sync
```

### Update all dependencies
```bash
uv lock --upgrade
uv sync
```

## Architecture Notes

- This is a **non-package project** (no importable Python package, only notebooks)
- Notebooks are self-contained; each can run independently
- No shared library code - each notebook implements what it needs
- Data files and outputs should go in `notebooks/tmp`, `notebooks/output`, or similar (gitignored)

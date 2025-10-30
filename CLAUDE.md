# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

AI/ML Jupyter notebooks for learning and experimentation. Includes course recreations and standalone experiments. The local `aiml_notebooks` package in `src/` provides shared utilities (tokenizers, datasets, etc.).

## Development Environment

**Package Manager**: Use `uv` (NOT conda or pip)

```bash
uv sync                    # Install/sync dependencies
uv run jupyter lab         # Run Jupyter Lab
```

**Important**: Always use `uv run` to execute commands. Don't manually activate venv unless requested.

## Repository Structure

```
notebooks/              # Active notebooks (work with these)
├── <prefix>-NNN-*.ipynb  # Numbered course recreations
├── wip-*.ipynb           # Work in progress
└── *.ipynb               # Completed standalone

_deprecated-notebooks/  # IGNORE - archived notebooks

src/aiml_notebooks/     # Shared library
├── tokenizers.py       # CharacterTokenizer
└── datasets.py         # Datasets, factories, utilities

pyproject.toml          # Dependencies (uv config)
uv.lock                 # Locked dependencies
```

**IMPORTANT**: Only work with notebooks in `notebooks/` directory. Ignore deprecated directories.

## Notebook Conventions

**Naming**:
- `<prefix>-NNN-description.ipynb` - Numbered course recreations
- `wip-description.ipynb` - Work in progress
- `description.ipynb` - Completed standalone

**Philosophy**: Create highly engaging, interactive notebooks (50-80 cells) that build deep intuitions effortlessly through:
- **Building blocks approach**: Progress step-by-step, each concept building on the last
- **Concise with flow**: Clear narrative, always markdown before code (even one line)
- **Theory before practice**: Explain "why" before "how"
- **Progressive visualizations**: Show concepts visually as they develop
- **Self-contained**: Runnable end-to-end, no external dependencies on other notebooks

## Working with Notebooks

### Creating/Editing

**Critical rules**:
1. **Always** place markdown cell before each code cell (even if just one explanatory line)
2. Each cell should have a clear purpose in the learning progression
3. Build intuition through small, incremental steps (not large code dumps)
4. Include visualizations after introducing new concepts
5. Use formulas, diagrams, and examples liberally
6. For WIP notebooks, use `.todo.md` for tracking

Place notebooks in `notebooks/` with appropriate naming.

### Using Shared Library

```python
from aiml_notebooks import CharacterTokenizer, NamesDataset, create_dataset, create_dataloaders, get_device, set_seed

%load_ext autoreload
%autoreload 2  # Hot reload library changes
```

**Available Components**:
- `CharacterTokenizer` - Character-level tokenizer (attrs: `chars`, `vocab_size`; methods: `encode()`, `decode()`)
- `NamesDataset` - PyTorch Dataset (methods: `get_texts()`)
- `create_dataset(dataset_id, splits)` - Factory for data loading/splitting (supports "names", "words")
- `create_dataloaders(train_dataset, val_dataset, ...)` - DataLoader factory
- `get_device(prefer_cpu=False)` - Smart device detection (use `prefer_cpu=True` for Transformers)
- `set_seed(42)` - Set random seeds
- `collate_fn`, `count_parameters`, `print_model_summary` - Utilities

**Typical Pattern**:
```python
full_dataset, train_dataset, val_dataset = create_dataset("names", splits=[0.9, 0.1])
tokenizer = full_dataset.tokenizer
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=32)
```

**Common Mistakes**:
- Using non-existent methods (e.g., `tokenizer.get_vocab()` doesn't exist - use `tokenizer.chars`)
- Not checking API before use (check source in `src/aiml_notebooks/` or docs above)

### Testing Notebooks

**IMPORTANT**: Always test end-to-end after creation/modification.

```bash
# Standard execution
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb

# Quick test with shorter timeout
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 --inplace notebooks/your-notebook.ipynb

# macOS with MPS fallback (required for nn.Transformer)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb
```

### GPU Acceleration

**Device Selection**: Use the shared utility:

```python
from aiml_notebooks import get_device

device = get_device()                   # Standard (MPS > CUDA > CPU)
device = get_device(prefer_cpu=True)    # Safe mode for Transformers (CUDA > CPU, avoids MPS)
```

**MPS (macOS Metal) Notes**:
- `nn.Transformer` not fully supported on MPS
- Use `prefer_cpu=True` for Transformer notebooks OR set `PYTORCH_ENABLE_MPS_FALLBACK=1` before starting Python
- Environment variable must be set BEFORE Python starts (not in notebook with `os.environ`)

```bash
# Run Jupyter with MPS fallback
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter lab
```

## Common Tasks

```bash
# Add dependency
# Edit pyproject.toml, then:
uv sync

# Update dependencies
uv lock --upgrade && uv sync

# Run sweep
uv run python run_sweep.py sweeps/config.yaml notebooks/notebook.ipynb --count 10
```

## Hyperparameter Sweeps

Structure: `sweeps/*.yaml` (configs) → `run_sweep.py` (runner) → `tmp/sweeps/` (temp files, gitignored)

**Basic sweep config**:
```yaml
name: sweep-name
method: bayes  # or grid, random
project: wandb-project
metric:
  name: val_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
    distribution: log_uniform_values
  batch_size:
    values: [64, 128, 256]
```

## Git Workflow

- Main branch: `main`
- `.gitignore`: `.venv/`, `.ipynb_checkpoints`, `notebooks/tmp`, `notebooks/output`, `notebooks/wandb`, `notebooks/lightning_logs`, `.env`
- `uv.lock` is committed for reproducibility

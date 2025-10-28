# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of AI/ML Jupyter notebooks for learning and experimentation. The project contains both course recreations and standalone experiments exploring various AI/ML concepts.

## Development Environment

This project uses **uv** for fast, reliable Python dependency management (NOT conda or pip). The project includes a local `aiml_notebooks` package containing shared utilities (tokenizers, datasets, etc.) that are reused across multiple notebooks.

### Setup Commands

```bash
# Install dependencies (creates/syncs .venv)
uv sync

# Run Jupyter Lab
uv run jupyter lab
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
notebooks/          # All Jupyter notebooks (ACTIVE - work with these)
├── <prefix>-NNN-*.ipynb     # Numbered course recreation notebooks (e.g., zero2hero-001-*.ipynb)
├── wip-*.ipynb              # Work-in-progress experiments
├── *.ipynb                  # Completed standalone notebooks
└── *.todo.md                # Todo lists for specific WIP notebooks

_deprecated-notebooks/  # Deprecated/archived notebooks (IGNORE - do not use)

src/aiml_notebooks/ # Shared library for reusable components
├── __init__.py             # Package initialization
├── tokenizers.py           # Character-level tokenizers (reused across notebooks)
└── datasets.py             # PyTorch Dataset classes (reused across notebooks)

pyproject.toml      # Project dependencies (uv configuration)
uv.lock            # Locked dependencies for reproducibility
environment.yml     # Legacy conda config (not used; use uv instead)
```

**IMPORTANT**: Only notebooks in the `notebooks/` directory are active and should be considered for all tasks. Any notebooks in `_deprecated-notebooks/` or other deprecated directories should be completely ignored - they are not maintained and should not be referenced, edited, or used as examples.

## Notebook Conventions

### Naming Patterns

- `<prefix>-NNN-description.ipynb` - Numbered course recreation notebooks (e.g., `zero2hero-001-backprop.ipynb`)
- `wip-description.ipynb` - Work-in-progress experiments (incomplete/draft)
- `description.ipynb` - Completed standalone experiments

## Working with Notebooks

### Creating New Notebooks

**IMPORTANT**: When the user asks to create a new educational notebook, **ALWAYS use the `ml-notebook-educator` agent** via the Task tool. This specialized agent creates highly interactive, pedagogical notebooks with:
- Step-by-step learning progression (50-80 cells, 10-20 parts)
- Theory before practice for each concept
- Progressive visualizations and comparisons
- Reflection questions and experiments
- Clear learning objectives and practical guidance

**Standard workflow**:
1. User requests notebook on a topic
2. Use Task tool with `subagent_type="ml-notebook-educator"` and detailed prompt
3. Agent creates notebook in `notebooks/` directory with appropriate naming

**Manual notebook creation** (only if not using agent):
1. Place notebooks in the `notebooks/` directory
2. Use appropriate naming prefix (`wip-` for incomplete, `<prefix>-NNN-` for numbered course recreations)

### Editing Notebooks

- Notebooks are meant to be detailed with extensive explanations
- Include mathematical formulas, visualizations, and step-by-step commentary
- Use markdown cells liberally to explain concepts
- For WIP notebooks, consider creating a `.todo.md` file to track progress

### Testing Notebooks

**IMPORTANT**: Always test notebooks end-to-end after creation or modification.

```bash
# Test a notebook by executing all cells
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb

# Or use a shorter timeout for quick testing (e.g., with 1 epoch)
uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 --inplace notebooks/your-notebook.ipynb

# On macOS with MPS: Enable CPU fallback for unsupported operations
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb
```

**Important for macOS users**: If a notebook uses PyTorch Transformers (`nn.Transformer`) or other advanced operations, you MUST set `PYTORCH_ENABLE_MPS_FALLBACK=1` when testing. See [GPU Acceleration](#gpu-acceleration) section for details.

**When using shared library components**:
1. Verify the API by checking the source code in `src/aiml_notebooks/` or referring to the API documentation in CLAUDE.md
2. Use `%autoreload 2` to enable hot reloading during development
3. Test common operations (encode/decode, dataset access, etc.) in early notebook cells

**Common mistakes to avoid**:
- Using non-existent methods (e.g., `tokenizer.get_vocab()` doesn't exist - use `tokenizer.chars` instead)
- Assuming APIs without checking documentation

### Using the Shared Library

Notebooks can import reusable components from the `aiml_notebooks` package:

```python
# Import shared utilities
from aiml_notebooks import CharacterTokenizer, NamesDataset, collate_fn

# Enable autoreload for hot reloading of library changes
%load_ext autoreload
%autoreload 2
```

The `%autoreload 2` magic command ensures that any changes to library code in `src/aiml_notebooks/` are automatically reloaded without needing to restart the kernel. This is essential for iterative development.

**Available shared components**:

#### CharacterTokenizer
Character-level tokenizer for text sequences.

**Attributes**:
- `chars` - List of all characters in vocabulary (special token first)
- `char_to_idx` - Dict mapping characters to indices
- `idx_to_char` - Dict mapping indices to characters
- `special_token` - The special start/end token character (default '.')
- `vocab_size` - Total number of characters in vocabulary

**Methods**:
- `encode(text)` - Convert text to list of indices
- `decode(indices)` - Convert list of indices to text
- `encode_char(char)` - Convert single character to index
- `decode_char(idx)` - Convert single index to character
- `get_special_token_idx()` - Get index of special token
- `is_special_token(char)` - Check if character is special token

**Example**:
```python
tokenizer = full_dataset.tokenizer
vocab_chars = ''.join(tokenizer.chars)  # Get all vocabulary characters
encoded = tokenizer.encode('.hello.')   # [0, 8, 5, 12, 12, 15, 0]
decoded = tokenizer.decode(encoded)     # '.hello.'
```

#### NamesDataset
PyTorch Dataset for character-level name generation tasks.

**Attributes**:
- `names` - List of name strings
- `tokenizer` - CharacterTokenizer instance
- `max_length` - Maximum sequence length

**Methods**:
- `__len__()` - Returns number of names in dataset
- `__getitem__(idx)` - Returns (input_tensor, target_tensor) for training
- `get_texts()` - Returns raw list of names (generic interface for accessing underlying data)

**Example**:
```python
dataset = NamesDataset(names, tokenizer)
x, y = dataset[0]  # Get first training example
original_names = dataset.get_texts()  # Get raw names
```

#### Device and Seed Utilities

**get_device()** - Smart device detection with configurable strategy

**Parameters**:
- `verbose` (bool, default=True) - Whether to print device information
- `prefer_cpu` (bool, default=False) - If True, avoid MPS (safe mode for Transformers)
- `show_mps_warning` (bool, default=True) - Show fallback instructions when MPS is skipped

**Returns**: `torch.device` object

**Examples**:
```python
# Standard mode (MPS-compatible notebooks)
device = get_device()

# Safe mode (Transformer notebooks)
device = get_device(prefer_cpu=True)

# Quiet mode
device = get_device(verbose=False)
```

**set_seed()** - Set random seeds for reproducibility

**Parameters**:
- `seed` (int, default=42) - Random seed value

**Example**:
```python
from aiml_notebooks import set_seed

set_seed(42)  # Makes all random operations deterministic
```

#### Other Components
- `collate_fn(batch)` - Collate function for padding variable-length sequences in batches (used with DataLoader)
- `create_dataset(dataset_id, splits)` - Factory function for creating datasets with automatic data loading and splitting
- `create_dataloaders(train_dataset, val_dataset, ...)` - Factory function for creating DataLoaders with proper configuration
- `count_parameters(model)` - Count trainable parameters in a PyTorch model
- `print_model_summary(model)` - Print model architecture and parameter counts

**Factory Usage**:

The factories provide a clean API for data preparation:

```python
# 1. Create dataset with automatic data loading, tokenization, and splitting
full_dataset, train_dataset, val_dataset = create_dataset(
    dataset_id="names",  # Try "names" or "words"
    splits=[0.9, 0.1]  # 90% train, 10% validation
)

# 2. Extract tokenizer for later use (e.g., for generation)
tokenizer = full_dataset.tokenizer

# 3. Create data loaders with the factory
train_loader, val_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32
)

# 4. Access raw texts from dataset (generic interface)
original_texts = full_dataset.get_texts()  # Works for any dataset type
```

**Supported dataset IDs**:
- `"names"` - Character-level name generation dataset (~32K names)
- `"words"` - English words dataset (3-12 characters, filtered for generation, ~370K words)

**DataLoader factory options**:
```python
# With all options
train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=128,
    num_workers=4,
    shuffle_train=True  # default
)
```

### Notebook Philosophy

This repository emphasizes **interactive, pedagogical notebooks** designed for learning:
- **Granular**: 50-80 cells breaking down complex topics into digestible steps
- **Theory-first**: Explain concepts before implementing them
- **Visual**: Multiple visualizations showing progressive understanding
- **Interactive**: Reflection questions, experiments, and hands-on exploration
- **Self-contained**: Each notebook can be run independently from start to finish
- **Progressive**: Build complexity gradually from simple foundations

### GPU Acceleration

#### Device Selection in Notebooks

**Recommended Pattern: Use the Shared Utility**

The `aiml_notebooks` package provides a `get_device()` utility for consistent device detection across all notebooks:

```python
from aiml_notebooks import get_device

# For standard notebooks (MPS-compatible)
device = get_device()

# For Transformer notebooks (safe mode, avoids MPS)
device = get_device(prefer_cpu=True)

# Quiet mode (no output)
device = get_device(verbose=False)
```

**How it works:**
- **Standard mode** (`prefer_cpu=False`): Prefers MPS > CUDA > CPU
- **Safe mode** (`prefer_cpu=True`): Prefers CUDA > CPU (avoids MPS for Transformer compatibility)
- Automatically detects available hardware
- Provides helpful hints when MPS is available but not used

**When to use each mode:**
- **Standard mode**: For notebooks without `nn.Transformer` or nested tensor operations
- **Safe mode**: For notebooks using `nn.Transformer`, HuggingFace Transformers, or other MPS-incompatible operations

**Manual pattern (only if not using shared library):**

```python
import torch

# For MPS-compatible notebooks
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for GPU acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

# For Transformer notebooks (safe mode)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for GPU acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")
    if torch.backends.mps.is_available():
        print("Note: MPS is available but not used due to compatibility issues")
        print("To use MPS with CPU fallback, run: PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab")
```

#### Platform Notes

- **macOS**: PyTorch uses MPS (Metal Performance Shaders) for GPU acceleration
- **CUDA**: Only available on Linux/Windows (not macOS)
- Check device availability: `torch.backends.mps.is_available()` or `torch.cuda.is_available()`

#### MPS Compatibility Issues

**Known Issues:**
- `nn.Transformer` and nested tensor operations are not fully supported on MPS
- Some advanced PyTorch operations may fall back to CPU automatically

**Solution:**
Use the `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable to enable automatic CPU fallback for unsupported operations:

```bash
# When running Jupyter Lab
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter lab

# When testing notebooks
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter nbconvert --to notebook --execute notebooks/your-notebook.ipynb
```

**Important:** The environment variable MUST be set before starting Python. Setting it inside notebook code with `os.environ` does NOT work because PyTorch initializes before that code runs.

**When to Use CPU vs MPS:**
- **Use CPU (safer)**: For notebooks with `nn.Transformer`, nested tensors, or complex operations
- **Use MPS with fallback**: When you want best performance and don't mind occasional CPU fallbacks
- **Use MPS directly**: For simple operations (basic tensors, simple models) that are known to work

## Git Workflow

- Main branch: `main`
- `.gitignore` excludes: `.venv/`, `.ipynb_checkpoints`, `notebooks/tmp`, `notebooks/output`, `notebooks/wandb`, `notebooks/lightning_logs`, `.env`
- The `uv.lock` file is committed for reproducible installations

## Common Tasks

### Run a notebook
```bash
# Standard way
uv run jupyter lab notebooks/your-notebook.ipynb

# On macOS with MPS fallback enabled (for notebooks using Transformers)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run jupyter lab notebooks/your-notebook.ipynb
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

### Run hyperparameter sweeps with W&B
```bash
# Quick test sweep (3 trials)
uv run python sweep.py sweeps/your-config.yaml notebooks/your-notebook.ipynb

# Full Bayesian sweep (10 trials)
uv run python sweep.py sweeps/your-config.yaml notebooks/your-notebook.ipynb --count 10
```

## Hyperparameter Sweeps

The repository includes a generic W&B sweep runner (`run_sweep.py`) for optimizing notebook hyperparameters.

### Sweep Structure
```
sweeps/                          # Sweep configurations (YAML files, committed)
└── *.yaml                       # Sweep configs for various notebooks

tmp/sweeps/                      # Temporary sweep files (gitignored)
├── scripts/                     # Auto-converted notebooks
└── checkpoints/                 # Model checkpoints (if any)

run_sweep.py                     # Generic sweep runner
```

### Creating Sweeps

1. **Create sweep config**: Copy an existing `.yaml` from `sweeps/` and modify parameters
2. **Run sweep**: `uv run python run_sweep.py sweeps/your-config.yaml notebooks/your-notebook.ipynb`
3. **View results**: Check the W&B dashboard URL printed by the sweep runner

### Sweep Config Format

```yaml
name: my-sweep-name
method: bayes  # or 'grid', 'random'
project: wandb-project-name

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

  # ... more parameters
```

### Notes
- Converted scripts are automatically saved to `tmp/sweeps/scripts/` (gitignored)
- Sweeps run from the repository root
- Multiple agents can run in parallel for faster optimization

## Architecture Notes

- This project includes a local **aiml_notebooks** package (in `src/`) with shared utilities
- Notebooks can import from the shared library: `from aiml_notebooks import CharacterTokenizer, NamesDataset, collate_fn`
- Use `%autoreload 2` in notebooks to enable hot reloading of library changes
- Notebooks remain largely self-contained, but common components (tokenizers, datasets) are extracted to the library for reuse
- Data files and outputs should go in `notebooks/tmp`, `notebooks/output`, or similar (gitignored)
- Temporary sweep files go in `tmp/sweeps/` (gitignored)

### Shared Library Components

The `src/aiml_notebooks/` package contains:
- **tokenizers.py**: Character-level tokenizers (CharacterTokenizer)
- **datasets.py**: PyTorch Dataset classes (NamesDataset), utilities (collate_fn), and factories (create_dataset, create_dataloaders)

**Data preparation pipeline**:
1. `create_dataset` - Downloads/loads raw data, creates tokenizer, builds dataset, and splits
2. `create_dataloaders` - Wraps datasets in DataLoaders with proper batching and collate functions

These factories reduce boilerplate and ensure consistent data handling across all notebooks.

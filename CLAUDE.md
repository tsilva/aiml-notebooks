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

skills/                 # Task-specific instructions
└── *.skill.md          # Markdown files with step-by-step guides

src/aiml_notebooks/     # Shared library
├── tokenizers.py       # CharacterTokenizer
└── datasets.py         # Datasets, factories, utilities

TOC.md                  # Table of Contents (optimal learning order)
pyproject.toml          # Dependencies (uv config)
uv.lock                 # Locked dependencies
```

**IMPORTANT**: Only work with notebooks in `notebooks/` directory. Ignore deprecated directories.

## Skills Directory

The `skills/` folder contains task-specific instruction files (`.skill.md`). When performing a task, **check if a matching skill file exists** and follow those instructions.

**Usage**:
1. When starting a task, look for a skill file with a relevant name (e.g., `create-flashcard.skill.md` for flashcard creation)
2. If a matching skill file exists, read it and follow the step-by-step instructions
3. Skills provide standardized workflows and best practices for common tasks

## Notebook Conventions

**Naming**:
- `<prefix>-NNN-description.ipynb` - Numbered course recreations
- `wip-description.ipynb` - Work in progress
- `description.ipynb` - Completed standalone

## Working with Notebooks

### Creating/Editing

**IMPORTANT**: When creating notebooks, read and follow `skills/create-notebook.skill.md` for comprehensive guidance on structure, flow, and teaching principles.

**Quick reference** (see skill file for full details):
- Always markdown before code (even one line)
- Build intuition through small, incremental steps
- Theory before practice, progressive visualizations
- Self-contained and runnable end-to-end

Place notebooks in `notebooks/` with appropriate naming.

**IMPORTANT**: After creating or significantly modifying a notebook, you MUST update `TOC.md` to include it in the optimal learning order. See the "Maintaining the Table of Contents" section below for detailed instructions.

### Using Shared Library

**Available Components**: Check source files for latest API:
- `src/aiml_notebooks/tokenizers.py` - Tokenizer classes
- `src/aiml_notebooks/datasets.py` - Datasets, factories, utilities

**Setup in notebooks**:
```python
from aiml_notebooks import CharacterTokenizer, create_dataset, create_dataloaders, get_device, set_seed

%load_ext autoreload
%autoreload 2  # Hot reload library changes
```

**Typical usage pattern**:
```python
full_dataset, train_dataset, val_dataset = create_dataset("names", splits=[0.9, 0.1])
tokenizer = full_dataset.tokenizer
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=32)
```

**IMPORTANT**: Always check the source files in `src/aiml_notebooks/` for current API before using. Don't assume methods exist.

### Defensive Programming for Array Operations

**Why**: Notebook execution testing is expensive; shape/index errors may not appear until late in execution.

**Best practices**:
- Add comments documenting expected shapes for non-trivial operations
- Use assertions to validate intermediate shapes
- Verify computed indices are in valid range before array lookups
- Remember: `np.argmax()` on 2D arrays returns flattened indices unless `axis` is specified

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

## Maintaining the Table of Contents (TOC.md)

**CRITICAL**: The `TOC.md` file MUST be kept up to date whenever notebooks are created, modified, or deleted.

### Purpose of TOC.md
The TOC delineates the **optimal learning order** for a complete AI/ML n00b to make it to god-tier. This is the PRIMARY organizing principle - notebooks are ordered by conceptual dependencies, not alphabetically or by topic.

### When to Update TOC.md
1. **Creating a new notebook**: Analyze its conceptual prerequisites and insert it in the appropriate tier
2. **Modifying existing notebook content**: If changes significantly alter the difficulty or prerequisites, consider repositioning
3. **Completing WIP notebooks**: Update status and potentially move to more appropriate tier

### How to Update TOC.md

**Step 1: Analyze Prerequisites**
- What concepts must a learner understand before tackling this notebook?
- Check existing tier structure in `TOC.md` to understand progression

**Step 2: Determine Appropriate Tier**
- Review `TOC.md` to see current tier organization and where notebook fits
- Position based on conceptual prerequisites, not difficulty or topic

**Step 3: Insert with Description**
- Follow existing format in `TOC.md`
- Use ⭐ for notebooks foundational to multiple advanced topics
- Ensure description explains what the notebook teaches AND why it matters

**Maintenance Rules**:
- Insert based on conceptual prerequisites, not alphabetically or by topic
- May require repositioning if content changes significantly
- Keep tier structure reflecting clear learning progression

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

**Config format**: Check existing files in `sweeps/` for examples. Common methods: `bayes`, `grid`, `random`.

## Git Workflow

- Main branch: `main`
- Check `.gitignore` for ignored patterns (venvs, temp files, logs, etc.)
- `uv.lock` is committed for reproducibility

## Self-Reinforcement Learning Loop

**CRITICAL**: After completing tasks, update this file with high-value learnings that prevent recurring mistakes.

### When to Update CLAUDE.md

Add a tip ONLY if all of these are true:
1. **High frequency**: The mistake would happen in >90% of similar tasks
2. **Easily preventable**: A simple workflow change would avoid it
3. **Generic pattern**: Applies broadly, not specific to one file/function/library
4. **Process/approach**: About tool calling, environment understanding, or workflow - NOT specific code fixes

### What TO Add

Focus on workflow patterns and tool-calling strategies:
- "Always check X before doing Y to avoid Z"
- "Use Grep tool instead of assuming file locations when searching for..."
- "Verify assumption A by reading B before proceeding with C"
- "Don't assume X exists in environment - check with Y tool first"
- "When doing task type X, always start by reading Y to understand Z"

### What NOT to Add

Avoid these (debug on the fly instead):
- Specific API quirks for third-party libraries (will change)
- Programming language basics (should be known)
- One-off edge cases specific to single files
- Detailed code patterns (belongs in skill files or code comments)
- Fixes for temporary environment issues

### Format for New Tips

Add to the relevant section with clear context:
```markdown
**Pattern learned**: [One-line summary]
- Why: [Brief explanation of what went wrong]
- Fix: [Generic approach to prevent it]
```

**Example**:
```markdown
**Pattern learned**: Always read skill files before starting standardized tasks
- Why: Skill files contain detailed, tested workflows that prevent common mistakes
- Fix: Use Glob to find matching `*.skill.md` files, read relevant ones before proceeding
```

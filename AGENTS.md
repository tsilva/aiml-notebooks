# CLAUDE.md

AI/ML Jupyter notebooks for learning and experimentation. Shared utilities in `src/aiml_notebooks/`.

## Development Environment

**Package Manager**: Use `uv` (NOT conda or pip). Always use `uv run` to execute commands.

```bash
uv sync                    # Install/sync dependencies
uv run jupyter lab         # Run Jupyter Lab
uv lock --upgrade && uv sync  # Update dependencies
```

## Repository Structure

```
notebooks/              # ONLY work with active notebooks here
├── archive/            # Archived notebooks for reference only; do not edit
├── <prefix>-NNN-*.ipynb  # Numbered course recreations
├── wip-*.ipynb           # Work in progress
└── *.ipynb               # Completed standalone

skills/                 # Check for task-specific guides (*.skill.md)
src/aiml_notebooks/     # Shared library (always check source for API)
README.md               # Optimal learning order (MUST update after notebook changes)
```

## Notebook Workflow

### Before Starting
- Check `skills/` for matching `.skill.md` file (e.g., `create-notebook.skill.md`)
- If skill file exists, read and follow it

### Creating/Editing
**CRITICAL**: Follow `skills/create-notebook.skill.md`. Quick rules:
- **MANDATORY: Markdown cell before EVERY code cell** - Even if it's just a single line introducing what is going to happen. NO exceptions.
- Small incremental steps, theory before practice
- Self-contained and runnable end-to-end
- Place in `notebooks/` with proper naming:
  - `<prefix>-NNN-description.ipynb` (course recreations)
  - `wip-description.ipynb` (work in progress)
  - `description.ipynb` (completed standalone)

### After Creation/Modification
1. **Test end-to-end**: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb`
2. **Update README.md**: Insert based on conceptual prerequisites (not alphabetically)

## Shared Library Usage

Always check `src/aiml_notebooks/` source files for current API before using.

```python
from aiml_notebooks import CharacterTokenizer, create_dataset, create_dataloaders, get_device, set_seed

%load_ext autoreload
%autoreload 2  # Hot reload

# Typical pattern
full_dataset, train_dataset, val_dataset = create_dataset("names", splits=[0.9, 0.1])
tokenizer = full_dataset.tokenizer
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=32)

device = get_device()  # MPS > CUDA > CPU
device = get_device(prefer_cpu=True)  # For Transformers (CUDA > CPU, avoids MPS)
```

## Testing & GPU Notes

**Test command**:
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb
# Add --ExecutePreprocessor.timeout=300 for quick test
# Prefix PYTORCH_ENABLE_MPS_FALLBACK=1 for Transformer notebooks on macOS
```

**Defensive programming** (testing is expensive):
- Document expected shapes in comments
- Assert intermediate shapes
- Verify indices before array lookups
- `np.argmax()` on 2D arrays returns flattened indices unless `axis` specified

## README.md Maintenance

**Purpose**: Optimal learning order by conceptual prerequisites (not alphabetical/topical).

**When**: After creating/modifying notebooks or completing WIPs.

**How**:
1. Identify conceptual prerequisites
2. Insert in appropriate tier in `README.md`
3. Use ⭐ for foundational notebooks
4. Add description explaining what it teaches and why it matters

## Other Tasks

```bash
# Hyperparameter sweep
uv run python sweep.py path/to/config.yaml notebooks/notebook.ipynb --count 10
# Provide a W&B sweep config that matches your notebook parameters
```

**Git**: Main branch is `main`. `uv.lock` is committed for reproducibility.

## Self-Reinforcement

Update this file with high-frequency, easily preventable workflow patterns (NOT code-specific fixes).

**Add** generic workflow/tool-calling patterns that apply broadly:
- "Always check X before Y to avoid Z"
- "Read skill files before starting standardized tasks"

**Don't add** library quirks, one-off cases, or code patterns (belongs in skill files).

**Format**:
```markdown
**Pattern**: [Summary] - Why: [Explanation] - Fix: [Approach]
```

**Pattern**: Probe notebook execution early with `uv run` before investing in edits - Why: platform-specific lock or wheel issues can block validation late in the task - Fix: verify the runner up front and, if project resolution is broken, use `uv run --no-project --python <known-good-interpreter>` for notebook execution while keeping the repository unchanged.

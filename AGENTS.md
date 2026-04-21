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

### Paired Python Editing with Jupytext
Prefer Jupytext paired notebooks when editing `.ipynb` files. The repo config pairs notebooks
as `ipynb,py:percent`, so the `.py` file is the safe editing surface and the `.ipynb`
remains the runnable artifact.

For an existing notebook:
```bash
uv run jupytext --set-formats ipynb,py:percent notebooks/your-notebook.ipynb
```

Normal edit loop:
```bash
uv run jupytext --sync notebooks/your-notebook.ipynb
# edit notebooks/your-notebook.py
uv run jupytext --sync notebooks/your-notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/your-notebook.ipynb
```

Keep both paired files committed when a notebook is paired. Do not bulk-pair or sync archived
notebooks unless the task explicitly asks for it.

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

**Remote CUDA notebooks with Modal**:
Use Modal for notebooks that intentionally require CUDA and cannot be validated locally on
Apple Silicon or CPU-only machines.

```bash
./run_modal.sh notebooks/your-notebook.ipynb
```

This runs `scripts/run_modal_notebook.py` with `--gpu` and writes the executed notebook as
`notebooks/your-notebook.modal.ipynb` unless an explicit output path is provided. Prefer this
for CUDA-only notebooks such as Unsloth training demos instead of adding CPU/MPS fallbacks.

**Modal GPU types**:
Modal currently accepts these `gpu=` strings: `T4`, `L4`, `A10`, `L40S`, `A100`,
`A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100`, `H100!`, `H200`, `B200`, and
`B200+`. Append a count for multi-GPU containers, for example `gpu="H100:8"`.
`B200`, `H200`, `H100`, `A100`, `L4`, `T4`, and `L40S` support up to 8 GPUs per
container; `A10` supports up to 4. `H100` may auto-upgrade to `H200`; use `H100!`
to avoid that for benchmarking. `A100` may auto-upgrade from 40GB to 80GB; use
`A100-40GB` or `A100-80GB` when memory size must be fixed. The repo's Modal
runner accepts `--gpu-type`, defaults to `A10`, and keeps `A10G` as a backwards
compatible alias for Modal's `A10` class. Use `./run_modal.sh
notebooks/your-notebook.ipynb --gpu-type L40S` to request a different GPU. Check
the Modal GPU docs before changing long-lived runner defaults because available
GPU names and upgrade behavior can change.

Practical Modal GPU selection for this repo:
- Use `A10` for cheap CUDA smoke tests.
- Use `L40S` for actual Unsloth QLoRA fine-tuning runs.
- Use `A100-80GB` or `H100` only if `L40S` still hits memory limits or runtime is
  the main concern.

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

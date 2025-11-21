# Modal Progress Tracking - Required Changes

## Summary

To get **real-time progress logs in Modal**, you need:

1. ✅ **Modal Runner Changes** (Already done!)
   - Added `log_output=True` and `progress_bar=True` to papermill

2. **Notebook Changes** (Apply these)
   - Add `flush=True` to all print statements
   - Add progress callback for cloud environments
   - Optionally disable Lightning's progress bar (doesn't work in non-TTY)

---

## Changes to Apply to `karpathy-build-gpt.ipynb`

### Change 1: Add Progress Callback Cell (After imports)

**Location:** Add a new cell after the imports (around cell 4)

```python
import sys
from lightning.pytorch.callbacks import Callback

class CloudProgressCallback(Callback):
    """Progress callback for Modal/cloud environments with explicit flushing."""

    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.train_start_step = None

    def on_train_start(self, trainer, pl_module):
        print("\n" + "="*80, flush=True)
        print("🚀 TRAINING STARTED", flush=True)
        print("="*80, flush=True)
        print(f"Device: {pl_module.device}", flush=True)
        print(f"Max steps: {trainer.max_steps}", flush=True)
        print(f"Validation every: {trainer.val_check_interval} steps", flush=True)
        print("="*80 + "\n", flush=True)
        self.train_start_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            loss = outputs['loss'].item() if 'loss' in outputs else 'N/A'
            progress_pct = (trainer.global_step + 1) / trainer.max_steps * 100
            print(
                f"[Step {trainer.global_step + 1:5d}/{trainer.max_steps}] "
                f"({progress_pct:5.1f}%) | Loss: {loss:.4f}",
                flush=True
            )

    def on_validation_start(self, trainer, pl_module):
        print(f"\n{'─'*80}", flush=True)
        print(f"📊 VALIDATION (Step {trainer.global_step})", flush=True)
        print(f"{'─'*80}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss', None)
        val_perplexity = trainer.callback_metrics.get('val_perplexity', None)
        print(f"{'─'*80}", flush=True)
        if val_loss is not None:
            print(f"✓ Val Loss: {val_loss:.4f}", flush=True)
        if val_perplexity is not None:
            print(f"✓ Perplexity: {val_perplexity:.2f}", flush=True)
        print(f"{'─'*80}\n", flush=True)

    def on_train_end(self, trainer, pl_module):
        total_steps = trainer.global_step - (self.train_start_step or 0)
        print("\n" + "="*80, flush=True)
        print("🎉 TRAINING COMPLETED", flush=True)
        print("="*80, flush=True)
        print(f"Total steps: {total_steps}", flush=True)
        print("="*80 + "\n", flush=True)

print("CloudProgressCallback loaded")
```

### Change 2: Add Callback to Trainer Setup

**Location:** In the cell that creates the Trainer (around cell 26)

**Find this code:**
```python
# Build callbacks list
callbacks = [best_checkpoint_callback, periodic_checkpoint_callback, cleanup_callback]
if early_stopping_callback is not None:
    callbacks.append(early_stopping_callback)
```

**Change to:**
```python
# Build callbacks list
cloud_progress = CloudProgressCallback(log_every_n_steps=10)  # Log every 10 steps
callbacks = [
    best_checkpoint_callback,
    periodic_checkpoint_callback,
    cleanup_callback,
    cloud_progress  # <-- Add this for Modal progress tracking
]
if early_stopping_callback is not None:
    callbacks.append(early_stopping_callback)
```

### Change 3: Disable Lightning Progress Bar (Optional but Recommended)

**Location:** Same cell as above (Trainer creation)

**Find this code:**
```python
trainer = L.Trainer(
    ...
    enable_progress_bar=True,
    ...
)
```

**Change to:**
```python
trainer = L.Trainer(
    ...
    enable_progress_bar=False,  # Disable for Modal (doesn't work in non-TTY)
    ...
)
```

### Change 4: Add flush=True to Existing Prints (Optional but Recommended)

**Throughout the notebook**, change print statements from:
```python
print("Some message")
```

To:
```python
print("Some message", flush=True)
```

**Key locations:**
- Hardware detection output (cell 4)
- Dataset info (cells 6, 8, 10)
- Model initialization (cell 22)
- Training start message (cell 28)
- Any other print statements

---

## Expected Output in Modal Logs

After these changes, you'll see:

```
📓 Executing notebook: karpathy-build-gpt.ipynb
📊 GPU: A100-40GB
💾 Checkpoints: /checkpoints

[Cell 1/35] Executing...
[Cell 2/35] Executing...
...

================================================================================
🚀 TRAINING STARTED
================================================================================
Device: cuda:0
Max steps: 5000
Validation every: 100 steps
================================================================================

[Step    10/5000] (  0.2%) | Loss: 4.1234
[Step    20/5000] (  0.4%) | Loss: 3.9876
[Step    30/5000] (  0.6%) | Loss: 3.8432
...
[Step   100/5000] (  2.0%) | Loss: 2.3456

────────────────────────────────────────────────────────────────────────────────
📊 VALIDATION (Step 100)
────────────────────────────────────────────────────────────────────────────────
────────────────────────────────────────────────────────────────────────────────
✓ Val Loss: 2.1234
✓ Perplexity: 8.34
────────────────────────────────────────────────────────────────────────────────

[Step   110/5000] (  2.2%) | Loss: 2.2987
...
```

---

## Quick Apply Script

Run this to automatically apply the changes:

```bash
# TODO: Create automated patch script if needed
# For now, manually apply the changes above
```

---

## Why These Changes?

1. **`flush=True`**: Ensures output appears immediately in Modal logs, not buffered
2. **CloudProgressCallback**: Lightning's progress bars don't work in non-TTY environments
3. **Explicit step logging**: Shows training progress every N steps with loss values
4. **Validation logging**: Shows when validation starts/ends and metrics
5. **Modal runner changes**: Papermill now streams all outputs in real-time

---

## Testing Locally

To test if the changes work correctly:

```bash
# Run notebook with papermill locally (simulates Modal environment)
uv run papermill \
    notebooks/karpathy-build-gpt.ipynb \
    /tmp/test_output.ipynb \
    --log-output \
    --progress-bar

# You should see all the progress logs in real-time
```

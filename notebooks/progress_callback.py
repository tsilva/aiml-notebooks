"""
Progress callback for cloud/non-TTY environments like Modal.

Add this to your notebook to get detailed progress logging that shows up in logs.
"""

import sys
from lightning.pytorch.callbacks import Callback


class CloudProgressCallback(Callback):
    """
    Progress callback optimized for cloud/Modal environments.

    Prints detailed progress to stdout with explicit flushing,
    ensuring logs appear in real-time even in non-TTY environments.
    """

    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.train_start_step = None

    def on_train_start(self, trainer, pl_module):
        """Log training start."""
        print("\n" + "="*80, flush=True)
        print("🚀 TRAINING STARTED", flush=True)
        print("="*80, flush=True)
        print(f"Device: {pl_module.device}", flush=True)
        print(f"Max steps: {trainer.max_steps}", flush=True)
        print(f"Validation every: {trainer.val_check_interval} steps", flush=True)
        print("="*80 + "\n", flush=True)
        self.train_start_step = trainer.global_step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log progress every N steps."""
        if (trainer.global_step + 1) % self.log_every_n_steps == 0:
            # Get current loss
            loss = outputs['loss'].item() if 'loss' in outputs else 'N/A'

            # Calculate progress
            progress_pct = (trainer.global_step + 1) / trainer.max_steps * 100

            print(
                f"[Step {trainer.global_step + 1:5d}/{trainer.max_steps}] "
                f"({progress_pct:5.1f}%) | "
                f"Loss: {loss:.4f}",
                flush=True
            )

    def on_validation_start(self, trainer, pl_module):
        """Log validation start."""
        print(f"\n{'─'*80}", flush=True)
        print(f"📊 VALIDATION STARTING (Step {trainer.global_step})", flush=True)
        print(f"{'─'*80}", flush=True)

    def on_validation_end(self, trainer, pl_module):
        """Log validation results."""
        # Get validation metrics from callback_metrics
        val_loss = trainer.callback_metrics.get('val_loss', None)
        val_perplexity = trainer.callback_metrics.get('val_perplexity', None)

        print(f"{'─'*80}", flush=True)
        if val_loss is not None:
            print(f"✓ Validation Loss: {val_loss:.4f}", flush=True)
        if val_perplexity is not None:
            print(f"✓ Perplexity: {val_perplexity:.2f}", flush=True)
        print(f"{'─'*80}\n", flush=True)

    def on_train_end(self, trainer, pl_module):
        """Log training completion."""
        total_steps = trainer.global_step - (self.train_start_step or 0)
        print("\n" + "="*80, flush=True)
        print("🎉 TRAINING COMPLETED", flush=True)
        print("="*80, flush=True)
        print(f"Total steps trained: {total_steps}", flush=True)
        print(f"Final global step: {trainer.global_step}", flush=True)
        print("="*80 + "\n", flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start."""
        print(f"\n📅 Epoch {trainer.current_epoch} starting...", flush=True)


# Quick integration example:
"""
# In your notebook, add this callback:

from progress_callback import CloudProgressCallback

# Add to your callbacks list:
cloud_progress = CloudProgressCallback(log_every_n_steps=10)

callbacks = [
    best_checkpoint_callback,
    periodic_checkpoint_callback,
    cleanup_callback,
    cloud_progress,  # <-- Add this
]

if early_stopping_callback is not None:
    callbacks.append(early_stopping_callback)

# Also disable Lightning's built-in progress bar for cleaner logs:
trainer = L.Trainer(
    ...
    enable_progress_bar=False,  # <-- Change to False for Modal
    ...
)
"""

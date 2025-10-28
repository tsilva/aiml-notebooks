"""Training utilities for PyTorch Lightning workflows.

This module provides common utilities for setting up training experiments:
- W&B logger initialization with sweep support
- PyTorch Lightning trainer configuration
- Papermill parameter handling for hyperparameter sweeps
- Generic training loops for different task types
- TrainingHistory for metrics tracking
- Generic Trainer class for common training patterns
"""

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as L
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt


def create_wandb_logger(
    project: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    log_model: bool = True
) -> Optional[WandbLogger]:
    """Create a W&B logger, or return None if already in a W&B run.

    This handles the common pattern where notebooks can be run standalone
    (create new W&B run) or as part of a sweep (use existing run).

    Args:
        project: W&B project name
        run_name: Optional run name (None = auto-generated)
        config: Configuration dict to log
        log_model: Whether to log model checkpoints

    Returns:
        WandbLogger if new run created, None if using existing run

    Example:
        >>> wandb_logger = create_wandb_logger('my-project', config=CONFIG)
        >>> if wandb_logger:
        ...     print(f"Created run: {wandb_logger.experiment.name}")
        >>> else:
        ...     print(f"Using existing run: {wandb.run.name}")
    """
    if wandb.run is not None:
        print(f"Using existing W&B run: {wandb.run.name}")
        return None
    else:
        logger = WandbLogger(
            project=project,
            name=run_name,
            config=config,
            log_model=log_model
        )
        print(f"Created W&B run: {logger.experiment.name}")
        return logger


def watch_model(model, log: str = 'all', log_freq: int = 100):
    """Watch model gradients and parameters in W&B.

    Only logs if inside an active W&B run (safe to call always).

    Args:
        model: PyTorch model to watch
        log: What to log - 'gradients', 'parameters', 'all', or None
        log_freq: How often to log (in steps)

    Example:
        >>> model = MyModel()
        >>> watch_model(model)
    """
    if wandb.run is not None:
        wandb.watch(model, log=log, log_freq=log_freq)


def create_trainer(
    max_epochs: int,
    log_every_n_steps: int = 50,
    accelerator: str = 'auto',
    devices: int = 1,
    enable_progress_bar: bool = True,
    logger=None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_log_model: bool = True,
    model=None,
    watch_log: str = 'all',
    watch_log_freq: int = 100,
    enable_checkpointing: bool = True,
    checkpoint_monitor: str = 'val_loss',
    checkpoint_mode: str = 'min',
    **kwargs
) -> L.Trainer:
    """Create a PyTorch Lightning trainer with standard defaults and optional W&B integration.

    Args:
        max_epochs: Number of training epochs
        log_every_n_steps: How often to log metrics
        accelerator: Device accelerator ('auto', 'gpu', 'cpu', 'mps')
        devices: Number of devices to use
        enable_progress_bar: Whether to show progress bar
        logger: PyTorch Lightning logger (overrides W&B logger if provided)
        wandb_project: W&B project name (creates logger if provided)
        wandb_run_name: Optional W&B run name (None = auto-generated)
        wandb_config: Configuration dict to log to W&B
        wandb_log_model: Whether to log model checkpoints to W&B
        model: Model to watch with W&B (only if W&B logger created)
        watch_log: What to log - 'gradients', 'parameters', 'all', or None
        watch_log_freq: How often to log model gradients/params (in steps)
        enable_checkpointing: Whether to enable model checkpointing (default: True)
        checkpoint_monitor: Metric to monitor for checkpointing (default: 'val_loss')
        checkpoint_mode: 'min' or 'max' for the monitored metric (default: 'min')
        **kwargs: Additional trainer arguments

    Returns:
        Configured PyTorch Lightning Trainer

    Example:
        >>> # Simple usage with W&B integration
        >>> trainer = create_trainer(
        ...     max_epochs=100,
        ...     log_every_n_steps=20,
        ...     wandb_project='my-project',
        ...     wandb_config=CONFIG,
        ...     model=model
        ... )
        >>> trainer.fit(model, train_loader, val_loader)

        >>> # Advanced: bring your own logger
        >>> custom_logger = WandbLogger(project='my-project', ...)
        >>> trainer = create_trainer(
        ...     max_epochs=100,
        ...     logger=custom_logger
        ... )
    """
    # Create W&B logger if wandb_project provided and no logger given
    if logger is None and wandb_project is not None:
        logger = create_wandb_logger(
            project=wandb_project,
            run_name=wandb_run_name,
            config=wandb_config,
            log_model=wandb_log_model
        )

        # Watch model if provided and logger was created
        if logger is not None and model is not None:
            watch_model(model, log=watch_log, log_freq=watch_log_freq)

    # Setup callbacks
    callbacks = kwargs.pop('callbacks', [])

    # Add ModelCheckpoint callback if checkpointing is enabled
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
            save_top_k=1,
            save_last=False,
            verbose=False,
            filename=f'best-{{epoch:02d}}-{{{checkpoint_monitor}:.4f}}'
        )
        callbacks.append(checkpoint_callback)

    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        **kwargs
    )


def setup_papermill_params(config: Dict[str, Any], **params) -> Dict[str, Any]:
    """Update config with papermill parameters for hyperparameter sweeps.

    This is a common pattern where notebooks have:
    1. Base CONFIG dictionary with defaults
    2. Papermill cell with parameter variables
    3. CONFIG.update() to merge them

    This function simplifies that pattern.

    Args:
        config: Base configuration dictionary
        **params: Papermill parameters to override

    Returns:
        Updated configuration dictionary

    Example:
        >>> CONFIG = {'learning_rate': 1e-3, 'batch_size': 128}
        >>> # In papermill cell:
        >>> learning_rate = CONFIG['learning_rate']
        >>> batch_size = CONFIG['batch_size']
        >>> # Update config:
        >>> CONFIG = setup_papermill_params(CONFIG,
        ...     learning_rate=learning_rate,
        ...     batch_size=batch_size
        ... )
    """
    config.update(params)
    print(f"Running with config: {config}")
    return config


# ============================================================================
# Training and Evaluation Loops
# ============================================================================

def train_epoch_classification(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: Optional[float] = None,
    desc: str = "Training"
) -> Tuple[float, float]:
    """
    Standard training epoch for classification tasks.

    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        clip_grad_norm: Maximum gradient norm (None = no clipping)
        desc: Progress bar description

    Returns:
        avg_loss: Average loss over epoch
        accuracy: Training accuracy

    Example:
        >>> loss, acc = train_epoch_classification(
        ...     model, train_loader, optimizer, criterion, device
        ... )
        >>> print(f"Train Loss: {loss:.4f}, Acc: {acc:.4f}")
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc=desc):
        # Handle different batch formats
        if isinstance(batch, dict):
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
        else:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        # Statistics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating"
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate classification model.

    Args:
        model: PyTorch model
        loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        desc: Progress bar description

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
        predictions: All predictions
        labels: All true labels

    Example:
        >>> loss, acc, preds, labels = evaluate_classification(
        ...     model, val_loader, criterion, device
        ... )
        >>> print(f"Val Loss: {loss:.4f}, Acc: {acc:.4f}")
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            # Handle different batch formats
            if isinstance(batch, dict):
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Collect predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_epoch_seq2seq(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
    desc: str = "Training"
) -> float:
    """
    Training epoch for seq2seq tasks.

    Args:
        model: Seq2seq model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        clip_grad_norm: Maximum gradient norm
        teacher_forcing_ratio: Probability of using teacher forcing
        desc: Progress bar description

    Returns:
        avg_loss: Average loss over epoch

    Example:
        >>> loss = train_epoch_seq2seq(
        ...     model, train_loader, optimizer, criterion, device
        ... )
    """
    model.train()
    total_loss = 0

    for src, tgt in tqdm(loader, desc=desc):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()

        # Forward pass (model should handle teacher forcing internally)
        output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

        # Reshape for loss computation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip SOS token
        tgt = tgt[:, 1:].reshape(-1)  # Skip SOS token

        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_seq2seq(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating"
) -> float:
    """
    Evaluate seq2seq model.

    Args:
        model: Seq2seq model
        loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        desc: Progress bar description

    Returns:
        avg_loss: Average loss

    Example:
        >>> loss = evaluate_seq2seq(model, val_loader, criterion, device)
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(loader, desc=desc):
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass (no teacher forcing during evaluation)
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            # Reshape for loss computation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================================
# Training History
# ============================================================================

class TrainingHistory:
    """
    Track training metrics across epochs.

    Provides convenient storage and visualization of training metrics.

    Example:
        >>> history = TrainingHistory(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        >>> for epoch in range(num_epochs):
        ...     # ... training ...
        ...     history.update(train_loss=0.5, val_loss=0.6, train_acc=0.8, val_acc=0.75)
        >>> history.plot()
        >>> best_epoch = history.get_best_epoch('val_loss', mode='min')
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize training history.

        Args:
            metrics: List of metric names to track (e.g., ['train_loss', 'val_loss'])
                    If None, metrics will be inferred from first update() call
        """
        if metrics is None:
            self.metrics = []
            self.history = {}
        else:
            self.metrics = metrics
            self.history = {metric: [] for metric in metrics}
        self.epoch = 0

    def update(self, **kwargs):
        """
        Update history with new metric values.

        Args:
            **kwargs: Metric name-value pairs

        Example:
            >>> history.update(train_loss=0.5, val_loss=0.6)
        """
        # Initialize metrics on first call if not provided in __init__
        if not self.metrics:
            self.metrics = list(kwargs.keys())
            self.history = {metric: [] for metric in self.metrics}

        # Update values
        for key, value in kwargs.items():
            if key not in self.history:
                # Add new metric if not seen before
                self.metrics.append(key)
                self.history[key] = [None] * self.epoch  # Fill previous epochs with None
            self.history[key].append(value)

        self.epoch += 1

    def plot(self, figsize: Tuple[int, int] = (14, 5), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Plot training history.

        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style

        Example:
            >>> history.plot()
        """
        plt.style.use(style)

        # Group metrics by base name (train_loss and val_loss together)
        metric_groups = {}
        for metric in self.metrics:
            base_name = metric.replace('train_', '').replace('val_', '').replace('test_', '')
            if base_name not in metric_groups:
                metric_groups[base_name] = []
            metric_groups[base_name].append(metric)

        num_groups = len(metric_groups)
        fig, axes = plt.subplots(1, num_groups, figsize=figsize)

        if num_groups == 1:
            axes = [axes]

        for ax, (base_name, group_metrics) in zip(axes, metric_groups.items()):
            for metric in group_metrics:
                values = self.history[metric]
                label = metric.replace('_', ' ').title()
                ax.plot(values, marker='o', label=label, alpha=0.8)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(base_name.replace('_', ' ').title())
            ax.set_title(f'{base_name.replace("_", " ").title()} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_best_epoch(self, metric: str, mode: str = 'min') -> int:
        """
        Get epoch with best metric value.

        Args:
            metric: Metric name
            mode: 'min' or 'max'

        Returns:
            Epoch index with best value

        Example:
            >>> best_epoch = history.get_best_epoch('val_loss', mode='min')
            >>> print(f"Best model at epoch {best_epoch + 1}")
        """
        values = self.history[metric]
        if mode == 'min':
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))

    def get_best_value(self, metric: str, mode: str = 'min') -> float:
        """Get best metric value."""
        best_epoch = self.get_best_epoch(metric, mode)
        return self.history[metric][best_epoch]

    def __repr__(self):
        return f"TrainingHistory(epochs={self.epoch}, metrics={self.metrics})"

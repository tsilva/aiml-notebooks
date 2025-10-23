"""Training utilities for PyTorch Lightning workflows.

This module provides common utilities for setting up training experiments:
- W&B logger initialization with sweep support
- PyTorch Lightning trainer configuration
- Papermill parameter handling for hyperparameter sweeps
"""

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as L
from typing import Optional, Dict, Any


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
    **kwargs
) -> L.Trainer:
    """Create a PyTorch Lightning trainer with standard defaults.

    Args:
        max_epochs: Number of training epochs
        log_every_n_steps: How often to log metrics
        accelerator: Device accelerator ('auto', 'gpu', 'cpu', 'mps')
        devices: Number of devices to use
        enable_progress_bar: Whether to show progress bar
        logger: PyTorch Lightning logger (e.g., WandbLogger)
        **kwargs: Additional trainer arguments

    Returns:
        Configured PyTorch Lightning Trainer

    Example:
        >>> wandb_logger = create_wandb_logger('my-project', config=CONFIG)
        >>> trainer = create_trainer(
        ...     max_epochs=100,
        ...     log_every_n_steps=20,
        ...     logger=wandb_logger
        ... )
        >>> trainer.fit(model, train_loader, val_loader)
    """
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
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

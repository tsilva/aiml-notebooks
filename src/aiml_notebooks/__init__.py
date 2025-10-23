"""
aiml_notebooks - Shared utilities for AI/ML notebooks

This package contains reusable components for multiple notebooks:
- tokenizers: Character-level and other tokenizer implementations
- datasets: PyTorch Dataset classes for various tasks
- create_dataset: Factory function for creating datasets with automatic splits
- create_dataloaders: Factory function for creating DataLoaders
- training: PyTorch Lightning training utilities (W&B logger, trainer setup)
- visualization: Plotting and visualization utilities
"""

__version__ = "0.1.0"

from .tokenizers import CharacterTokenizer
from .datasets import NamesDataset, collate_fn, create_dataset, create_dataloaders
from .logging import log_gradients, log_model_weights, log_gradient_flow
from .training import create_wandb_logger, watch_model, create_trainer, setup_papermill_params
from .visualization import plot_image_grid, log_images_to_wandb, plot_training_curves

__all__ = [
    "CharacterTokenizer",
    "NamesDataset",
    "collate_fn",
    "create_dataset",
    "create_dataloaders",
    "log_gradients",
    "log_model_weights",
    "log_gradient_flow",
    "create_wandb_logger",
    "watch_model",
    "create_trainer",
    "setup_papermill_params",
    "plot_image_grid",
    "log_images_to_wandb",
    "plot_training_curves",
]

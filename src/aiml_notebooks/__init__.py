"""
aiml_notebooks - Shared utilities for AI/ML notebooks

This package contains reusable components for multiple notebooks:
- tokenizers: Character-level and other tokenizer implementations
- datasets: PyTorch Dataset classes for various tasks
- create_dataset: Factory function for creating datasets with automatic splits
- create_dataloaders: Factory function for creating DataLoaders
"""

__version__ = "0.1.0"

from .tokenizers import CharacterTokenizer
from .datasets import NamesDataset, collate_fn, create_dataset, create_dataloaders
from .logging import log_gradients, log_model_weights, log_gradient_flow

__all__ = [
    "CharacterTokenizer",
    "NamesDataset",
    "collate_fn",
    "create_dataset",
    "create_dataloaders",
    "log_gradients",
    "log_model_weights",
    "log_gradient_flow",
]

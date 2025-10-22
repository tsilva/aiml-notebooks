"""
aiml_notebooks - Shared utilities for AI/ML notebooks

This package contains reusable components for multiple notebooks:
- tokenizers: Character-level and other tokenizer implementations
- datasets: PyTorch Dataset classes for various tasks
- create_dataset: Factory function for creating datasets with automatic splits
"""

__version__ = "0.1.0"

from .tokenizers import CharacterTokenizer
from .datasets import NamesDataset, collate_fn, create_dataset

__all__ = [
    "CharacterTokenizer",
    "NamesDataset",
    "collate_fn",
    "create_dataset",
]

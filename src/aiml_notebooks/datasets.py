"""
PyTorch Dataset classes for various machine learning tasks.

This module provides reusable Dataset implementations for different types of data:
- NamesDataset: Character-level sequence dataset for name generation
- collate_fn: Utility function for padding variable-length sequences in batches
- create_dataset: Factory function for creating datasets with automatic splits
- create_dataloaders: Factory function for creating DataLoaders with proper configuration
"""

import urllib.request
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class NamesDataset(Dataset):
    """
    PyTorch Dataset for character-level name generation.

    This dataset converts a list of names into training examples where:
    - Input (x): sequence of characters starting with special token
    - Target (y): sequence shifted by one position (next character to predict)

    For example, with special_token='.', the name "alice" becomes:
    - Input:  ['.', 'a', 'l', 'i', 'c', 'e']
    - Target: ['a', 'l', 'i', 'c', 'e', '.']

    Args:
        names: List of name strings
        tokenizer: CharacterTokenizer instance for encoding/decoding
        max_length: Maximum sequence length (default: longest name + 1)

    Example:
        >>> from aiml_notebooks import CharacterTokenizer, NamesDataset
        >>> names = ['alice', 'bob', 'charlie']
        >>> tokenizer = CharacterTokenizer(names)
        >>> dataset = NamesDataset(names, tokenizer)
        >>> x, y = dataset[0]  # Get first training example
    """

    def __init__(self, names, tokenizer, max_length=None):
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length or max(len(n) for n in names) + 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        """
        Get a single training example.

        Args:
            idx: Index of the name to retrieve

        Returns:
            Tuple of (input_tensor, target_tensor) where:
            - input_tensor: Character indices for input sequence
            - target_tensor: Character indices for target sequence (shifted by 1)
        """
        name = self.names[idx]
        # Add start and end tokens (using tokenizer method to get special token)
        special_token_idx = self.tokenizer.get_special_token_idx()
        special_token_char = self.tokenizer.decode_char(special_token_idx)
        name_with_tokens = special_token_char + name + special_token_char
        indices = self.tokenizer.encode(name_with_tokens)

        # Create input (all chars except last) and target (all chars except first)
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        return x, y

    def get_texts(self) -> List[str]:
        """
        Get the raw text data from the dataset.

        This method provides a generic way to access the underlying text data
        without exposing internal implementation details.

        Returns:
            List of text strings (names in this case)
        """
        return self.names


def collate_fn(batch):
    """
    Collate function to pad variable-length sequences in a batch.

    This function is used by DataLoader to combine individual examples into batches.
    It pads sequences to the same length within each batch.

    Args:
        batch: List of (input, target) tuples from NamesDataset

    Returns:
        Tuple of (inputs_padded, targets_padded) where:
        - inputs_padded: Batched input sequences, padded with 0
        - targets_padded: Batched target sequences, padded with -100
          (CrossEntropyLoss ignores -100 padding values)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> from aiml_notebooks import collate_fn
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    xs, ys = zip(*batch)
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    return xs_padded, ys_padded


# Dataset loaders for different dataset IDs
def _load_names_dataset(url: Optional[str] = None) -> List[str]:
    """
    Load the names dataset from a URL or default source.

    Args:
        url: Optional URL to download names from. If None, uses default Karpathy names.txt

    Returns:
        List of name strings (lowercase, stripped)
    """
    if url is None:
        url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'

    # Download to temporary file
    urllib.request.urlretrieve(url, 'names.txt')

    # Load and process names
    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

    names = [name.strip().lower() for name in names if name.strip()]
    return names


def create_dataset(
    dataset_id: str,
    splits: Optional[Union[List[float], Tuple[float, ...]]] = None,
    tokenizer=None,
    **kwargs
) -> Tuple[Dataset, ...]:
    """
    Factory function to create datasets with automatic data loading and splitting.

    This function provides a clean API for creating datasets:
    1. Loads raw data based on dataset_id
    2. Creates or uses provided tokenizer
    3. Creates the full dataset
    4. Splits dataset according to specified proportions
    5. Returns tuple of (full_dataset, *split_datasets)

    Args:
        dataset_id: Identifier for the dataset type (e.g., "names")
        splits: Optional list/tuple of split proportions (must sum to 1.0)
                Example: [0.9, 0.1] for 90% train, 10% val
                If None, returns only the full dataset
        tokenizer: Optional pre-built tokenizer. If None, creates one from the data
        **kwargs: Additional keyword arguments passed to the dataset constructor
                  or data loader (e.g., url for custom data source)

    Returns:
        Tuple of datasets: (full_dataset, *split_datasets)
        - If splits is None: returns (full_dataset,)
        - If splits is [0.9, 0.1]: returns (full_dataset, train_dataset, val_dataset)

    Example:
        >>> # Load names dataset with 90/10 train/val split
        >>> full, train, val = create_dataset("names", splits=[0.9, 0.1])
        >>> print(f"Train: {len(train)}, Val: {len(val)}")

        >>> # Load names dataset with custom tokenizer
        >>> from aiml_notebooks import CharacterTokenizer
        >>> tokenizer = CharacterTokenizer(['alice', 'bob'], special_token='.')
        >>> full, train, val = create_dataset("names", splits=[0.9, 0.1], tokenizer=tokenizer)

        >>> # Load entire dataset without splits
        >>> full, = create_dataset("names")

    Supported dataset IDs:
        - "names": Character-level name generation dataset (Karpathy's names.txt)
    """
    # Validate splits if provided
    if splits is not None:
        if not isinstance(splits, (list, tuple)):
            raise ValueError("splits must be a list or tuple of floats")
        if abs(sum(splits) - 1.0) > 1e-6:
            raise ValueError(f"splits must sum to 1.0, got {sum(splits)}")

    # Load data based on dataset_id
    if dataset_id == "names":
        # Load raw names data
        url = kwargs.get('url', None)
        names = _load_names_dataset(url)

        # Create tokenizer if not provided
        if tokenizer is None:
            from .tokenizers import CharacterTokenizer
            tokenizer = CharacterTokenizer(names, special_token='.')

        # Create full dataset
        full_dataset = NamesDataset(names, tokenizer)

    else:
        raise ValueError(f"Unknown dataset_id: {dataset_id}. Supported: 'names'")

    # Return full dataset if no splits requested
    if splits is None:
        return (full_dataset,)

    # Create splits
    split_sizes = [int(len(full_dataset) * split) for split in splits]

    # Adjust last split to account for rounding
    split_sizes[-1] = len(full_dataset) - sum(split_sizes[:-1])

    # Create random splits
    split_datasets = random_split(full_dataset, split_sizes)

    # Return tuple: (full_dataset, *splits)
    return (full_dataset, *split_datasets)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    **kwargs
) -> Tuple[DataLoader, ...]:
    """
    Factory function to create DataLoaders from datasets.

    This function provides a clean API for creating dataloaders with proper
    collate functions and batching configurations.

    Args:
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for all dataloaders (default: 32)
        num_workers: Number of worker processes for data loading (default: 0)
        shuffle_train: Whether to shuffle training data (default: True)
        **kwargs: Additional keyword arguments passed to DataLoader
                  (e.g., pin_memory, drop_last)

    Returns:
        Tuple of DataLoaders: (train_loader, *optional_loaders)
        - If only train_dataset: returns (train_loader,)
        - If train + val: returns (train_loader, val_loader)
        - If train + val + test: returns (train_loader, val_loader, test_loader)

    Example:
        >>> # Create train and val loaders
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     batch_size=128
        ... )

        >>> # Create only train loader
        >>> train_loader, = create_dataloaders(
        ...     train_dataset=train_dataset,
        ...     batch_size=64
        ... )

        >>> # Create all three loaders
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     test_dataset=test_dataset,
        ...     batch_size=32,
        ...     num_workers=4
        ... )
    """
    loaders = []

    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        **kwargs
    )
    loaders.append(train_loader)

    # Create validation loader if dataset provided
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **kwargs
        )
        loaders.append(val_loader)

    # Create test loader if dataset provided
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **kwargs
        )
        loaders.append(test_loader)

    return tuple(loaders)

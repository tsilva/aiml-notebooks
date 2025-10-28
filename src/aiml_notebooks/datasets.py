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
from torchvision import datasets, transforms


# Class names for standard vision datasets
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

MNIST_CLASSES = [str(i) for i in range(10)]
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

FASHIONMNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)


def get_dataset_config(dataset_id: str) -> dict:
    """
    Get configuration metadata for a vision dataset.

    Returns a dictionary with dataset metadata including:
    - name: Human-readable dataset name
    - classes: List of class names
    - mean: Normalization mean for each channel
    - std: Normalization std for each channel
    - num_channels: Number of image channels (1 for grayscale, 3 for RGB)
    - image_size: Height/width of images (assumes square images)

    Args:
        dataset_id: Dataset identifier ('cifar10', 'mnist', 'fashionmnist')

    Returns:
        Dictionary with dataset configuration

    Raises:
        ValueError: If dataset_id is not recognized

    Example:
        >>> config = get_dataset_config('mnist')
        >>> print(config['name'])
        'MNIST'
        >>> print(config['num_channels'])
        1
        >>> print(config['image_size'])
        28

        >>> # Use in training
        >>> config = get_dataset_config('cifar10')
        >>> model = CNN(
        ...     num_classes=len(config['classes']),
        ...     in_channels=config['num_channels'],
        ...     input_size=config['image_size']
        ... )
    """
    configs = {
        'mnist': {
            'name': 'MNIST',
            'classes': MNIST_CLASSES,
            'mean': MNIST_MEAN,
            'std': MNIST_STD,
            'num_channels': 1,
            'image_size': 28,
        },
        'fashionmnist': {
            'name': 'Fashion-MNIST',
            'classes': FASHIONMNIST_CLASSES,
            'mean': FASHIONMNIST_MEAN,
            'std': FASHIONMNIST_STD,
            'num_channels': 1,
            'image_size': 28,
        },
        'cifar10': {
            'name': 'CIFAR-10',
            'classes': CIFAR10_CLASSES,
            'mean': CIFAR10_MEAN,
            'std': CIFAR10_STD,
            'num_channels': 3,
            'image_size': 32,
        },
    }

    dataset_id = dataset_id.lower()
    if dataset_id not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_id}. "
            f"Available datasets: {available}"
        )

    return configs[dataset_id]


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


def _load_words_dataset(url: Optional[str] = None) -> List[str]:
    """
    Load an English words dataset from a URL or default source.

    Args:
        url: Optional URL to download words from. If None, uses default word list

    Returns:
        List of word strings (lowercase, stripped)
    """
    if url is None:
        # Using a curated list of common English words
        url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt'

    # Download to temporary file
    urllib.request.urlretrieve(url, 'words.txt')

    # Load and process words
    with open('words.txt', 'r') as f:
        words = f.read().splitlines()

    # Filter: only words between 3-12 characters for reasonable generation
    words = [word.strip().lower() for word in words if word.strip()]
    words = [word for word in words if 3 <= len(word) <= 12 and word.isalpha()]
    return words


def _load_palindromes_dataset(min_length: int = 7, max_length: int = 15, num_samples: int = 10000) -> List[str]:
    """
    Generate synthetic palindromic sequences with long-range dependencies.

    This dataset is designed to expose RNN limitations with long-range dependencies.
    A palindrome requires the model to remember characters from the beginning when
    generating the end, creating dependencies that span the entire sequence.

    Format: "abcXcba" where X is optional middle character(s)
    Examples: "abccba", "abcdcba", "racecar"

    Args:
        min_length: Minimum palindrome length (default: 7)
        max_length: Maximum palindrome length (default: 15)
        num_samples: Number of palindromes to generate (default: 10000)

    Returns:
        List of palindromic strings
    """
    import random
    import string

    palindromes = []
    chars = string.ascii_lowercase[:15]  # Use subset for reasonable vocab size

    for _ in range(num_samples):
        # Random length for this palindrome
        length = random.randint(min_length, max_length)
        half_len = length // 2

        # Generate first half randomly
        first_half = ''.join(random.choices(chars, k=half_len))

        # Add middle character if odd length
        if length % 2 == 1:
            middle = random.choice(chars)
            palindrome = first_half + middle + first_half[::-1]
        else:
            palindrome = first_half + first_half[::-1]

        palindromes.append(palindrome)

    return palindromes


def _load_mnist_dataset(
    root: str = './data',
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> Dataset:
    """
    Load the MNIST handwritten digits dataset.

    Args:
        root: Root directory where dataset will be downloaded/stored (default: './data')
        train: If True, load training set; if False, load test set (default: True)
        download: If True, download dataset if not already present (default: True)
        transform: Optional custom transform. If None, uses ToTensor() (default: None)

    Returns:
        MNIST dataset with specified transforms
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    return datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )


def _load_cifar10_dataset(
    root: str = './data',
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> Dataset:
    """
    Load the CIFAR-10 dataset (32x32 color images, 10 classes).

    Args:
        root: Root directory where dataset will be downloaded/stored (default: './data')
        train: If True, load training set; if False, load test set (default: True)
        download: If True, download dataset if not already present (default: True)
        transform: Optional custom transform. If None, uses ToTensor() (default: None)

    Returns:
        CIFAR-10 dataset with specified transforms
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    return datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )


def _load_fashionmnist_dataset(
    root: str = './data',
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None
) -> Dataset:
    """
    Load the Fashion-MNIST dataset (28x28 grayscale clothing images, 10 classes).

    Args:
        root: Root directory where dataset will be downloaded/stored (default: './data')
        train: If True, load training set; if False, load test set (default: True)
        download: If True, download dataset if not already present (default: True)
        transform: Optional custom transform. If None, uses ToTensor() (default: None)

    Returns:
        Fashion-MNIST dataset with specified transforms
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    return datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )


def _load_bitflipping_dataset(min_length: int = 5, max_length: int = 10, num_samples: int = 10000) -> List[str]:
    """
    Generate synthetic bit-flipping sequences for transformation learning.

    This dataset tests the model's ability to learn a simple transformation:
    given a binary sequence, flip all bits (0→1, 1→0). The format is:
    "input_sequence>output_sequence" where output is the bitwise NOT of input.

    Format: "01011>10100" (input > flipped_output)
    Examples:
        - "0101>1010" (4-bit flip)
        - "11000>00111" (5-bit flip)
        - "0110101>1001010" (7-bit flip)

    Args:
        min_length: Minimum bit sequence length (default: 5)
        max_length: Maximum bit sequence length (default: 10)
        num_samples: Number of sequences to generate (default: 10000)

    Returns:
        List of bitflipping strings in format "input>output"
    """
    import random

    sequences = []

    for _ in range(num_samples):
        # Random length for this sequence
        length = random.randint(min_length, max_length)

        # Generate random binary sequence
        input_bits = ''.join(random.choices('01', k=length))

        # Flip all bits (0→1, 1→0)
        output_bits = ''.join('1' if bit == '0' else '0' for bit in input_bits)

        # Format: "input>output"
        sequence = f"{input_bits}>{output_bits}"
        sequences.append(sequence)

    return sequences


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
        - "words": English words dataset (3-12 characters, filtered for generation)
        - "palindromes": Synthetic palindromic sequences (7-15 chars, long-range dependencies)
        - "bitflipping": Synthetic bit-flipping sequences (5-10 bits, transformation learning)
        - "mnist": MNIST handwritten digits dataset (28x28 grayscale, 10 classes)
        - "cifar10": CIFAR-10 dataset (32x32 color, 10 classes)
        - "fashionmnist": Fashion-MNIST dataset (28x28 grayscale, 10 clothing classes)
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
        texts = _load_names_dataset(url)

        # Create tokenizer if not provided
        if tokenizer is None:
            from .tokenizers import CharacterTokenizer
            tokenizer = CharacterTokenizer(texts, special_token='.')

        # Create full dataset
        full_dataset = NamesDataset(texts, tokenizer)

    elif dataset_id == "words":
        # Load raw words data
        url = kwargs.get('url', None)
        texts = _load_words_dataset(url)

        # Create tokenizer if not provided
        if tokenizer is None:
            from .tokenizers import CharacterTokenizer
            tokenizer = CharacterTokenizer(texts, special_token='.')

        # Create full dataset (reusing NamesDataset - it works for any text!)
        full_dataset = NamesDataset(texts, tokenizer)

    elif dataset_id == "palindromes":
        # Generate palindromes data with long-range dependencies
        min_length = kwargs.get('min_length', 7)
        max_length = kwargs.get('max_length', 15)
        num_samples = kwargs.get('num_samples', 10000)
        texts = _load_palindromes_dataset(min_length, max_length, num_samples)

        # Create tokenizer if not provided
        if tokenizer is None:
            from .tokenizers import CharacterTokenizer
            tokenizer = CharacterTokenizer(texts, special_token='.')

        # Create full dataset (reusing NamesDataset - it works for any text!)
        full_dataset = NamesDataset(texts, tokenizer)

    elif dataset_id == "bitflipping":
        # Generate bitflipping data for transformation learning
        min_length = kwargs.get('min_length', 5)
        max_length = kwargs.get('max_length', 10)
        num_samples = kwargs.get('num_samples', 10000)
        texts = _load_bitflipping_dataset(min_length, max_length, num_samples)

        # Create tokenizer if not provided
        if tokenizer is None:
            from .tokenizers import CharacterTokenizer
            tokenizer = CharacterTokenizer(texts, special_token='.')

        # Create full dataset (reusing NamesDataset - it works for any text!)
        full_dataset = NamesDataset(texts, tokenizer)

    elif dataset_id in ["mnist", "cifar10", "fashionmnist"]:
        # Load vision dataset (MNIST, CIFAR-10, Fashion-MNIST)
        root = kwargs.get('root', './data')
        download = kwargs.get('download', True)
        train_transform = kwargs.get('train_transform', None)
        test_transform = kwargs.get('test_transform', None)

        # Map dataset_id to loader function
        loaders = {
            "mnist": _load_mnist_dataset,
            "cifar10": _load_cifar10_dataset,
            "fashionmnist": _load_fashionmnist_dataset,
        }
        loader_fn = loaders[dataset_id]

        # Load both train and test sets with appropriate transforms
        train_dataset = loader_fn(root=root, train=True, download=download, transform=train_transform)
        test_dataset = loader_fn(root=root, train=False, download=download, transform=test_transform)

        # For vision datasets, return train and test separately (not combined)
        # This preserves the standard train/test split and allows different transforms
        if splits is None:
            # Return standard train/test split as tuple: (train, test)
            return (train_dataset, test_dataset)
        else:
            # If custom splits requested, combine and re-split
            from torch.utils.data import ConcatDataset
            full_dataset = ConcatDataset([train_dataset, test_dataset])

            # Create splits
            split_sizes = [int(len(full_dataset) * split) for split in splits]
            split_sizes[-1] = len(full_dataset) - sum(split_sizes[:-1])
            split_datasets = random_split(full_dataset, split_sizes)

            return (full_dataset, *split_datasets)

    else:
        raise ValueError(f"Unknown dataset_id: {dataset_id}. Supported: 'names', 'words', 'palindromes', 'bitflipping', 'mnist', 'cifar10', 'fashionmnist'")

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
    use_collate_fn: bool = True,
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
        use_collate_fn: Whether to use custom collate_fn for padding (default: True)
                        Set to False for vision datasets that don't need padding
        **kwargs: Additional keyword arguments passed to DataLoader
                  (e.g., pin_memory, drop_last)

    Returns:
        Tuple of DataLoaders: (train_loader, *optional_loaders)
        - If only train_dataset: returns (train_loader,)
        - If train + val: returns (train_loader, val_loader)
        - If train + val + test: returns (train_loader, val_loader, test_loader)

    Example:
        >>> # Create train and val loaders for sequence data (with padding)
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     batch_size=128
        ... )

        >>> # Create loaders for vision data (no padding needed)
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_dataset=train_dataset,
        ...     val_dataset=val_dataset,
        ...     batch_size=128,
        ...     use_collate_fn=False
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
    import torch

    loaders = []

    # Disable pin_memory on MPS (not supported)
    if 'pin_memory' in kwargs and kwargs['pin_memory']:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            kwargs['pin_memory'] = False

    # Determine collate_fn to use
    collate = collate_fn if use_collate_fn else None

    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate,
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
            collate_fn=collate,
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
            collate_fn=collate,
            num_workers=num_workers,
            **kwargs
        )
        loaders.append(test_loader)

    return tuple(loaders)


# ============================================================================
# Collate Function Factories
# ============================================================================

def create_seq2seq_collate_fn(pad_idx: int = 0):
    """
    Create collate function for seq2seq tasks with variable-length sequences.

    This factory function creates a collate_fn that pads sequences to the same
    length within each batch, which is required for efficient batch processing.

    Args:
        pad_idx: Padding index to use (default: 0)

    Returns:
        Collate function suitable for DataLoader

    Example:
        >>> collate_fn = create_seq2seq_collate_fn(pad_idx=0)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for src, tgt in loader:
        ...     # src and tgt are padded to same length in batch
        ...     print(src.shape, tgt.shape)
    """
    def collate_fn(batch):
        """Collate function for seq2seq with padding."""
        # Separate source and target sequences
        src_batch, tgt_batch = zip(*batch)

        # Pad sequences to max length in batch
        src_padded = nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=pad_idx
        )
        tgt_padded = nn.utils.rnn.pad_sequence(
            tgt_batch, batch_first=True, padding_value=pad_idx
        )

        return src_padded, tgt_padded

    return collate_fn


def create_classification_collate_fn(dict_format: bool = True):
    """
    Create collate function for classification tasks.

    Args:
        dict_format: If True, return dict with 'input_ids' and 'label' keys
                    If False, return tuple of (inputs, labels)

    Returns:
        Collate function suitable for DataLoader

    Example:
        >>> # Dictionary format (default)
        >>> collate_fn = create_classification_collate_fn(dict_format=True)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for batch in loader:
        ...     inputs = batch['input_ids']
        ...     labels = batch['label']

        >>> # Tuple format
        >>> collate_fn = create_classification_collate_fn(dict_format=False)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for inputs, labels in loader:
        ...     pass
    """
    if dict_format:
        def collate_fn(batch):
            """Collate function returning dictionary."""
            input_ids = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            return {'input_ids': input_ids, 'label': labels}
    else:
        def collate_fn(batch):
            """Collate function returning tuple."""
            if isinstance(batch[0], dict):
                input_ids = torch.stack([item['input_ids'] for item in batch])
                labels = torch.stack([item['label'] for item in batch])
            else:
                input_ids = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
            return input_ids, labels

    return collate_fn


def create_variable_length_collate_fn(
    pad_idx: int = 0,
    return_lengths: bool = False,
    sort_by_length: bool = False
):
    """
    Create collate function for variable-length sequences with optional features.

    Args:
        pad_idx: Padding index
        return_lengths: If True, also return sequence lengths
        sort_by_length: If True, sort batch by sequence length (descending)
                       Useful for packed sequences in RNNs

    Returns:
        Collate function suitable for DataLoader

    Example:
        >>> collate_fn = create_variable_length_collate_fn(
        ...     pad_idx=0, return_lengths=True, sort_by_length=True
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        >>> for batch in loader:
        ...     sequences, labels, lengths = batch
        ...     # Use lengths for packed sequences
        ...     packed = nn.utils.rnn.pack_padded_sequence(
        ...         sequences, lengths, batch_first=True
        ...     )
    """
    def collate_fn(batch):
        """Collate variable-length sequences."""
        sequences, labels = zip(*batch)

        # Get original lengths
        lengths = torch.tensor([len(seq) for seq in sequences])

        # Sort by length if requested
        if sort_by_length:
            lengths, sort_idx = lengths.sort(descending=True)
            sequences = [sequences[i] for i in sort_idx]
            labels = [labels[i] for i in sort_idx]

        # Pad sequences
        sequences_padded = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_idx
        )
        labels_tensor = torch.tensor(labels)

        if return_lengths:
            return sequences_padded, labels_tensor, lengths
        else:
            return sequences_padded, labels_tensor

    return collate_fn

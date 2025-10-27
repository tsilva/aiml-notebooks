"""
General utility functions for notebooks.

Common helper functions used across multiple notebooks.
"""

import torch
import numpy as np
import random


def get_device(
    verbose: bool = True,
    prefer_cpu: bool = False,
    show_mps_warning: bool = True
) -> torch.device:
    """
    Get best available device with configurable strategy.

    Supports two device selection strategies:
    1. Standard (prefer_cpu=False): MPS > CUDA > CPU
    2. Safe mode (prefer_cpu=True): CUDA > CPU (avoids MPS)

    Safe mode is recommended for notebooks using:
    - nn.Transformer or nested tensor operations
    - Advanced PyTorch operations with known MPS issues

    For MPS-compatible notebooks, use standard mode for best performance.

    Args:
        verbose: Whether to print device information
        prefer_cpu: If True, avoid MPS even if available (safe mode for Transformers)
        show_mps_warning: If True and MPS is available but not used, show fallback instructions

    Returns:
        torch.device object

    Examples:
        >>> # Standard usage (MPS-compatible notebooks)
        >>> device = get_device()
        Using MPS (Metal Performance Shaders) for GPU acceleration

        >>> # Safe mode (Transformer notebooks)
        >>> device = get_device(prefer_cpu=True)
        Using CPU
        Note: MPS is available but not used due to compatibility issues
        To use MPS with CPU fallback, run: PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab

        >>> # Quiet mode
        >>> device = get_device(verbose=False)
    """
    if not prefer_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
            if show_mps_warning and prefer_cpu and torch.backends.mps.is_available():
                print("Note: MPS is available but not used due to compatibility issues")
                print("To use MPS with CPU fallback, run: PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab")

    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters

    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model

    Example:
        >>> print_model_summary(model)
        Model: MyModel
        Total parameters: 1,234,567
        Trainable parameters: 1,234,567
        Non-trainable parameters: 0
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

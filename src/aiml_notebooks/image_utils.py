"""
Image utility functions for normalization, denormalization, and transforms.

This module provides utilities for working with images in PyTorch:
- Normalization/denormalization using dataset statistics
- Standard transform creation
- Visualization helpers
"""

import torch
import torchvision.transforms as T
from typing import Optional, Tuple, List, Union
import numpy as np


def normalize_image(
    image: torch.Tensor,
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]]
) -> torch.Tensor:
    """
    Normalize image using mean and std.

    Applies: normalized = (image - mean) / std

    Args:
        image: Input image tensor [C, H, W] or [B, C, H, W]
        mean: Mean for each channel (single value or per-channel)
        std: Standard deviation for each channel

    Returns:
        Normalized image tensor

    Example:
        >>> from aiml_notebooks.datasets import CIFAR10_MEAN, CIFAR10_STD
        >>> normalized = normalize_image(image, CIFAR10_MEAN, CIFAR10_STD)
    """
    # Convert to tensors if needed
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=image.dtype, device=image.device)

    # Reshape for broadcasting
    if image.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif image.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

    return (image - mean) / std


def denormalize_image(
    image: torch.Tensor,
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]],
    clip: bool = True
) -> torch.Tensor:
    """
    Denormalize image to original scale.

    Reverses normalization: denormalized = image * std + mean

    Args:
        image: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        clip: Whether to clip values to [0, 1] range

    Returns:
        Denormalized image tensor

    Example:
        >>> from aiml_notebooks.datasets import CIFAR10_MEAN, CIFAR10_STD
        >>> denormalized = denormalize_image(normalized_img, CIFAR10_MEAN, CIFAR10_STD)
        >>> # Now ready for visualization
    """
    # Convert to tensors if needed
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=image.dtype, device=image.device)

    # Reshape for broadcasting
    if image.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif image.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

    denorm = image * std + mean

    if clip:
        denorm = torch.clamp(denorm, 0.0, 1.0)

    return denorm


def create_standard_transforms(
    image_size: Optional[int] = None,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
    augment: bool = True,
    grayscale: bool = False
) -> Tuple[T.Compose, T.Compose]:
    """
    Create standard train and test transforms for image datasets.

    Train transforms include augmentation (if enabled):
    - Random crop and resize
    - Random horizontal flip
    - Color jitter
    - Normalization

    Test transforms are simple:
    - Resize (if image_size provided)
    - To tensor
    - Normalization

    Args:
        image_size: Target image size (None = no resizing)
        mean: Mean for normalization (None = no normalization)
        std: Std for normalization (None = no normalization)
        augment: Whether to apply data augmentation for training
        grayscale: Whether images are grayscale (affects normalization)

    Returns:
        train_transform, test_transform: Tuple of torchvision transforms

    Example:
        >>> from aiml_notebooks.datasets import CIFAR10_MEAN, CIFAR10_STD
        >>> train_tf, test_tf = create_standard_transforms(
        ...     image_size=32,
        ...     mean=CIFAR10_MEAN,
        ...     std=CIFAR10_STD,
        ...     augment=True
        ... )
        >>> train_dataset = CIFAR10(root='./data', train=True, transform=train_tf)
    """
    train_transforms = []
    test_transforms = []

    # Training augmentation
    if augment and image_size is not None:
        train_transforms.extend([
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(),
        ])

        if not grayscale:
            train_transforms.append(
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )
    elif image_size is not None:
        train_transforms.append(T.Resize(image_size))

    # Test transforms (no augmentation)
    if image_size is not None:
        test_transforms.append(T.Resize(image_size))

    # Convert to tensor (both train and test)
    train_transforms.append(T.ToTensor())
    test_transforms.append(T.ToTensor())

    # Normalization (both train and test)
    if mean is not None and std is not None:
        normalize = T.Normalize(mean=mean, std=std)
        train_transforms.append(normalize)
        test_transforms.append(normalize)

    return T.Compose(train_transforms), T.Compose(test_transforms)


def create_denoising_transforms(
    noise_type: str = 'gaussian',
    noise_std: float = 0.1,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None
) -> T.Compose:
    """
    Create transforms for image denoising tasks.

    Returns a transform that:
    1. Converts to tensor
    2. Normalizes (if mean/std provided)
    3. Adds noise (returns both clean and noisy)

    Args:
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
        noise_std: Noise level parameter
        mean: Mean for normalization
        std: Std for normalization

    Returns:
        Transform that returns (noisy_image, clean_image) tuple

    Example:
        >>> from aiml_notebooks.datasets import MNIST_MEAN, MNIST_STD
        >>> transform = create_denoising_transforms(
        ...     noise_type='gaussian',
        ...     noise_std=0.2,
        ...     mean=MNIST_MEAN,
        ...     std=MNIST_STD
        ... )
    """
    from .augmentation import ImageNoiser

    def denoising_transform(image):
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = T.ToTensor()(image)

        # Normalize
        if mean is not None and std is not None:
            image = T.Normalize(mean=mean, std=std)(image)

        # Keep clean copy
        clean_image = image.clone()

        # Add noise
        if noise_type == 'gaussian':
            noisy_image = ImageNoiser.add_gaussian_noise(image, std=noise_std)
        elif noise_type == 'salt_pepper':
            noisy_image = ImageNoiser.add_salt_pepper_noise(image, prob=noise_std)
        elif noise_type == 'speckle':
            noisy_image = ImageNoiser.add_speckle_noise(image, std=noise_std)
        else:
            noisy_image = image

        return noisy_image, clean_image

    return denoising_transform


def prepare_for_visualization(
    images: torch.Tensor,
    mean: Optional[Union[float, Tuple[float, ...]]] = None,
    std: Optional[Union[float, Tuple[float, ...]]] = None,
    denormalize: bool = True
) -> np.ndarray:
    """
    Prepare images for visualization with matplotlib/PIL.

    Handles:
    - Denormalization (if needed)
    - Moving to CPU
    - Converting to numpy
    - Transposing to [H, W, C] format
    - Clipping to [0, 1]

    Args:
        images: Image tensor [B, C, H, W], [C, H, W], or [H, W]
        mean: Mean used for normalization (None = no denormalization)
        std: Std used for normalization (None = no denormalization)
        denormalize: Whether to apply denormalization

    Returns:
        Numpy array ready for visualization
        - If batch: [B, H, W, C] or [B, H, W] for grayscale
        - If single: [H, W, C] or [H, W] for grayscale

    Example:
        >>> from aiml_notebooks.datasets import CIFAR10_MEAN, CIFAR10_STD
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Visualize normalized images
        >>> vis_images = prepare_for_visualization(
        ...     batch_images,
        ...     mean=CIFAR10_MEAN,
        ...     std=CIFAR10_STD
        ... )
        >>> plt.imshow(vis_images[0])
    """
    # Denormalize if needed
    if denormalize and mean is not None and std is not None:
        images = denormalize_image(images, mean, std, clip=True)

    # Move to CPU and convert to numpy
    images_np = images.detach().cpu().numpy()

    # Clip to [0, 1]
    images_np = np.clip(images_np, 0.0, 1.0)

    # Transpose to [H, W, C] format
    if images_np.ndim == 4:  # [B, C, H, W]
        images_np = np.transpose(images_np, (0, 2, 3, 1))
        # Remove channel dim if grayscale
        if images_np.shape[-1] == 1:
            images_np = images_np.squeeze(-1)
    elif images_np.ndim == 3:  # [C, H, W]
        images_np = np.transpose(images_np, (1, 2, 0))
        # Remove channel dim if grayscale
        if images_np.shape[-1] == 1:
            images_np = images_np.squeeze(-1)

    return images_np


def batch_normalize(
    images: torch.Tensor,
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]]
) -> torch.Tensor:
    """
    Normalize a batch of images (convenience wrapper).

    Args:
        images: Batch of images [B, C, H, W]
        mean: Mean for each channel
        std: Standard deviation for each channel

    Returns:
        Normalized batch

    Example:
        >>> from aiml_notebooks.datasets import MNIST_MEAN, MNIST_STD
        >>> normalized_batch = batch_normalize(images, MNIST_MEAN, MNIST_STD)
    """
    return normalize_image(images, mean, std)


def batch_denormalize(
    images: torch.Tensor,
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]],
    clip: bool = True
) -> torch.Tensor:
    """
    Denormalize a batch of images (convenience wrapper).

    Args:
        images: Batch of normalized images [B, C, H, W]
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        clip: Whether to clip to [0, 1]

    Returns:
        Denormalized batch

    Example:
        >>> from aiml_notebooks.datasets import MNIST_MEAN, MNIST_STD
        >>> denormalized_batch = batch_denormalize(
        ...     normalized_images,
        ...     MNIST_MEAN,
        ...     MNIST_STD
        ... )
    """
    return denormalize_image(images, mean, std, clip)


def get_dataset_stats(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Get normalization statistics for common datasets.

    Args:
        dataset_name: Name of dataset ('cifar10', 'mnist', 'fashionmnist')

    Returns:
        (mean, std): Tuple of normalization statistics

    Raises:
        ValueError: If dataset_name is not recognized

    Example:
        >>> mean, std = get_dataset_stats('cifar10')
        >>> train_transform = T.Compose([
        ...     T.ToTensor(),
        ...     T.Normalize(mean=mean, std=std)
        ... ])
    """
    from . import datasets

    stats_map = {
        'cifar10': (datasets.CIFAR10_MEAN, datasets.CIFAR10_STD),
        'mnist': (datasets.MNIST_MEAN, datasets.MNIST_STD),
        'fashionmnist': (datasets.FASHIONMNIST_MEAN, datasets.FASHIONMNIST_STD),
        'fashion_mnist': (datasets.FASHIONMNIST_MEAN, datasets.FASHIONMNIST_STD),
    }

    dataset_name = dataset_name.lower()
    if dataset_name not in stats_map:
        available = ', '.join(stats_map.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )

    return stats_map[dataset_name]

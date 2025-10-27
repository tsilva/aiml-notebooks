"""
Data augmentation utilities for text and images.

This module provides noise injection and augmentation techniques for:
- Text: character-level noise (substitution, insertion, deletion, transposition)
- Images: Gaussian noise, salt-and-pepper noise, blur
"""

import torch
import numpy as np
import random
from typing import Optional, List
import string


class TextNoiser:
    """
    Apply various types of character-level noise to text for augmentation.

    Useful for:
    - Training denoising models
    - Data augmentation for text classification
    - Robustness testing

    Example:
        >>> noiser = TextNoiser(sub_prob=0.1, ins_prob=0.1, del_prob=0.1)
        >>> noisy_text = noiser.add_noise("hello world")
        >>> print(noisy_text)  # e.g., "helli world" (char substituted)
    """

    def __init__(
        self,
        sub_prob: float = 0.1,
        ins_prob: float = 0.1,
        del_prob: float = 0.1,
        swap_prob: float = 0.05,
        char_set: Optional[str] = None
    ):
        """
        Initialize text noiser.

        Args:
            sub_prob: Probability of substituting a character
            ins_prob: Probability of inserting a character
            del_prob: Probability of deleting a character
            swap_prob: Probability of swapping adjacent characters
            char_set: Characters to use for substitution/insertion
                     (None = lowercase letters + space)
        """
        self.sub_prob = sub_prob
        self.ins_prob = ins_prob
        self.del_prob = del_prob
        self.swap_prob = swap_prob

        if char_set is None:
            self.char_set = string.ascii_lowercase + ' '
        else:
            self.char_set = char_set

    def add_noise(self, text: str) -> str:
        """
        Apply random noise to text.

        Args:
            text: Input text

        Returns:
            Noisy text

        Example:
            >>> noiser = TextNoiser(sub_prob=0.2)
            >>> noiser.add_noise("hello")
            'hxllo'  # 'e' substituted with 'x'
        """
        chars = list(text)
        i = 0

        while i < len(chars):
            # Character substitution
            if random.random() < self.sub_prob:
                chars[i] = random.choice(self.char_set)

            # Character insertion
            if random.random() < self.ins_prob:
                chars.insert(i, random.choice(self.char_set))
                i += 1

            # Character deletion
            if random.random() < self.del_prob and len(chars) > 1:
                chars.pop(i)
                continue

            # Character transposition (swap with next)
            if random.random() < self.swap_prob and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 1

            i += 1

        return ''.join(chars)

    def add_noise_batch(self, texts: List[str]) -> List[str]:
        """
        Apply noise to a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of noisy texts

        Example:
            >>> noiser = TextNoiser()
            >>> noiser.add_noise_batch(["hello", "world"])
            ['heLlo', 'wopld']
        """
        return [self.add_noise(text) for text in texts]

    def __call__(self, text: str) -> str:
        """Allow using noiser as a callable."""
        return self.add_noise(text)


class ImageNoiser:
    """
    Apply various types of noise to images.

    Supports:
    - Gaussian noise
    - Salt-and-pepper noise
    - Speckle noise

    Example:
        >>> noiser = ImageNoiser()
        >>> noisy_images = noiser.add_gaussian_noise(clean_images, std=0.1)
    """

    @staticmethod
    def add_gaussian_noise(
        images: torch.Tensor,
        mean: float = 0.0,
        std: float = 0.1,
        clip: bool = True
    ) -> torch.Tensor:
        """
        Add Gaussian (normal) noise to images.

        Args:
            images: Input images [batch, channels, height, width] or [channels, height, width]
            mean: Mean of Gaussian distribution
            std: Standard deviation of Gaussian distribution
            clip: Whether to clip values to [0, 1] range

        Returns:
            Noisy images (same shape as input)

        Example:
            >>> clean_images = torch.rand(10, 3, 64, 64)
            >>> noisy = ImageNoiser.add_gaussian_noise(clean_images, std=0.2)
        """
        noise = torch.randn_like(images) * std + mean
        noisy_images = images + noise

        if clip:
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

        return noisy_images

    @staticmethod
    def add_salt_pepper_noise(
        images: torch.Tensor,
        prob: float = 0.05,
        salt_vs_pepper: float = 0.5
    ) -> torch.Tensor:
        """
        Add salt-and-pepper noise to images.

        Salt = white pixels (1.0)
        Pepper = black pixels (0.0)

        Args:
            images: Input images [batch, channels, height, width] or [channels, height, width]
            prob: Total probability of noise (salt + pepper)
            salt_vs_pepper: Ratio of salt to total noise (0.5 = equal salt and pepper)

        Returns:
            Noisy images (same shape as input)

        Example:
            >>> clean_images = torch.rand(10, 1, 28, 28)
            >>> noisy = ImageNoiser.add_salt_pepper_noise(clean_images, prob=0.1)
        """
        noisy_images = images.clone()

        # Add salt (white pixels)
        salt_mask = torch.rand_like(images) < (prob * salt_vs_pepper)
        noisy_images[salt_mask] = 1.0

        # Add pepper (black pixels)
        pepper_mask = torch.rand_like(images) < (prob * (1 - salt_vs_pepper))
        noisy_images[pepper_mask] = 0.0

        return noisy_images

    @staticmethod
    def add_speckle_noise(
        images: torch.Tensor,
        std: float = 0.1,
        clip: bool = True
    ) -> torch.Tensor:
        """
        Add speckle (multiplicative) noise to images.

        Speckle noise: noisy = image + image * noise
        Common in radar/ultrasound imaging.

        Args:
            images: Input images
            std: Standard deviation of noise
            clip: Whether to clip to [0, 1]

        Returns:
            Noisy images

        Example:
            >>> clean = torch.rand(5, 3, 32, 32)
            >>> noisy = ImageNoiser.add_speckle_noise(clean, std=0.2)
        """
        noise = torch.randn_like(images) * std
        noisy_images = images + images * noise

        if clip:
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

        return noisy_images

    @staticmethod
    def add_uniform_noise(
        images: torch.Tensor,
        low: float = -0.1,
        high: float = 0.1,
        clip: bool = True
    ) -> torch.Tensor:
        """
        Add uniform noise to images.

        Args:
            images: Input images
            low: Lower bound of uniform distribution
            high: Upper bound of uniform distribution
            clip: Whether to clip to [0, 1]

        Returns:
            Noisy images

        Example:
            >>> clean = torch.rand(10, 1, 28, 28)
            >>> noisy = ImageNoiser.add_uniform_noise(clean, low=-0.05, high=0.05)
        """
        noise = torch.rand_like(images) * (high - low) + low
        noisy_images = images + noise

        if clip:
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

        return noisy_images


def add_gaussian_noise_numpy(
    images: np.ndarray,
    mean: float = 0.0,
    std: float = 0.1,
    clip: bool = True
) -> np.ndarray:
    """
    Add Gaussian noise to numpy images.

    Args:
        images: Input images as numpy array
        mean: Mean of Gaussian
        std: Standard deviation
        clip: Whether to clip to [0, 1]

    Returns:
        Noisy images as numpy array

    Example:
        >>> import numpy as np
        >>> clean = np.random.rand(10, 28, 28, 1)
        >>> noisy = add_gaussian_noise_numpy(clean, std=0.2)
    """
    noise = np.random.randn(*images.shape) * std + mean
    noisy_images = images + noise

    if clip:
        noisy_images = np.clip(noisy_images, 0.0, 1.0)

    return noisy_images


def random_dropout_pixels(
    images: torch.Tensor,
    prob: float = 0.1,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Randomly drop out (zero out) pixels.

    Similar to dropout in neural networks but applied to input images.

    Args:
        images: Input images
        prob: Probability of dropping each pixel
        fill_value: Value to use for dropped pixels

    Returns:
        Images with random pixels dropped

    Example:
        >>> images = torch.rand(5, 3, 32, 32)
        >>> dropped = random_dropout_pixels(images, prob=0.2)
    """
    mask = torch.rand_like(images) > prob
    return images * mask + fill_value * (~mask)


def add_random_occlusion(
    images: torch.Tensor,
    num_patches: int = 1,
    patch_size: int = 8,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Add random occlusion patches to images.

    Useful for testing model robustness.

    Args:
        images: Input images [batch, channels, height, width]
        num_patches: Number of occlusion patches per image
        patch_size: Size of square occlusion patch
        fill_value: Value to fill occluded regions

    Returns:
        Images with random occlusions

    Example:
        >>> images = torch.rand(10, 3, 64, 64)
        >>> occluded = add_random_occlusion(images, num_patches=2, patch_size=16)
    """
    batch_size, channels, height, width = images.shape
    occluded = images.clone()

    for b in range(batch_size):
        for _ in range(num_patches):
            # Random position
            y = random.randint(0, height - patch_size)
            x = random.randint(0, width - patch_size)

            # Occlude patch
            occluded[b, :, y:y+patch_size, x:x+patch_size] = fill_value

    return occluded


class RandomNoise:
    """
    Callable augmentation that randomly applies one of several noise types.

    Useful for torchvision.transforms.Compose pipelines.

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.ToTensor(),
        ...     RandomNoise(noise_types=['gaussian', 'salt_pepper'], p=0.5)
        ... ])
    """

    def __init__(
        self,
        noise_types: List[str] = ['gaussian'],
        p: float = 0.5,
        gaussian_std: float = 0.1,
        salt_pepper_prob: float = 0.05
    ):
        """
        Initialize random noise augmentation.

        Args:
            noise_types: List of noise types to randomly choose from
                        Options: 'gaussian', 'salt_pepper', 'speckle', 'uniform'
            p: Probability of applying noise
            gaussian_std: Std for Gaussian noise
            salt_pepper_prob: Probability for salt-and-pepper noise
        """
        self.noise_types = noise_types
        self.p = p
        self.gaussian_std = gaussian_std
        self.salt_pepper_prob = salt_pepper_prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random noise to image."""
        if random.random() > self.p:
            return image

        noise_type = random.choice(self.noise_types)

        if noise_type == 'gaussian':
            return ImageNoiser.add_gaussian_noise(image, std=self.gaussian_std)
        elif noise_type == 'salt_pepper':
            return ImageNoiser.add_salt_pepper_noise(image, prob=self.salt_pepper_prob)
        elif noise_type == 'speckle':
            return ImageNoiser.add_speckle_noise(image, std=self.gaussian_std)
        elif noise_type == 'uniform':
            return ImageNoiser.add_uniform_noise(image)
        else:
            return image

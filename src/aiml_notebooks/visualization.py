"""Visualization utilities for ML experiments.

This module provides common plotting and visualization utilities:
- Image grid plotting (for GANs, VAEs, etc.)
- W&B logging helpers
- Common matplotlib configurations
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Optional, List, Union
import torch


def plot_image_grid(
    images: Union[torch.Tensor, np.ndarray],
    nrows: int = 3,
    ncols: int = 16,
    titles: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    figsize: Optional[tuple] = None,
    cmap: str = 'gray',
    title: Optional[str] = None
) -> plt.Figure:
    """Plot a grid of images.

    Args:
        images: Images to plot, shape (num_images, H, W) or (num_images, C, H, W)
                If 4D with C channels, will be converted appropriately
        nrows: Number of rows in grid
        ncols: Number of columns in grid
        titles: Optional list of titles for each image
        row_labels: Optional list of labels for each row (shown on left)
        col_labels: Optional list of labels for each column (shown on top)
        figsize: Figure size (width, height). If None, auto-calculated
        cmap: Colormap for grayscale images
        title: Overall figure title

    Returns:
        Matplotlib figure object

    Example:
        >>> # Plot VAE reconstructions
        >>> fig = plot_image_grid(
        ...     images=torch.cat([originals, reconstructions, generated]),
        ...     nrows=3,
        ...     ncols=16,
        ...     row_labels=['Original', 'Reconstructed', 'Generated'],
        ...     col_labels=[f'Sample {i}' for i in range(16)]
        ... )
        >>> plt.show()
    """
    # Convert to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # Handle channel dimension: (N, C, H, W) -> (N, H, W, C)
    if images.ndim == 4:
        if images.shape[1] in [1, 3]:  # (N, C, H, W)
            images = np.transpose(images, (0, 2, 3, 1))
        if images.shape[-1] == 1:  # (N, H, W, 1) -> (N, H, W)
            images = images.squeeze(-1)

    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (ncols * 1, nrows * 1)

    # Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is 2D array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot images
    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            ax = axes[row, col]

            if idx < len(images):
                img = images[idx]
                # Handle RGB vs grayscale
                if img.ndim == 3 and img.shape[-1] == 3:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap=cmap)

                # Add individual image title
                if titles and idx < len(titles):
                    ax.set_title(titles[idx], fontsize=8)

            ax.axis('off')

            # Add row label on first column
            if col == 0 and row_labels and row < len(row_labels):
                ax.set_ylabel(row_labels[row], fontsize=10, rotation=0,
                            ha='right', va='center', labelpad=20)

            # Add column label on first row
            if row == 0 and col_labels and col < len(col_labels):
                ax.set_title(col_labels[col], fontsize=8, pad=5)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    plt.tight_layout()
    return fig


def log_images_to_wandb(
    images: Union[torch.Tensor, np.ndarray, plt.Figure],
    name: str,
    caption: Optional[str] = None
):
    """Log images to W&B (only if inside an active run).

    Args:
        images: Images to log (tensor, array, or matplotlib figure)
        name: Name for the W&B log entry
        caption: Optional caption for the images

    Example:
        >>> fig = plot_image_grid(samples, nrows=3, ncols=16)
        >>> log_images_to_wandb(fig, 'generated_samples')
    """
    if wandb.run is None:
        return

    if isinstance(images, plt.Figure):
        wandb.log({name: wandb.Image(images, caption=caption)})
    else:
        wandb.log({name: wandb.Image(images, caption=caption)})


def plot_training_curves(
    metrics: dict,
    title: str = "Training Curves",
    figsize: tuple = (12, 4)
) -> plt.Figure:
    """Plot training and validation metrics over time.

    Args:
        metrics: Dictionary with keys like 'train_loss', 'val_loss', etc.
                 Values should be lists of metric values per epoch/step
        title: Figure title
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object

    Example:
        >>> metrics = {
        ...     'train_loss': [0.5, 0.4, 0.3],
        ...     'val_loss': [0.6, 0.5, 0.4],
        ... }
        >>> fig = plot_training_curves(metrics)
        >>> plt.show()
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xlabel('Step/Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    return fig

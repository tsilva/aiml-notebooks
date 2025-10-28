"""Visualization utilities for ML experiments.

This module provides common plotting and visualization utilities:
- Image grid plotting (for GANs, VAEs, etc.)
- W&B logging helpers
- Common matplotlib configurations
- Training curves, confusion matrices, reconstructions
- Sample predictions and model comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Optional, List, Union, Dict, Tuple
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


def imshow_normalized(
    ax: plt.Axes,
    img_tensor: Union[torch.Tensor, np.ndarray],
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]],
    title: Optional[str] = None,
    **kwargs
):
    """
    Display a normalized image tensor on a matplotlib axis.

    Args:
        ax: Matplotlib axis
        img_tensor: Image tensor in CHW format
        mean: Normalization mean (single value for grayscale, tuple/list for RGB)
        std: Normalization std (single value for grayscale, tuple/list for RGB)
        title: Optional title for the image
        **kwargs: Additional arguments passed to ax.imshow()

    Example:
        >>> fig, ax = plt.subplots()
        >>> imshow_normalized(
        ...     ax, normalized_img,
        ...     mean=(0.1307,), std=(0.3081,),
        ...     title='Digit 5'
        ... )
        >>> plt.show()
    """
    from .image_utils import prepare_for_visualization

    # Convert to torch tensor if numpy
    if isinstance(img_tensor, np.ndarray):
        img_tensor = torch.from_numpy(img_tensor)

    # Prepare for visualization (denormalize and convert to HW or HWC format)
    img = prepare_for_visualization(img_tensor, mean, std, denormalize=True)

    # Auto-detect grayscale
    cmap = 'gray' if len(img.shape) == 2 else None

    ax.imshow(img, cmap=cmap, **kwargs)
    if title:
        ax.set_title(title)
    ax.axis('off')


def show_image_grid_normalized(
    images: Union[torch.Tensor, np.ndarray],
    mean: Union[float, Tuple[float, ...], List[float]],
    std: Union[float, Tuple[float, ...], List[float]],
    labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    class_names: Optional[List[str]] = None,
    predictions: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    confidences: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    nrows: int = 2,
    ncols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Display a grid of normalized images with optional labels and predictions.

    This is a high-level convenience function that combines denormalization
    with grid display. Supports both dataset samples and prediction visualization.

    Args:
        images: Batch of normalized images in NCHW format
        mean: Normalization mean (single value for grayscale, tuple/list for RGB)
        std: Normalization std (single value for grayscale, tuple/list for RGB)
        labels: Optional true labels for each image
        class_names: Optional list of class names (used with labels/predictions)
        predictions: Optional predicted labels (enables color-coded display)
        confidences: Optional prediction confidences (shown as percentages)
        nrows: Number of rows in grid
        ncols: Number of columns in grid
        figsize: Figure size (width, height). If None, auto-calculated
        title: Optional overall figure title

    Returns:
        Matplotlib figure object

    Example:
        >>> # Display dataset samples with labels
        >>> show_image_grid_normalized(
        ...     images=batch_images,
        ...     mean=(0.1307,), std=(0.3081,),
        ...     labels=batch_labels,
        ...     class_names=['0', '1', '2', ...],
        ...     nrows=2, ncols=4
        ... )

        >>> # Display predictions (green=correct, red=incorrect)
        >>> show_image_grid_normalized(
        ...     images=test_images,
        ...     mean=(0.4914, 0.4822, 0.4465),
        ...     std=(0.2470, 0.2435, 0.2616),
        ...     labels=true_labels,
        ...     predictions=pred_labels,
        ...     confidences=pred_confidences,
        ...     class_names=CIFAR10_CLASSES,
        ...     nrows=4, ncols=4
        ... )
    """
    # Convert tensors to numpy/lists
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().cpu().numpy()

    # Auto-calculate figsize
    if figsize is None:
        figsize = (ncols * 3, nrows * 3)

    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            ax.axis('off')
            continue

        # Build title
        img_title = None
        title_color = None

        if predictions is not None:
            # Prediction mode: show true vs predicted with color coding
            correct = (predictions[i] == labels[i]) if labels is not None else False
            title_color = 'green' if correct else 'red'

            if labels is not None and class_names:
                img_title = f"True: {class_names[labels[i]]}\n"
            elif labels is not None:
                img_title = f"True: {labels[i]}\n"
            else:
                img_title = ""

            if class_names:
                img_title += f"Pred: {class_names[predictions[i]]}"
            else:
                img_title += f"Pred: {predictions[i]}"

            if confidences is not None:
                img_title += f" ({confidences[i]*100:.1f}%)"

        elif labels is not None:
            # Label mode: show just the true label
            if class_names:
                img_title = class_names[labels[i]]
            else:
                img_title = str(labels[i])

        # Display image with denormalization
        imshow_normalized(ax, images[i], mean, std, title=img_title)

        # Apply color to title if needed
        if title_color is not None and img_title:
            ax.set_title(img_title, color=title_color, fontsize=10)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    elif predictions is not None:
        fig.suptitle('Predictions (Green=Correct, Red=Incorrect)',
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


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


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: tuple = (10, 8),
    cmap: str = 'Blues',
    title: str = 'Confusion Matrix',
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix with proper formatting.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names for labels
        normalize: Whether to normalize by row (true labels)
        figsize: Figure size (width, height)
        cmap: Color map
        title: Plot title
        show: Whether to call plt.show() (set to False for W&B logging)

    Returns:
        Matplotlib figure object

    Example:
        >>> # Simple usage
        >>> plot_confusion_matrix(
        ...     y_true=test_labels,
        ...     y_pred=predictions,
        ...     class_names=['Cat', 'Dog'],
        ...     normalize=True
        ... )

        >>> # With W&B logging
        >>> fig = plot_confusion_matrix(
        ...     y_true=test_labels,
        ...     y_pred=predictions,
        ...     class_names=['Cat', 'Dog'],
        ...     show=False
        ... )
        >>> wandb.log({'confusion_matrix': wandb.Image(fig)})
    """
    # Compute confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        cm_display = (cm * 100).astype(int)  # For display as percentages
    else:
        fmt = 'd'
        cm_display = cm

    # Create plot
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt if not normalize else 'd',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    # Print statistics
    print(f"\n{title} Statistics:")
    print("=" * 50)
    if class_names:
        for i, name in enumerate(class_names):
            if normalize:
                accuracy = cm[i, i]
                print(f"  {name:15s}: {accuracy*100:5.2f}% correct")
            else:
                correct = cm[i, i]
                total = cm[i].sum()
                print(f"  {name:15s}: {correct:5d} / {total:5d} ({100*correct/total:.2f}%)")
    print("=" * 50)

    return fig


def visualize_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n_samples: int = 10,
    figsize: Optional[tuple] = None,
    titles: Optional[List[str]] = None,
    cmap: str = 'gray'
):
    """
    Visualize original and reconstructed images side by side.

    Args:
        originals: Original images [batch_size, channels, height, width]
        reconstructions: Reconstructed images [batch_size, channels, height, width]
        n_samples: Number of samples to display
        figsize: Figure size (None = auto)
        titles: Optional titles for each column ['Original', 'Reconstructed']
        cmap: Color map ('gray' for grayscale, None for RGB)

    Example:
        >>> visualize_reconstructions(
        ...     originals=test_images,
        ...     reconstructions=model(test_images),
        ...     n_samples=8
        ... )
    """
    if figsize is None:
        figsize = (n_samples * 1.5, 3)

    if titles is None:
        titles = ['Original', 'Reconstructed']

    # Convert to numpy and denormalize if needed
    originals = originals.detach().cpu()
    reconstructions = reconstructions.detach().cpu()

    fig, axes = plt.subplots(2, n_samples, figsize=figsize)

    for i in range(n_samples):
        # Original
        img_orig = originals[i]
        if img_orig.shape[0] == 1:  # Grayscale
            axes[0, i].imshow(img_orig.squeeze(), cmap=cmap)
        else:  # RGB
            axes[0, i].imshow(img_orig.permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(titles[0], fontsize=12, fontweight='bold', loc='left')

        # Reconstruction
        img_recon = reconstructions[i]
        if img_recon.shape[0] == 1:  # Grayscale
            axes[1, i].imshow(img_recon.squeeze(), cmap=cmap)
        else:  # RGB
            axes[1, i].imshow(img_recon.permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(titles[1], fontsize=12, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.show()


def visualize_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: Optional[List[str]] = None,
    n_samples: int = 16,
    figsize: Optional[tuple] = None,
    confidences: Optional[torch.Tensor] = None,
    cmap: str = 'gray'
):
    """
    Visualize sample predictions with labels.

    Correct predictions shown in green, incorrect in red.

    Args:
        images: Input images [batch_size, channels, height, width]
        labels: True labels [batch_size]
        predictions: Predicted labels [batch_size]
        class_names: List of class names
        n_samples: Number of samples to display
        figsize: Figure size (None = auto)
        confidences: Optional prediction confidences [batch_size]
        cmap: Color map

    Example:
        >>> logits = model(images)
        >>> preds = logits.argmax(dim=1)
        >>> probs = F.softmax(logits, dim=1)
        >>> confidences = probs.max(dim=1)[0]
        >>> visualize_sample_predictions(
        ...     images, labels, preds, class_names, confidences=confidences
        ... )
    """
    if figsize is None:
        rows = (n_samples + 3) // 4
        figsize = (16, rows * 4)

    images = images.detach().cpu()
    labels = labels.detach().cpu()
    predictions = predictions.detach().cpu()
    if confidences is not None:
        confidences = confidences.detach().cpu()

    fig, axes = plt.subplots((n_samples + 3) // 4, 4, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_samples):
        # Get image
        img = images[i]
        if img.shape[0] == 1:  # Grayscale
            axes[i].imshow(img.squeeze(), cmap=cmap)
        else:  # RGB
            axes[i].imshow(img.permute(1, 2, 0))

        # Determine color (green=correct, red=incorrect)
        correct = (predictions[i] == labels[i]).item()
        color = 'green' if correct else 'red'

        # Create title
        true_label = class_names[labels[i]] if class_names else f'{labels[i]}'
        pred_label = class_names[predictions[i]] if class_names else f'{predictions[i]}'

        title = f"True: {true_label}\nPred: {pred_label}"
        if confidences is not None:
            title += f" ({confidences[i]*100:.1f}%)"

        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)',
                 fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.show()


def plot_interpolation(
    interpolations: List[torch.Tensor],
    start_label: Optional[str] = None,
    end_label: Optional[str] = None,
    figsize: Optional[tuple] = None,
    cmap: str = 'gray',
    title: str = "Latent Space Interpolation"
):
    """
    Visualize latent space interpolation between two samples.

    Args:
        interpolations: List of interpolated images
        start_label: Label for start image
        end_label: Label for end image
        figsize: Figure size (None = auto)
        cmap: Color map
        title: Plot title

    Example:
        >>> interpolations = interpolate_latents(model, img1, img2, n_steps=10)
        >>> plot_interpolation(interpolations, "Cat", "Dog")
    """
    n_steps = len(interpolations)
    if figsize is None:
        figsize = (n_steps * 1.5, 2)

    fig, axes = plt.subplots(1, n_steps, figsize=figsize)

    for i, (ax, img) in enumerate(zip(axes, interpolations)):
        img = img.detach().cpu()
        if img.ndim == 3 and img.shape[0] == 1:  # Grayscale with channel dim
            ax.imshow(img.squeeze(), cmap=cmap)
        elif img.ndim == 2:  # Grayscale without channel dim
            ax.imshow(img, cmap=cmap)
        else:  # RGB
            ax.imshow(img.permute(1, 2, 0))

        ax.axis('off')

        # Add labels to start and end
        if i == 0 and start_label:
            ax.set_title(start_label, fontsize=10)
        elif i == n_steps - 1 and end_label:
            ax.set_title(end_label, fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    model_names: List[str],
    metric_values: List[float],
    metric_name: str = "Accuracy",
    figsize: tuple = (10, 6),
    colors: Optional[List[str]] = None
):
    """
    Compare multiple models on a single metric.

    Args:
        model_names: List of model names
        metric_values: List of metric values (same order as model_names)
        metric_name: Name of metric being compared
        figsize: Figure size
        colors: Optional list of bar colors

    Example:
        >>> plot_model_comparison(
        ...     model_names=['Baseline', 'LSTM', 'BERT'],
        ...     metric_values=[0.85, 0.92, 0.95],
        ...     metric_name='Test Accuracy'
        ... )
    """
    if colors is None:
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12'][:len(model_names)]

    plt.figure(figsize=figsize)
    bars = plt.bar(model_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Model Comparison: {metric_name}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

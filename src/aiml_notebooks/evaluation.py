"""
Evaluation metrics and utilities for model assessment.

This module provides reusable evaluation functions for:
- Classification metrics (accuracy, precision, recall, F1)
- Per-class metrics
- Confusion matrix computation
- Model comparison utilities
"""

import torch
import numpy as np
from typing import Union, List, Tuple, Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List, torch.Tensor],
    y_pred: Union[np.ndarray, List, torch.Tensor],
    average: str = 'binary',
    labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
        labels: List of label indices to include (None = all)

    Returns:
        Dictionary with accuracy, precision, recall, f1

    Example:
        >>> metrics = compute_classification_metrics(true_labels, predictions)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"F1 Score: {metrics['f1']:.3f}")
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support if average is None else None
    }


def compute_per_class_metrics(
    y_true: Union[np.ndarray, List, torch.Tensor],
    y_pred: Union[np.ndarray, List, torch.Tensor],
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class individually.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)

    Returns:
        Dictionary mapping class names/indices to metrics

    Example:
        >>> per_class = compute_per_class_metrics(labels, preds, ['Cat', 'Dog'])
        >>> for class_name, metrics in per_class.items():
        ...     print(f"{class_name}: Precision={metrics['precision']:.3f}")
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Compute per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Create results dictionary
    results = {}
    for i in range(len(precision)):
        class_name = class_names[i] if class_names else str(i)
        results[class_name] = {
            'accuracy': per_class_accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }

    return results


def print_classification_report(
    y_true: Union[np.ndarray, List, torch.Tensor],
    y_pred: Union[np.ndarray, List, torch.Tensor],
    class_names: Optional[List[str]] = None,
    digits: int = 3
):
    """
    Print a formatted classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        digits: Number of decimal places

    Example:
        >>> print_classification_report(
        ...     test_labels,
        ...     predictions,
        ...     class_names=['Negative', 'Positive']
        ... )
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=digits,
        zero_division=0
    ))
    print("=" * 70)


def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model logits [batch_size, num_classes]
        labels: True labels [batch_size]
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a float

    Example:
        >>> top5_acc = compute_top_k_accuracy(logits, labels, k=5)
        >>> print(f"Top-5 Accuracy: {top5_acc:.3f}")
    """
    with torch.no_grad():
        # Get top k predictions
        _, top_k_preds = logits.topk(k, dim=1, largest=True, sorted=True)

        # Check if true label is in top k
        correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))

        # Compute accuracy
        top_k_acc = correct.sum().float() / labels.size(0)

    return top_k_acc.item()


def compute_confusion_matrix(
    y_true: Union[np.ndarray, List, torch.Tensor],
    y_pred: Union[np.ndarray, List, torch.Tensor],
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix as numpy array

    Example:
        >>> cm = compute_confusion_matrix(labels, preds, normalize='true')
        >>> print(f"Confusion matrix:\\n{cm}")
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()

    return cm


def compare_models(
    models_results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    ascending: bool = False
) -> List[Tuple[str, float]]:
    """
    Compare multiple models on a given metric.

    Args:
        models_results: Dictionary mapping model names to metric dictionaries
                       Example: {'Model1': {'accuracy': 0.9, 'f1': 0.85}, ...}
        metric: Metric to compare on
        ascending: Whether to sort in ascending order (default: descending)

    Returns:
        List of (model_name, metric_value) tuples sorted by metric

    Example:
        >>> results = {
        ...     'Baseline': {'accuracy': 0.85, 'f1': 0.83},
        ...     'LSTM': {'accuracy': 0.92, 'f1': 0.91},
        ...     'BERT': {'accuracy': 0.95, 'f1': 0.94}
        ... }
        >>> ranking = compare_models(results, metric='accuracy')
        >>> for rank, (name, value) in enumerate(ranking, 1):
        ...     print(f"{rank}. {name}: {value:.3f}")
    """
    # Extract metric values
    model_scores = [(name, results[metric]) for name, results in models_results.items()]

    # Sort by metric
    model_scores.sort(key=lambda x: x[1], reverse=not ascending)

    return model_scores


def print_model_comparison(
    models_results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
):
    """
    Print a formatted comparison table for multiple models.

    Args:
        models_results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to display (None = all)

    Example:
        >>> results = {
        ...     'Baseline': {'accuracy': 0.85, 'f1': 0.83, 'precision': 0.84},
        ...     'LSTM': {'accuracy': 0.92, 'f1': 0.91, 'precision': 0.90}
        ... }
        >>> print_model_comparison(results, metrics=['accuracy', 'f1'])
    """
    # Determine metrics to display
    if metrics is None:
        # Get all unique metrics
        all_metrics = set()
        for results in models_results.values():
            all_metrics.update(results.keys())
        metrics = sorted(list(all_metrics))

    # Print header
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    # Determine column widths
    model_width = max(len(name) for name in models_results.keys())
    metric_width = 10

    # Print table header
    header = f"{'Model':<{model_width}}"
    for metric in metrics:
        header += f" | {metric:>{metric_width}}"
    print(header)
    print("-" * len(header))

    # Print each model's results
    for model_name, results in models_results.items():
        row = f"{model_name:<{model_width}}"
        for metric in metrics:
            value = results.get(metric, 0.0)
            row += f" | {value:>{metric_width}.4f}"
        print(row)

    print("=" * 70)

    # Print best model for each metric
    print("\nBest Models:")
    for metric in metrics:
        best_model = max(models_results.items(), key=lambda x: x[1].get(metric, 0.0))
        print(f"  {metric:15s}: {best_model[0]} ({best_model[1][metric]:.4f})")
    print("=" * 70 + "\n")


def calculate_reconstruction_error(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    metric: str = 'mse',
    reduction: str = 'mean'
) -> Union[torch.Tensor, float]:
    """
    Calculate reconstruction error for autoencoders.

    Args:
        originals: Original data
        reconstructions: Reconstructed data
        metric: Error metric ('mse', 'mae', 'bce')
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Reconstruction error

    Example:
        >>> error = calculate_reconstruction_error(test_images, recon_images)
        >>> print(f"Average reconstruction error: {error:.4f}")
    """
    if metric == 'mse':
        error = torch.nn.functional.mse_loss(
            reconstructions, originals, reduction=reduction
        )
    elif metric == 'mae':
        error = torch.nn.functional.l1_loss(
            reconstructions, originals, reduction=reduction
        )
    elif metric == 'bce':
        error = torch.nn.functional.binary_cross_entropy(
            reconstructions, originals, reduction=reduction
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if reduction == 'none':
        return error
    else:
        return error.item()

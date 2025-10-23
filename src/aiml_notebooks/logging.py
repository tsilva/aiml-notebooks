"""
Logging utilities for W&B integration, gradient monitoring, and model visualization.
"""

import torch
import wandb
from typing import Dict, Optional


def log_gradients(model: torch.nn.Module, step: Optional[int] = None) -> Dict[str, float]:
    """
    Log gradient norms for model parameters to W&B.

    Args:
        model: PyTorch model
        step: Optional step number for logging

    Returns:
        Dictionary containing gradient statistics
    """
    total_norm = 0.0
    grad_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms[f"grad_norm/{name}"] = param_norm
            total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5
    grad_norms["grad_norm/total"] = total_norm

    # Log to W&B if available
    if wandb.run is not None:
        wandb.log(grad_norms, step=step)

    return grad_norms


def log_model_weights(model: torch.nn.Module, step: Optional[int] = None) -> Dict[str, float]:
    """
    Log parameter norms and statistics to W&B.

    Args:
        model: PyTorch model
        step: Optional step number for logging

    Returns:
        Dictionary containing parameter statistics
    """
    weight_stats = {}

    for name, param in model.named_parameters():
        param_data = param.data
        weight_stats[f"weight_norm/{name}"] = param_data.norm(2).item()
        weight_stats[f"weight_mean/{name}"] = param_data.mean().item()
        weight_stats[f"weight_std/{name}"] = param_data.std().item()

    # Log to W&B if available
    if wandb.run is not None:
        wandb.log(weight_stats, step=step)

    return weight_stats


def log_gradient_flow(model: torch.nn.Module, step: Optional[int] = None):
    """
    Create a visualization of gradient flow through the network.
    Logs average gradient per layer to help identify vanishing/exploding gradients.

    Args:
        model: PyTorch model
        step: Optional step number for logging
    """
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.grad is not None and "bias" not in name:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    # Create table for W&B
    if wandb.run is not None and len(layers) > 0:
        data = [[layer, avg, max_val] for layer, avg, max_val in zip(layers, ave_grads, max_grads)]
        table = wandb.Table(columns=["Layer", "Average Gradient", "Max Gradient"], data=data)
        wandb.log({"gradient_flow": table}, step=step)

    return {
        "layers": layers,
        "ave_grads": ave_grads,
        "max_grads": max_grads
    }

"""
Text and image generation utilities.

This module provides sampling strategies and generation utilities for:
- Text generation (temperature, top-k, nucleus sampling)
- Latent space interpolation
- Conditional generation helpers
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
import numpy as np


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1
) -> torch.Tensor:
    """
    Sample from logits with temperature scaling.

    Temperature controls randomness:
    - temperature > 1: More random (flatter distribution)
    - temperature < 1: More deterministic (sharper distribution)
    - temperature = 1: Unchanged distribution

    Args:
        logits: Logits tensor [..., vocab_size]
        temperature: Temperature parameter (must be > 0)
        dim: Dimension to sample from

    Returns:
        Sampled indices

    Example:
        >>> logits = model(input_ids)
        >>> next_token = sample_with_temperature(logits[:, -1, :], temperature=0.8)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Scale by temperature
    scaled_logits = logits / temperature

    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=dim)

    # Sample
    samples = torch.multinomial(probs, num_samples=1)

    return samples.squeeze(dim)


def sample_top_k(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    dim: int = -1
) -> torch.Tensor:
    """
    Sample from top-k most likely tokens.

    Filters out all tokens except the k most likely, then samples with temperature.

    Args:
        logits: Logits tensor [..., vocab_size]
        k: Number of top tokens to keep
        temperature: Temperature for sampling
        dim: Dimension to sample from

    Returns:
        Sampled indices

    Example:
        >>> # Only consider top 50 most likely tokens
        >>> next_token = sample_top_k(logits[:, -1, :], k=50, temperature=0.9)
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # Get top k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=dim)

    # Sample from top k with temperature
    scaled_logits = top_k_logits / temperature
    probs = F.softmax(scaled_logits, dim=dim)
    selected_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original vocabulary indices
    samples = torch.gather(top_k_indices, dim, selected_idx)

    return samples.squeeze(dim)


def sample_nucleus(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 1.0,
    dim: int = -1
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.

    Samples from the smallest set of tokens whose cumulative probability >= top_p.
    More dynamic than top-k as the number of tokens varies based on distribution.

    Args:
        logits: Logits tensor [..., vocab_size]
        top_p: Cumulative probability threshold (0 < top_p <= 1)
        temperature: Temperature for sampling
        dim: Dimension to sample from

    Returns:
        Sampled indices

    Example:
        >>> # Sample from tokens covering 90% of probability mass
        >>> next_token = sample_nucleus(logits[:, -1, :], top_p=0.9)
    """
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    # Scale by temperature
    scaled_logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=dim)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=dim)
    cumulative_probs = torch.cumsum(sorted_probs, dim=dim)

    # Find cutoff index (first index where cumsum > top_p)
    cutoff_mask = cumulative_probs > top_p

    # Keep at least one token
    if dim == -1:
        cutoff_mask[..., 0] = False
    else:
        cutoff_mask = cutoff_mask.transpose(dim, -1)
        cutoff_mask[..., 0] = False
        cutoff_mask = cutoff_mask.transpose(dim, -1)

    # Set logits of excluded tokens to -inf
    sorted_logits_filtered = sorted_logits.clone()
    sorted_logits_filtered[cutoff_mask] = float('-inf')

    # Sample from filtered distribution
    probs = F.softmax(sorted_logits_filtered, dim=dim)
    selected_idx = torch.multinomial(probs, num_samples=1)

    # Map back to original vocabulary indices
    samples = torch.gather(sorted_indices, dim, selected_idx)

    return samples.squeeze(dim)


def generate_text(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_length: int = 100,
    sampling_strategy: str = 'temperature',
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate text autoregressively with various sampling strategies.

    Args:
        model: Language model (should accept input_ids and return logits)
        prompt_ids: Initial prompt tokens [batch_size, seq_len]
        max_length: Maximum generation length
        sampling_strategy: 'greedy', 'temperature', 'top_k', or 'nucleus'
        temperature: Temperature for sampling
        top_k: K for top-k sampling
        top_p: P for nucleus sampling
        eos_token_id: End-of-sequence token (stops generation if sampled)
        device: Device to run on

    Returns:
        Generated token ids [batch_size, max_length]

    Example:
        >>> prompt = tokenizer.encode("Once upon a time")
        >>> prompt_ids = torch.tensor([prompt])
        >>> generated = generate_text(
        ...     model, prompt_ids, max_length=50,
        ...     sampling_strategy='nucleus', top_p=0.9, temperature=0.8
        ... )
        >>> print(tokenizer.decode(generated[0]))
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    generated = prompt_ids.to(device)

    with torch.no_grad():
        for _ in range(max_length - prompt_ids.size(1)):
            # Get logits for next token
            logits = model(generated)

            # Take logits for last position
            next_token_logits = logits[:, -1, :]

            # Sample based on strategy
            if sampling_strategy == 'greedy':
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            elif sampling_strategy == 'temperature':
                next_token = sample_with_temperature(
                    next_token_logits, temperature=temperature
                ).unsqueeze(-1)
            elif sampling_strategy == 'top_k':
                if top_k is None:
                    raise ValueError("top_k must be specified for top_k sampling")
                next_token = sample_top_k(
                    next_token_logits, k=top_k, temperature=temperature
                ).unsqueeze(-1)
            elif sampling_strategy == 'nucleus':
                if top_p is None:
                    raise ValueError("top_p must be specified for nucleus sampling")
                next_token = sample_nucleus(
                    next_token_logits, top_p=top_p, temperature=temperature
                ).unsqueeze(-1)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

    return generated


def interpolate_latents(
    model: torch.nn.Module,
    start_input: torch.Tensor,
    end_input: torch.Tensor,
    n_steps: int = 10,
    encode_fn: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Interpolate between two inputs in latent space.

    Args:
        model: Model with encode and decode methods (or use encode_fn/decode_fn)
        start_input: Starting input
        end_input: Ending input
        n_steps: Number of interpolation steps
        encode_fn: Optional custom encoding function (model, input) -> latent
        decode_fn: Optional custom decoding function (model, latent) -> output
        device: Device to run on

    Returns:
        List of interpolated outputs

    Example:
        >>> # For a VAE
        >>> interpolations = interpolate_latents(
        ...     vae, image1, image2, n_steps=10
        ... )
        >>> plot_interpolation(interpolations)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Default encode/decode functions
    if encode_fn is None:
        encode_fn = lambda m, x: m.encode(x) if hasattr(m, 'encode') else m.encoder(x)
    if decode_fn is None:
        decode_fn = lambda m, z: m.decode(z) if hasattr(m, 'decode') else m.decoder(z)

    with torch.no_grad():
        # Encode inputs
        start_input = start_input.unsqueeze(0).to(device) if start_input.dim() < 4 else start_input.to(device)
        end_input = end_input.unsqueeze(0).to(device) if end_input.dim() < 4 else end_input.to(device)

        z_start = encode_fn(model, start_input)
        z_end = encode_fn(model, end_input)

        # Handle VAE outputs (mu, logvar)
        if isinstance(z_start, tuple):
            z_start = z_start[0]  # Take mean
        if isinstance(z_end, tuple):
            z_end = z_end[0]  # Take mean

        # Interpolate in latent space
        interpolations = []
        for t in np.linspace(0, 1, n_steps):
            z_t = (1 - t) * z_start + t * z_end
            output_t = decode_fn(model, z_t)
            interpolations.append(output_t.squeeze(0).cpu())

    return interpolations


def spherical_interpolation(
    z1: torch.Tensor,
    z2: torch.Tensor,
    n_steps: int = 10
) -> List[torch.Tensor]:
    """
    Spherical linear interpolation (SLERP) between two latent vectors.

    Better than linear interpolation for normalized latent spaces
    (e.g., when using unit sphere constraints).

    Args:
        z1: Start latent vector
        z2: End latent vector
        n_steps: Number of interpolation steps

    Returns:
        List of interpolated latent vectors

    Example:
        >>> z_interp = spherical_interpolation(z1, z2, n_steps=10)
        >>> outputs = [model.decode(z) for z in z_interp]
    """
    # Normalize vectors
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    # Compute angle between vectors
    dot = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)

    # Compute interpolations
    sin_omega = torch.sin(omega)
    interpolations = []

    for t in np.linspace(0, 1, n_steps):
        if sin_omega.item() < 1e-6:
            # Vectors are parallel, use linear interpolation
            z_t = (1 - t) * z1 + t * z2
        else:
            # SLERP formula
            coef1 = torch.sin((1 - t) * omega) / sin_omega
            coef2 = torch.sin(t * omega) / sin_omega
            z_t = coef1 * z1 + coef2 * z2

        interpolations.append(z_t)

    return interpolations


def latent_arithmetic(
    model: torch.nn.Module,
    base_input: torch.Tensor,
    direction_vector: torch.Tensor,
    alpha: float = 1.0,
    encode_fn: Optional[Callable] = None,
    decode_fn: Optional[Callable] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Perform arithmetic in latent space: z' = z + alpha * direction.

    Useful for attribute manipulation (e.g., "smile direction" in faces).

    Args:
        model: Model with encode/decode
        base_input: Input to modify
        direction_vector: Direction in latent space
        alpha: Scaling factor for direction
        encode_fn: Optional custom encoding function
        decode_fn: Optional custom decoding function
        device: Device to run on

    Returns:
        Modified output

    Example:
        >>> # Compute "smile direction" from dataset
        >>> smile_dir = avg_latent(smiling_faces) - avg_latent(neutral_faces)
        >>> # Apply to new face
        >>> smiling_output = latent_arithmetic(vae, face, smile_dir, alpha=2.0)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Default encode/decode functions
    if encode_fn is None:
        encode_fn = lambda m, x: m.encode(x) if hasattr(m, 'encode') else m.encoder(x)
    if decode_fn is None:
        decode_fn = lambda m, z: m.decode(z) if hasattr(m, 'decode') else m.decoder(z)

    with torch.no_grad():
        # Encode base input
        base_input = base_input.unsqueeze(0).to(device) if base_input.dim() < 4 else base_input.to(device)
        z_base = encode_fn(model, base_input)

        # Handle VAE outputs (mu, logvar)
        if isinstance(z_base, tuple):
            z_base = z_base[0]  # Take mean

        # Apply direction
        direction_vector = direction_vector.to(device)
        z_modified = z_base + alpha * direction_vector

        # Decode
        output = decode_fn(model, z_modified)

    return output.squeeze(0).cpu()

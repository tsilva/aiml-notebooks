"""
Positional encoding implementations for sequence models.

This module provides various positional encoding strategies for transformers
and other sequence models:
- Sinusoidal (fixed) positional encoding from "Attention is All You Need"
- Learnable positional embeddings
- Relative positional encoding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Uses sine and cosine functions of different frequencies to encode positions:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This is a fixed encoding (no learnable parameters).

    Example:
        >>> pos_enc = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 100, 512)  # [batch, seq_len, d_model]
        >>> x_with_pos = pos_enc(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        scale: float = 1.0
    ):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Model dimensionality (must be even)
            max_len: Maximum sequence length to precompute
            dropout: Dropout probability applied after adding encoding
            scale: Scaling factor for positional encoding (default 1.0)

        Raises:
            ValueError: If d_model is not even
        """
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term for sine and cosine
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]

        Raises:
            ValueError: If sequence length exceeds max_len
        """
        batch_size, seq_len, d_model = x.size()

        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.pe.size(1)}"
            )

        # Add positional encoding (scaled)
        x = x + self.scale * self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Similar to word embeddings, but for positions. These are learned during training.
    Common in models like BERT and GPT.

    Example:
        >>> pos_emb = LearnablePositionalEmbedding(max_len=512, d_model=768)
        >>> x = torch.randn(32, 100, 768)  # [batch, seq_len, d_model]
        >>> x_with_pos = pos_emb(x)
    """

    def __init__(
        self,
        max_len: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Initialize learnable positional embedding.

        Args:
            max_len: Maximum sequence length
            d_model: Model dimensionality
            dropout: Dropout probability
        """
        super().__init__()

        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional embedding added [batch_size, seq_len, d_model]

        Raises:
            ValueError: If sequence length exceeds max_len
        """
        batch_size, seq_len, d_model = x.size()

        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]

        # Get positional embeddings [1, seq_len, d_model]
        pos_emb = self.pos_embedding(positions)

        # Add to input
        x = x + pos_emb
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for self-attention.

    Instead of absolute positions, this encodes relative distances between positions.
    Used in models like Transformer-XL and T5.

    This is a simplified implementation that adds relative position biases
    to attention scores.

    Example:
        >>> rel_pos = RelativePositionalEncoding(d_model=512, max_relative_position=32)
        >>> # In attention computation:
        >>> bias = rel_pos(seq_len_q=100, seq_len_k=100)
        >>> # Add bias to attention scores before softmax
    """

    def __init__(
        self,
        d_model: int,
        max_relative_position: int = 128,
        num_heads: int = 8
    ):
        """
        Initialize relative positional encoding.

        Args:
            d_model: Model dimensionality
            max_relative_position: Maximum relative distance to encode
            num_heads: Number of attention heads
        """
        super().__init__()

        self.max_relative_position = max_relative_position
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Learnable relative position embeddings
        # We need 2 * max_relative_position + 1 positions
        # (negative distances, zero, positive distances)
        num_relative_positions = 2 * max_relative_position + 1
        self.relative_position_embeddings = nn.Embedding(
            num_relative_positions,
            self.head_dim
        )

    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Clip relative positions to [-max_relative_position, max_relative_position].

        Args:
            relative_position: Relative positions tensor

        Returns:
            Clipped and shifted positions for embedding lookup
        """
        # Clip to [-max, max]
        relative_position = torch.clamp(
            relative_position,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift to [0, 2*max] for embedding lookup
        return relative_position + self.max_relative_position

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute relative position bias for attention.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            device: Device for tensor creation

        Returns:
            Relative position bias [num_heads, seq_len_q, seq_len_k]

        Example:
            >>> rel_pos = RelativePositionalEncoding(d_model=512, num_heads=8)
            >>> bias = rel_pos(seq_len_q=100, seq_len_k=100)
            >>> # In attention: scores = scores + bias.unsqueeze(0)  # Add batch dim
        """
        if device is None:
            device = self.relative_position_embeddings.weight.device

        # Create position indices
        q_positions = torch.arange(seq_len_q, device=device).unsqueeze(1)  # [seq_q, 1]
        k_positions = torch.arange(seq_len_k, device=device).unsqueeze(0)  # [1, seq_k]

        # Compute relative positions [seq_q, seq_k]
        relative_positions = q_positions - k_positions

        # Bucket and lookup embeddings
        bucketed = self._relative_position_bucket(relative_positions)
        embeddings = self.relative_position_embeddings(bucketed)  # [seq_q, seq_k, head_dim]

        # Reshape for multi-head attention [num_heads, seq_q, seq_k]
        # We replicate the same bias for all heads
        embeddings = embeddings.permute(2, 0, 1)  # [head_dim, seq_q, seq_k]

        # If head_dim doesn't match num_heads, we need to handle it
        # For simplicity, we'll just use the first num_heads dimensions
        if self.head_dim >= self.num_heads:
            bias = embeddings[:self.num_heads]
        else:
            # Repeat to match num_heads
            repeats = (self.num_heads + self.head_dim - 1) // self.head_dim
            bias = embeddings.repeat(repeats, 1, 1)[:self.num_heads]

        return bias


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal (triangular) mask for autoregressive models.

    Prevents positions from attending to future positions.
    Used in decoder-only models like GPT.

    Args:
        seq_len: Sequence length
        device: Device for tensor creation

    Returns:
        Boolean mask [seq_len, seq_len] where True = masked position
        Upper triangle is True (cannot attend to future)

    Example:
        >>> mask = create_causal_mask(5)
        >>> print(mask.int())
        tensor([[0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]])
        >>> # Use in attention: attn_scores.masked_fill(mask, float('-inf'))
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_padding_mask(
    seq: torch.Tensor,
    pad_idx: int = 0
) -> torch.Tensor:
    """
    Create padding mask from sequence with padding.

    Args:
        seq: Sequence tensor [batch_size, seq_len]
        pad_idx: Padding token index

    Returns:
        Boolean mask [batch_size, seq_len] where True = padding position

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        >>> mask = create_padding_mask(seq, pad_idx=0)
        >>> print(mask.int())
        tensor([[0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1]])
    """
    return seq == pad_idx


def create_attention_mask(
    seq: torch.Tensor,
    pad_idx: int = 0,
    causal: bool = False
) -> torch.Tensor:
    """
    Create combined attention mask (padding + causal).

    Args:
        seq: Sequence tensor [batch_size, seq_len]
        pad_idx: Padding token index
        causal: Whether to apply causal masking

    Returns:
        Boolean mask [batch_size, seq_len, seq_len] where True = masked position

    Example:
        >>> seq = torch.tensor([[1, 2, 3, 0, 0]])
        >>> mask = create_attention_mask(seq, pad_idx=0, causal=True)
        >>> # Use in attention: attn_scores.masked_fill(mask, float('-inf'))
    """
    batch_size, seq_len = seq.size()

    # Padding mask [batch_size, 1, seq_len]
    pad_mask = create_padding_mask(seq, pad_idx).unsqueeze(1)

    if causal:
        # Causal mask [seq_len, seq_len]
        causal_mask = create_causal_mask(seq_len, device=seq.device)

        # Combine: [batch_size, seq_len, seq_len]
        # Broadcasting: [batch_size, 1, seq_len] | [1, seq_len, seq_len]
        mask = pad_mask | causal_mask.unsqueeze(0)
    else:
        # Just expand padding mask to [batch_size, seq_len, seq_len]
        mask = pad_mask.expand(batch_size, seq_len, seq_len)

    return mask


def get_positional_encoding(
    encoding_type: str = 'sinusoidal',
    d_model: int = 512,
    max_len: int = 5000,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create positional encoding modules.

    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learnable', 'relative')
        d_model: Model dimensionality
        max_len: Maximum sequence length
        dropout: Dropout probability
        **kwargs: Additional arguments for specific encoding types

    Returns:
        Positional encoding module

    Raises:
        ValueError: If encoding_type is not recognized

    Example:
        >>> # Sinusoidal encoding
        >>> pos_enc = get_positional_encoding('sinusoidal', d_model=512)
        >>>
        >>> # Learnable embeddings
        >>> pos_emb = get_positional_encoding('learnable', d_model=768, max_len=512)
        >>>
        >>> # Relative encoding
        >>> rel_pos = get_positional_encoding(
        ...     'relative',
        ...     d_model=512,
        ...     max_relative_position=32,
        ...     num_heads=8
        ... )
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            **kwargs
        )
    elif encoding_type == 'learnable':
        return LearnablePositionalEmbedding(
            max_len=max_len,
            d_model=d_model,
            dropout=dropout
        )
    elif encoding_type == 'relative':
        num_heads = kwargs.get('num_heads', 8)
        max_relative_position = kwargs.get('max_relative_position', 128)
        return RelativePositionalEncoding(
            d_model=d_model,
            max_relative_position=max_relative_position,
            num_heads=num_heads
        )
    else:
        raise ValueError(
            f"Unknown encoding_type: {encoding_type}. "
            f"Choose from: 'sinusoidal', 'learnable', 'relative'"
        )

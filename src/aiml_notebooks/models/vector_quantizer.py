"""
Vector Quantization layer for VQ-VAE.

Implements the vector quantization operation that maps continuous latent
vectors to discrete codebook entries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.

    Maps continuous encoder outputs to discrete codebook vectors using
    nearest neighbor search, with straight-through estimator for gradients.

    Example:
        >>> vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
        >>> z_q, vq_loss, perplexity = vq(z_e)
        >>> # z_q has same shape as z_e but quantized
        >>> # Use z_q in decoder, vq_loss in total loss
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """
        Initialize Vector Quantizer.

        Args:
            num_embeddings: Size of codebook (number of discrete vectors)
            embedding_dim: Dimension of each codebook vector
            commitment_cost: Weight for commitment loss (encourages encoder commitment)
            decay: Decay for exponential moving average (if using EMA)
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        """
        Quantize continuous latent vectors.

        Args:
            z_e: Encoder outputs [batch_size, ..., embedding_dim]
                 Can be any shape as long as last dimension is embedding_dim

        Returns:
            z_q: Quantized vectors (same shape as z_e)
            vq_loss: Vector quantization loss (codebook + commitment)
            perplexity: Measure of codebook usage
        """
        # Flatten input while preserving batch and embedding dimensions
        input_shape = z_e.shape
        flat_input = z_e.view(-1, self.embedding_dim)

        # Calculate distances to codebook vectors
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2 * z_e @ e^T
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Find nearest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Convert to one-hot encodings
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=z_e.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize by retrieving codebook vectors
        quantized = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized.view(input_shape)

        # Compute losses
        # Codebook loss: move codebook vectors towards encoder outputs
        codebook_loss = F.mse_loss(quantized.detach(), z_e)

        # Commitment loss: encourage encoder to commit to codebook vectors
        commitment_loss = F.mse_loss(quantized, z_e.detach())

        # Combined VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from decoder to encoder
        quantized = z_e + (quantized - z_e).detach()

        # Calculate perplexity (measure of codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity

    def get_codebook_entry(self, indices):
        """
        Retrieve codebook vectors by index.

        Args:
            indices: Codebook indices [batch_size, ...]

        Returns:
            Codebook vectors [batch_size, ..., embedding_dim]
        """
        return self.embedding(indices)


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average (EMA) updates.

    Uses EMA to update codebook vectors instead of gradient descent,
    which can be more stable for training.

    Example:
        >>> vq_ema = VectorQuantizerEMA(num_embeddings=512, embedding_dim=64)
        >>> z_q, vq_loss, perplexity = vq_ema(z_e)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """
        Initialize VQ-EMA.

        Args:
            num_embeddings: Size of codebook
            embedding_dim: Dimension of vectors
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate (higher = slower updates)
            epsilon: Small constant for stability
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embeddings (not trained via gradient descent)
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embedding.clone())

    def forward(self, z_e):
        """
        Quantize with EMA updates.

        Args:
            z_e: Encoder outputs [batch_size, ..., embedding_dim]

        Returns:
            z_q: Quantized vectors
            vq_loss: Commitment loss (no codebook loss with EMA)
            perplexity: Codebook usage measure
        """
        # Flatten input
        input_shape = z_e.shape
        flat_input = z_e.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.t())
        )

        # Find nearest codebook vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # One-hot encodings
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=z_e.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view(input_shape)

        # Update embeddings with EMA (only during training)
        if self.training:
            with torch.no_grad():
                # Update cluster sizes
                self.ema_cluster_size = (
                    self.ema_cluster_size * self.decay
                    + (1 - self.decay) * torch.sum(encodings, dim=0)
                )

                # Laplace smoothing
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )

                # Update embeddings
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

                self.embedding = self.ema_w / self.ema_cluster_size.unsqueeze(1)

        # Commitment loss only (no codebook loss with EMA)
        commitment_loss = F.mse_loss(quantized, z_e.detach())
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z_e + (quantized - z_e).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity

    def get_codebook_entry(self, indices):
        """Retrieve codebook vectors by index."""
        return F.embedding(indices, self.embedding)

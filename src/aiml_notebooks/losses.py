"""
Loss functions for various autoencoder and generative models.

This module provides reusable loss functions that appear across multiple notebooks,
including VAE loss, VQ-VAE loss, and other common loss computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """
    Variational Autoencoder (VAE) loss function.

    Combines reconstruction loss (BCE or MSE) with KL divergence regularization.

    Loss = Reconstruction Loss + beta * KL Divergence

    Args:
        beta: Weight for KL divergence term (default: 1.0)
              beta > 1: Emphasize disentanglement (beta-VAE)
              beta < 1: Emphasize reconstruction quality
        reconstruction_loss: Type of reconstruction loss ('bce' or 'mse')
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> vae_loss = VAELoss(beta=1.0, reconstruction_loss='bce')
        >>> loss, recon, kl = vae_loss(recon_x, x, mu, logvar)
    """

    def __init__(self, beta: float = 1.0, reconstruction_loss: str = 'bce', reduction: str = 'mean'):
        super().__init__()
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.reduction = reduction

    def forward(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss.

        Args:
            recon_x: Reconstructed data [batch_size, ...]
            x: Original data [batch_size, ...]
            mu: Latent mean [batch_size, latent_dim]
            logvar: Latent log-variance [batch_size, latent_dim]

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        batch_size = x.size(0)

        # Reconstruction loss
        if self.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        # Normalize by batch size
        if self.reduction == 'mean':
            recon_loss = recon_loss / batch_size

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.reduction == 'mean':
            kl_loss = kl_loss / batch_size

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class VQVAELoss(nn.Module):
    """
    Vector-Quantized VAE (VQ-VAE) loss function.

    Combines reconstruction loss with vector quantization losses:
    - Codebook loss: Moves codebook vectors towards encoder outputs
    - Commitment loss: Encourages encoder to commit to codebook vectors

    Loss = Reconstruction Loss + Codebook Loss + commitment_cost * Commitment Loss

    Args:
        commitment_cost: Weight for commitment loss (default: 0.25)
        reconstruction_loss: Type of reconstruction loss ('bce' or 'mse')
        reduction: Reduction method ('mean', 'sum', or 'none')

    Example:
        >>> vqvae_loss = VQVAELoss(commitment_cost=0.25)
        >>> loss, recon, vq = vqvae_loss(recon_x, x, z_e, z_q)
    """

    def __init__(self, commitment_cost: float = 0.25, reconstruction_loss: str = 'bce', reduction: str = 'mean'):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.reconstruction_loss = reconstruction_loss
        self.reduction = reduction

    def forward(self, recon_x, x, z_e, z_q):
        """
        Compute VQ-VAE loss.

        Args:
            recon_x: Reconstructed data [batch_size, ...]
            x: Original data [batch_size, ...]
            z_e: Encoder output (continuous) [batch_size, ...]
            z_q: Quantized vectors (from codebook) [batch_size, ...]

        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            vq_loss: Vector quantization loss (codebook + commitment)
        """
        batch_size = x.size(0)

        # Reconstruction loss
        if self.reconstruction_loss == 'bce':
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        elif self.reconstruction_loss == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss}")

        if self.reduction == 'mean':
            recon_loss = recon_loss / batch_size

        # Vector quantization losses
        # Codebook loss: sg[z_e] -> e (move codebook towards encoder)
        codebook_loss = F.mse_loss(z_q.detach(), z_e)

        # Commitment loss: z_e -> sg[e] (encourage encoder to commit)
        commitment_loss = F.mse_loss(z_q, z_e.detach())

        # Combined VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Total loss
        total_loss = recon_loss + vq_loss

        return total_loss, recon_loss, vq_loss


class GANLoss(nn.Module):
    """
    GAN loss functions for generator and discriminator.

    Supports multiple GAN variants:
    - 'vanilla': Original GAN with BCE loss
    - 'lsgan': Least Squares GAN (MSE loss)
    - 'wgan': Wasserstein GAN (not implemented here, requires gradient penalty)

    Args:
        gan_mode: Type of GAN loss ('vanilla' or 'lsgan')
        label_smoothing: Whether to use label smoothing for discriminator
                        (e.g., 0.9 instead of 1.0 for real labels)

    Example:
        >>> gan_loss = GANLoss(gan_mode='vanilla', label_smoothing=True)
        >>> d_loss = gan_loss.discriminator_loss(real_pred, fake_pred)
        >>> g_loss = gan_loss.generator_loss(fake_pred)
    """

    def __init__(self, gan_mode: str = 'vanilla', label_smoothing: bool = True):
        super().__init__()
        self.gan_mode = gan_mode
        self.label_smoothing = label_smoothing
        self.real_label = 0.9 if label_smoothing else 1.0
        self.fake_label = 0.0

    def discriminator_loss(self, real_pred, fake_pred):
        """
        Compute discriminator loss.

        Args:
            real_pred: Discriminator predictions on real data [batch_size, 1]
            fake_pred: Discriminator predictions on fake data [batch_size, 1]

        Returns:
            loss: Discriminator loss
        """
        batch_size = real_pred.size(0)
        real_labels = torch.full((batch_size, 1), self.real_label, device=real_pred.device)
        fake_labels = torch.full((batch_size, 1), self.fake_label, device=fake_pred.device)

        if self.gan_mode == 'vanilla':
            real_loss = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
        elif self.gan_mode == 'lsgan':
            real_loss = F.mse_loss(real_pred, real_labels)
            fake_loss = F.mse_loss(fake_pred, fake_labels)
        else:
            raise ValueError(f"Unknown GAN mode: {self.gan_mode}")

        return (real_loss + fake_loss) / 2

    def generator_loss(self, fake_pred):
        """
        Compute generator loss.

        Args:
            fake_pred: Discriminator predictions on fake data [batch_size, 1]

        Returns:
            loss: Generator loss
        """
        batch_size = fake_pred.size(0)
        real_labels = torch.full((batch_size, 1), 1.0, device=fake_pred.device)

        if self.gan_mode == 'vanilla':
            loss = F.binary_cross_entropy_with_logits(fake_pred, real_labels)
        elif self.gan_mode == 'lsgan':
            loss = F.mse_loss(fake_pred, real_labels)
        else:
            raise ValueError(f"Unknown GAN mode: {self.gan_mode}")

        return loss


def perplexity(encodings):
    """
    Calculate perplexity of codebook usage in VQ-VAE.

    Perplexity measures how many codebook vectors are actively used.
    Higher perplexity = more diverse codebook usage = better.

    Args:
        encodings: One-hot encoded indices [batch_size, num_embeddings]

    Returns:
        perplexity: Scalar tensor

    Example:
        >>> perp = perplexity(encodings)
        >>> print(f"Codebook usage: {perp.item():.2f} / {num_embeddings}")
    """
    # Average probability of each codebook vector
    avg_probs = torch.mean(encodings, dim=0)

    # Perplexity = exp(entropy)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    return perplexity

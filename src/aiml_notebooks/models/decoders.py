"""
Decoder architectures for various data types.

Decoders map low-dimensional latent representations back to high-dimensional outputs.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class MLPDecoder(nn.Module):
    """
    Multi-Layer Perceptron decoder.

    Example:
        >>> decoder = MLPDecoder(latent_dim=32, output_dim=784, hidden_dims=[256, 512])
        >>> x_recon = decoder(z).view(-1, 1, 28, 28)  # Reshape to image
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = nn.Sigmoid(),
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        """
        Initialize MLP decoder.

        Args:
            latent_dim: Latent dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions (e.g., [256, 512])
            activation: Activation function for hidden layers
            output_activation: Activation for output (None for no activation)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512]

        layers = []
        prev_dim = latent_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.decoder = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for images (transpose convolutions).

    Example:
        >>> decoder = ConvDecoder(
        ...     latent_dim=256,
        ...     out_channels=3,
        ...     base_channels=64,
        ...     num_layers=4,
        ...     output_size=(64, 64)
        ... )
        >>> images = decoder(z)  # [batch, 256] -> [batch, 3, 64, 64]
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        base_channels: int = 64,
        num_layers: int = 4,
        output_size: Tuple[int, int] = (64, 64),
        activation: nn.Module = nn.ReLU(inplace=True),
        output_activation: Optional[nn.Module] = nn.Sigmoid(),
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize convolutional decoder.

        Args:
            latent_dim: Latent dimension
            out_channels: Number of output channels (e.g., 3 for RGB)
            base_channels: Base number of channels (halves each layer going up)
            num_layers: Number of transpose convolutional layers
            output_size: Target output size (height, width)
            activation: Activation function
            output_activation: Final activation (None for no activation)
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_size = output_size

        # Calculate initial spatial size
        # Each ConvTranspose layer doubles spatial dimensions
        initial_size = output_size[0] // (2 ** num_layers)
        self.initial_size = initial_size

        # Calculate initial number of channels (highest in decoder)
        initial_channels = base_channels * (2 ** (num_layers - 1))
        self.initial_channels = initial_channels

        # Project latent to initial feature map
        self.fc = nn.Linear(latent_dim, initial_channels * initial_size * initial_size)

        # Build transpose convolutional layers
        layers = []
        channels = initial_channels

        for i in range(num_layers):
            out_ch = channels // 2 if i < num_layers - 1 else out_channels

            # ConvTranspose block
            layers.extend([
                nn.ConvTranspose2d(channels, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch) if batch_norm and i < num_layers - 1 else nn.Identity(),
                activation if i < num_layers - 1 else nn.Identity()
            ])

            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout2d(dropout))

            channels = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, z):
        """Decode latent representation to image."""
        # Project and reshape to feature map
        features = self.fc(z)
        features = features.view(
            features.size(0),
            self.initial_channels,
            self.initial_size,
            self.initial_size
        )

        # Pass through transpose conv layers
        output = self.conv_layers(features)

        # Apply output activation
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output


class RNNDecoder(nn.Module):
    """
    RNN/LSTM/GRU decoder for sequences.

    Example:
        >>> decoder = RNNDecoder(
        ...     latent_dim=64,
        ...     vocab_size=10000,
        ...     embedding_dim=128,
        ...     hidden_dim=256,
        ...     rnn_type='lstm'
        ... )
        >>> logits = decoder(z, input_ids)  # [batch, seq_len, vocab_size]
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        """
        Initialize RNN decoder.

        Args:
            latent_dim: Latent dimension
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm', 'gru', or 'rnn')
            dropout: Dropout probability
            padding_idx: Padding index for embeddings
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        # Project latent to initial hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        # RNN layer
        rnn_class = {
            'lstm': nn.LSTM,
            'gru': nn.GRU,
            'rnn': nn.RNN
        }[rnn_type.lower()]

        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, input_ids):
        """
        Decode latent representation to sequence.

        Args:
            z: Latent vector [batch, latent_dim]
            input_ids: Input token IDs [batch, seq_len] (for teacher forcing)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = z.size(0)

        # Initialize hidden state from latent
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()  # [num_layers, batch, hidden_dim]

        # For LSTM, also initialize cell state
        if self.rnn_type == 'lstm':
            cell = torch.zeros_like(hidden)
            hidden = (hidden, cell)

        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Pass through RNN
        rnn_out, _ = self.rnn(embedded, hidden)  # [batch, seq_len, hidden_dim]

        # Project to vocabulary
        logits = self.fc_out(rnn_out)  # [batch, seq_len, vocab_size]

        return logits

    def generate(self, z, max_length=100, start_token_id=1, eos_token_id=None, temperature=1.0):
        """
        Generate sequence autoregressively from latent vector.

        Args:
            z: Latent vector [batch, latent_dim]
            max_length: Maximum generation length
            start_token_id: Starting token ID
            eos_token_id: End-of-sequence token ID (stops early if generated)
            temperature: Sampling temperature

        Returns:
            Generated token IDs [batch, max_length]
        """
        self.eval()
        batch_size = z.size(0)
        device = z.device

        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        # Initialize hidden state
        hidden = self.latent_to_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()

        if self.rnn_type == 'lstm':
            cell = torch.zeros_like(hidden)
            hidden = (hidden, cell)

        with torch.no_grad():
            for _ in range(max_length - 1):
                # Embed last token
                embedded = self.embedding(generated[:, -1:])

                # RNN step
                rnn_out, hidden = self.rnn(embedded, hidden)

                # Get logits
                logits = self.fc_out(rnn_out[:, -1, :])  # [batch, vocab_size]

                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if EOS token generated
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

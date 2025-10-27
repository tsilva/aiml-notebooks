"""
Encoder architectures for various data types.

Encoders map high-dimensional inputs to lower-dimensional latent representations.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron encoder.

    Example:
        >>> encoder = MLPEncoder(input_dim=784, latent_dim=32, hidden_dims=[512, 256])
        >>> z = encoder(x.view(x.size(0), -1))  # Flatten input
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        """
        Initialize MLP encoder.

        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256])
            activation: Activation function
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev_dim = input_dim

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
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, x):
        """Encode input to latent representation."""
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for images.

    Example:
        >>> encoder = ConvEncoder(
        ...     in_channels=3,
        ...     latent_dim=256,
        ...     base_channels=64,
        ...     num_layers=4
        ... )
        >>> z = encoder(images)  # [batch, 3, H, W] -> [batch, 256]
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int = 64,
        num_layers: int = 4,
        activation: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize convolutional encoder.

        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            latent_dim: Latent dimension
            base_channels: Base number of channels (doubles each layer)
            num_layers: Number of convolutional layers
            activation: Activation function
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        channels = in_channels

        # Build convolutional layers
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)

            # Conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
            layers.extend([
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                activation,
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                activation,
                nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
            ])

            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.final_channels = channels

        # Will be set in first forward pass
        self.flatten_dim = None
        self.fc = None

    def forward(self, x):
        """Encode image to latent representation."""
        # Pass through conv layers
        features = self.conv_layers(x)

        # Initialize FC layer on first forward pass
        if self.fc is None:
            self.flatten_dim = features.view(features.size(0), -1).size(1)
            self.fc = nn.Linear(self.flatten_dim, self.latent_dim).to(x.device)

        # Flatten and project to latent dimension
        features_flat = features.view(features.size(0), -1)
        latent = self.fc(features_flat)

        return latent


class RNNEncoder(nn.Module):
    """
    RNN/LSTM/GRU encoder for sequences.

    Example:
        >>> encoder = RNNEncoder(
        ...     vocab_size=10000,
        ...     embedding_dim=128,
        ...     hidden_dim=256,
        ...     latent_dim=64,
        ...     rnn_type='lstm'
        ... )
        >>> z = encoder(input_ids)  # [batch, seq_len] -> [batch, 64]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.5,
        padding_idx: int = 0
    ):
        """
        Initialize RNN encoder.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension of RNN
            latent_dim: Latent dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm', 'gru', or 'rnn')
            bidirectional: Whether to use bidirectional RNN
            dropout: Dropout probability
            padding_idx: Padding index for embeddings
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

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
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Project hidden state to latent dimension
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_dim, latent_dim)

    def forward(self, input_ids, lengths=None):
        """
        Encode sequence to latent representation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            lengths: Optional sequence lengths for packing

        Returns:
            Latent vector [batch, latent_dim]
        """
        # Embed tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

        # Pack if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )

        # Pass through RNN
        _, hidden = self.rnn(embedded)

        # Extract final hidden state
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]  # Take hidden, not cell state

        # Concatenate forward and backward if bidirectional
        if self.bidirectional:
            # hidden: [num_layers*2, batch, hidden_dim]
            hidden_fwd = hidden[-2, :, :]  # Forward direction from last layer
            hidden_bwd = hidden[-1, :, :]  # Backward direction from last layer
            hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        else:
            # hidden: [num_layers, batch, hidden_dim]
            hidden_concat = hidden[-1, :, :]  # Last layer

        # Project to latent dimension
        latent = self.fc(hidden_concat)

        return latent

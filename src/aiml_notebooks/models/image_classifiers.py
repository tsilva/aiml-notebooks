"""
Image classification models with hotswappable architectures.

This module provides:
- ImageClassifier: Base Lightning Module with common training logic
- CNNArchitecture: Convolutional neural network
- MLPArchitecture: Multi-layer perceptron
- create_image_classifier: Factory function for easy configuration-based instantiation
"""

import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Optional, List, Tuple


class ImageClassifier(L.LightningModule):
    """
    Base image classifier with common training logic and hotswappable architectures.

    This class handles:
    - Training and validation loops
    - Logging and visualization
    - Optimizer configuration
    - Loss computation

    The actual forward pass is delegated to the architecture (CNN or MLP).
    """

    def __init__(
        self,
        architecture: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        class_names: Optional[List[str]] = None,
        dataset_mean: Optional[Tuple[float, ...]] = None,
        dataset_std: Optional[Tuple[float, ...]] = None,
    ):
        """
        Args:
            architecture: The neural network architecture (CNN, MLP, etc.)
            num_classes: Number of output classes
            learning_rate: Learning rate for optimizer
            class_names: List of class names for logging
            dataset_mean: Dataset mean for visualization
            dataset_std: Dataset std for visualization
        """
        super().__init__()

        self.save_hyperparameters(ignore=['architecture'])

        # Store the architecture
        self.architecture = architecture

        # Cross-entropy loss function
        self.criterion = nn.CrossEntropyLoss()

        # Track predictions for evaluation
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []
        self.val_logits = []

    def forward(self, x):
        """Forward pass through the architecture."""
        return self.architecture(x)

    def training_step(self, batch, batch_idx):
        """Training step with logging."""
        # Import here to avoid circular dependency
        from ..logging import log_gradients

        x, y = batch
        logits = self(x)

        # Compute loss
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        # Log gradients every 10 steps to reduce overhead
        if batch_idx % 10 == 0:
            log_gradients(self, step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with logging."""
        x, y = batch
        logits = self(x)

        # Compute loss
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Store predictions and labels for visualization
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())

        # Store first 16 images and logits for prediction grid
        if len(self.val_images) < 16:
            remaining = 16 - len(self.val_images)
            self.val_images.extend(x[:remaining].cpu())
            self.val_logits.extend(logits[:remaining].cpu())

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step - same as validation but logs with 'test' prefix."""
        x, y = batch
        logits = self(x)

        # Compute loss
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """Log visualizations at end of validation epoch."""
        # Import here to avoid circular dependency
        from ..visualization import (
            log_confusion_matrix_callback,
            log_prediction_grid_callback,
        )

        # Only log visualizations if we have predictions and required metadata
        if len(self.val_predictions) > 0 and self.hparams.class_names is not None:
            # Log confusion matrix
            log_confusion_matrix_callback(
                predictions=self.val_predictions,
                labels=self.val_labels,
                class_names=self.hparams.class_names,
                epoch=self.current_epoch,
                log_to_wandb=True
            )

            # Log prediction grid
            log_prediction_grid_callback(
                images=self.val_images,
                logits=self.val_logits,
                labels=self.val_labels,
                class_names=self.hparams.class_names,
                dataset_mean=self.hparams.dataset_mean,
                dataset_std=self.hparams.dataset_std,
                epoch=self.current_epoch,
                log_to_wandb=True
            )

        # Clear stored data
        self.val_predictions = []
        self.val_labels = []
        self.val_images = []
        self.val_logits = []

    def on_train_epoch_end(self):
        """Log detailed gradient flow and weights at end of each epoch."""
        # Import here to avoid circular dependency
        from ..logging import log_gradient_flow, log_model_weights

        log_gradient_flow(self, step=self.global_step)
        log_model_weights(self, step=self.global_step)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

        # Reduce learning rate on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class CNNArchitecture(nn.Module):
    """
    Convolutional Neural Network architecture for image classification.

    Architecture:
    - Multiple convolutional blocks (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool)
    - Fully connected classifier (Flatten -> Linear -> ReLU -> Dropout -> Linear)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        input_size: int = 32,
        num_conv_layers: int = 3,
        base_channels: int = 32,
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            input_size: Input image size (assumes square images)
            num_conv_layers: Number of convolutional blocks
            base_channels: Number of channels in first conv layer (doubles each block)
            dropout: Dropout rate in classifier
        """
        super().__init__()

        # Build convolutional layers dynamically
        conv_layers = []
        channels = in_channels

        for i in range(num_conv_layers):
            out_channels = base_channels * (2 ** i)  # Double channels each layer

            # Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool
            conv_layers.extend([
                nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions
            ])

            channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of the flattened features
        # Input size after N pooling layers: input_size / (2^N)
        final_spatial_size = input_size // (2 ** num_conv_layers)
        final_channels = base_channels * (2 ** (num_conv_layers - 1))
        flattened_size = final_channels * final_spatial_size * final_spatial_size

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through CNN."""
        features = self.conv_layers(x)
        logits = self.classifier(features)
        return logits


class MLPArchitecture(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture for image classification.

    Architecture:
    - Flatten input image
    - Multiple fully connected layers with BatchNorm, ReLU, and Dropout
    - Output layer for classification
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        input_size: int = 32,
        hidden_sizes: List[int] = [512, 256, 128],
        dropout: float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            input_size: Input image size (assumes square images)
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate between layers
        """
        super().__init__()

        # Calculate input size (flattened image)
        input_features = in_channels * input_size * input_size

        # Build MLP layers dynamically
        layers = [nn.Flatten()]

        prev_size = input_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through MLP."""
        return self.mlp(x)


def create_image_classifier(
    classifier_type: str,
    num_classes: int,
    in_channels: int = 3,
    input_size: int = 32,
    learning_rate: float = 1e-3,
    class_names: Optional[List[str]] = None,
    dataset_mean: Optional[Tuple[float, ...]] = None,
    dataset_std: Optional[Tuple[float, ...]] = None,
    # CNN-specific parameters
    cnn_num_conv_layers: int = 3,
    cnn_base_channels: int = 32,
    # MLP-specific parameters
    mlp_hidden_sizes: List[int] = [512, 256, 128],
    # Common parameters
    dropout: float = 0.5,
) -> ImageClassifier:
    """
    Factory function to create an image classifier with the specified architecture.

    Args:
        classifier_type: Type of classifier ('cnn' or 'mlp')
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        input_size: Input image size (assumes square images)
        learning_rate: Learning rate for optimizer
        class_names: List of class names for logging
        dataset_mean: Dataset mean for visualization
        dataset_std: Dataset std for visualization
        cnn_num_conv_layers: (CNN only) Number of convolutional blocks
        cnn_base_channels: (CNN only) Base number of channels in first conv layer
        mlp_hidden_sizes: (MLP only) List of hidden layer sizes
        dropout: Dropout rate

    Returns:
        ImageClassifier instance with the specified architecture

    Example:
        >>> # Create a CNN classifier
        >>> model = create_image_classifier(
        ...     classifier_type='cnn',
        ...     num_classes=10,
        ...     in_channels=3,
        ...     input_size=32,
        ...     cnn_num_conv_layers=3,
        ...     cnn_base_channels=32,
        ...     dropout=0.5,
        ...     learning_rate=1e-3
        ... )

        >>> # Create an MLP classifier
        >>> model = create_image_classifier(
        ...     classifier_type='mlp',
        ...     num_classes=10,
        ...     in_channels=3,
        ...     input_size=32,
        ...     mlp_hidden_sizes=[512, 256, 128],
        ...     dropout=0.5,
        ...     learning_rate=1e-3
        ... )
    """
    classifier_type = classifier_type.lower()

    if classifier_type == 'cnn':
        architecture = CNNArchitecture(
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            num_conv_layers=cnn_num_conv_layers,
            base_channels=cnn_base_channels,
            dropout=dropout,
        )
    elif classifier_type == 'mlp':
        architecture = MLPArchitecture(
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            hidden_sizes=mlp_hidden_sizes,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown classifier_type: {classifier_type}. "
            f"Supported types: 'cnn', 'mlp'"
        )

    # Create the classifier with common training logic
    classifier = ImageClassifier(
        architecture=architecture,
        num_classes=num_classes,
        learning_rate=learning_rate,
        class_names=class_names,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
    )

    return classifier

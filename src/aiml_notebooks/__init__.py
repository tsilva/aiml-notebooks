"""
aiml_notebooks - Shared utilities for AI/ML notebooks

This package contains reusable components for multiple notebooks:
- tokenizers: Character-level and word-level tokenizer implementations
- datasets: PyTorch Dataset classes for various tasks
- create_dataset: Factory function for creating datasets with automatic splits
- create_dataloaders: Factory function for creating DataLoaders
- training: PyTorch Lightning training utilities (W&B logger, trainer setup)
- visualization: Plotting and visualization utilities
- losses: Loss functions (VAE, VQ-VAE, GAN)
- evaluation: Metrics and evaluation utilities
- generation: Sampling strategies and generation utilities
- analysis: Latent space analysis and visualization
- preprocessing: Text preprocessing utilities
- models: Common encoder/decoder architectures
"""

__version__ = "0.2.0"

# Core datasets and data utilities
from .datasets import (
    NamesDataset,
    collate_fn,
    create_dataset,
    create_dataloaders,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    MNIST_CLASSES,
    MNIST_MEAN,
    MNIST_STD,
    FASHIONMNIST_CLASSES,
    FASHIONMNIST_MEAN,
    FASHIONMNIST_STD,
)

# Tokenizers
from .tokenizers import CharacterTokenizer, WordTokenizer

# Training utilities
from .logging import log_gradients, log_model_weights, log_gradient_flow
from .training import create_wandb_logger, watch_model, create_trainer, setup_papermill_params

# Visualization
from .visualization import (
    plot_image_grid,
    log_images_to_wandb,
    plot_training_curves,
    plot_confusion_matrix,
    visualize_reconstructions,
    visualize_sample_predictions,
    plot_interpolation,
    plot_model_comparison,
)

# Loss functions
from .losses import (
    VAELoss,
    VQVAELoss,
    GANLoss,
    perplexity,
)

# Evaluation
from .evaluation import (
    compute_classification_metrics,
    compute_per_class_metrics,
    print_classification_report,
    compute_top_k_accuracy,
    compute_confusion_matrix,
    compare_models,
    print_model_comparison,
    calculate_reconstruction_error,
)

# Generation
from .generation import (
    sample_with_temperature,
    sample_top_k,
    sample_nucleus,
    generate_text,
    interpolate_latents,
    spherical_interpolation,
    latent_arithmetic,
)

# Analysis
from .analysis import (
    extract_latent_representations,
    reduce_dimensions,
    visualize_latent_space,
    analyze_latent_clusters,
    latent_traversal,
    compute_latent_statistics,
)

# Preprocessing
from .preprocessing import (
    TextPreprocessor,
    simple_tokenize,
    tokenize_with_punctuation,
    build_vocabulary_from_texts,
    encode_text,
    decode_text,
    pad_sequence,
    batch_encode_texts,
    remove_stopwords,
    normalize_text,
)

# Models
from .models import (
    MLPEncoder,
    ConvEncoder,
    RNNEncoder,
    MLPDecoder,
    ConvDecoder,
    RNNDecoder,
    VectorQuantizer,
)

__all__ = [
    # Datasets
    "NamesDataset",
    "collate_fn",
    "create_dataset",
    "create_dataloaders",
    "CIFAR10_CLASSES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "MNIST_CLASSES",
    "MNIST_MEAN",
    "MNIST_STD",
    "FASHIONMNIST_CLASSES",
    "FASHIONMNIST_MEAN",
    "FASHIONMNIST_STD",
    # Tokenizers
    "CharacterTokenizer",
    "WordTokenizer",
    # Training
    "log_gradients",
    "log_model_weights",
    "log_gradient_flow",
    "create_wandb_logger",
    "watch_model",
    "create_trainer",
    "setup_papermill_params",
    # Visualization
    "plot_image_grid",
    "log_images_to_wandb",
    "plot_training_curves",
    "plot_confusion_matrix",
    "visualize_reconstructions",
    "visualize_sample_predictions",
    "plot_interpolation",
    "plot_model_comparison",
    # Losses
    "VAELoss",
    "VQVAELoss",
    "GANLoss",
    "perplexity",
    # Evaluation
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "print_classification_report",
    "compute_top_k_accuracy",
    "compute_confusion_matrix",
    "compare_models",
    "print_model_comparison",
    "calculate_reconstruction_error",
    # Generation
    "sample_with_temperature",
    "sample_top_k",
    "sample_nucleus",
    "generate_text",
    "interpolate_latents",
    "spherical_interpolation",
    "latent_arithmetic",
    # Analysis
    "extract_latent_representations",
    "reduce_dimensions",
    "visualize_latent_space",
    "analyze_latent_clusters",
    "latent_traversal",
    "compute_latent_statistics",
    # Preprocessing
    "TextPreprocessor",
    "simple_tokenize",
    "tokenize_with_punctuation",
    "build_vocabulary_from_texts",
    "encode_text",
    "decode_text",
    "pad_sequence",
    "batch_encode_texts",
    "remove_stopwords",
    "normalize_text",
    # Models
    "MLPEncoder",
    "ConvEncoder",
    "RNNEncoder",
    "MLPDecoder",
    "ConvDecoder",
    "RNNDecoder",
    "VectorQuantizer",
]

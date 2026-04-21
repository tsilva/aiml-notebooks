"""Shared utilities for the AI/ML notebooks.

The package exports a broad convenience API, but many notebooks only need a
small subset of it. Keep imports lazy so lightweight helpers such as
`get_device` and `set_seed` still work in remote kernels that do not have every
optional dependency installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

_SYMBOL_MODULES = {
    # Datasets
    "NamesDataset": "datasets",
    "collate_fn": "datasets",
    "create_dataset": "datasets",
    "create_dataloaders": "datasets",
    "create_seq2seq_collate_fn": "datasets",
    "create_classification_collate_fn": "datasets",
    "create_variable_length_collate_fn": "datasets",
    "get_dataset_config": "datasets",
    "CIFAR10_CLASSES": "datasets",
    "CIFAR10_MEAN": "datasets",
    "CIFAR10_STD": "datasets",
    "MNIST_CLASSES": "datasets",
    "MNIST_MEAN": "datasets",
    "MNIST_STD": "datasets",
    "FASHIONMNIST_CLASSES": "datasets",
    "FASHIONMNIST_MEAN": "datasets",
    "FASHIONMNIST_STD": "datasets",
    # DataLoader benchmarking
    "DataLoaderConfig": "dataloader_benchmark",
    "BenchmarkResult": "dataloader_benchmark",
    "benchmark_dataloader": "dataloader_benchmark",
    "find_optimal_dataloader_config": "dataloader_benchmark",
    "quick_benchmark_dataloader": "dataloader_benchmark",
    # Tokenizers
    "CharacterTokenizer": "tokenizers",
    "WordTokenizer": "tokenizers",
    # General utilities
    "get_device": "utils",
    "set_seed": "utils",
    "count_parameters": "utils",
    "print_model_summary": "utils",
    # Environment checks
    "PackageRequirement": "environment",
    "check_package_requirements": "environment",
    "environment_report": "environment",
    "get_cuda_environment_info": "environment",
    "get_python_environment_info": "environment",
    "print_environment_report": "environment",
    "require_environment": "environment",
    # Notebook bootstrap
    "bootstrap_notebook": "bootstrap",
    "enable_autoreload": "bootstrap",
    # Hardware detection and configuration
    "HardwareConfig": "hardware",
    "detect_hardware": "hardware",
    "check_flash_attention": "hardware",
    "configure_cuda_optimizations": "hardware",
    "apply_torch_compile": "hardware",
    "flash_attention_func": "hardware",
    "auto_optimizer": "hardware",
    "auto_precision": "hardware",
    "auto_attention_backend": "hardware",
    "auto_compile_model": "hardware",
    "auto_pin_memory": "hardware",
    "auto_num_workers": "hardware",
    "auto_batch_size": "hardware",
    "auto_gradient_accumulation_steps": "hardware",
    "auto_learning_rate": "hardware",
    "estimate_training_memory_gb": "hardware",
    "print_memory_estimate": "hardware",
    "find_max_batch_size": "hardware",
    "free_memory": "hardware",
    "show_memory_usage": "hardware",
    # Training and logging
    "log_gradients": "logging",
    "log_model_weights": "logging",
    "log_gradient_flow": "logging",
    "create_wandb_logger": "training",
    "watch_model": "training",
    "create_trainer": "training",
    "setup_papermill_params": "training",
    "train_epoch_classification": "training",
    "evaluate_classification": "training",
    "train_epoch_seq2seq": "training",
    "evaluate_seq2seq": "training",
    "TrainingHistory": "training",
    # Visualization
    "imshow_normalized": "visualization",
    "show_image_grid_normalized": "visualization",
    "plot_image_grid": "visualization",
    "log_images_to_wandb": "visualization",
    "plot_training_curves": "visualization",
    "plot_confusion_matrix": "visualization",
    "visualize_reconstructions": "visualization",
    "visualize_sample_predictions": "visualization",
    "plot_interpolation": "visualization",
    "plot_model_comparison": "visualization",
    # Losses
    "VAELoss": "losses",
    "VQVAELoss": "losses",
    "GANLoss": "losses",
    "perplexity": "losses",
    # Evaluation
    "compute_classification_metrics": "evaluation",
    "compute_per_class_metrics": "evaluation",
    "print_classification_report": "evaluation",
    "compute_top_k_accuracy": "evaluation",
    "compute_confusion_matrix": "evaluation",
    "compare_models": "evaluation",
    "print_model_comparison": "evaluation",
    "calculate_reconstruction_error": "evaluation",
    # Generation
    "sample_with_temperature": "generation",
    "sample_top_k": "generation",
    "sample_nucleus": "generation",
    "generate_text": "generation",
    "interpolate_latents": "generation",
    "spherical_interpolation": "generation",
    "latent_arithmetic": "generation",
    # Analysis
    "extract_latent_representations": "analysis",
    "reduce_dimensions": "analysis",
    "visualize_latent_space": "analysis",
    "analyze_latent_clusters": "analysis",
    "latent_traversal": "analysis",
    "compute_latent_statistics": "analysis",
    # Preprocessing
    "TextPreprocessor": "preprocessing",
    "simple_tokenize": "preprocessing",
    "tokenize_with_punctuation": "preprocessing",
    "build_vocabulary_from_texts": "preprocessing",
    "encode_text": "preprocessing",
    "decode_text": "preprocessing",
    "pad_sequence": "preprocessing",
    "batch_encode_texts": "preprocessing",
    "remove_stopwords": "preprocessing",
    "normalize_text": "preprocessing",
    # Models
    "MLPEncoder": "models.encoders",
    "ConvEncoder": "models.encoders",
    "RNNEncoder": "models.encoders",
    "MLPDecoder": "models.decoders",
    "ConvDecoder": "models.decoders",
    "RNNDecoder": "models.decoders",
    "VectorQuantizer": "models.vector_quantizer",
    "ImageClassifier": "models.image_classifiers",
    "CNNArchitecture": "models.image_classifiers",
    "MLPArchitecture": "models.image_classifiers",
    "create_image_classifier": "models.image_classifiers",
    # Augmentation
    "TextNoiser": "augmentation",
    "ImageNoiser": "augmentation",
    "RandomNoise": "augmentation",
    "add_gaussian_noise_numpy": "augmentation",
    "random_dropout_pixels": "augmentation",
    "add_random_occlusion": "augmentation",
    # Image utilities
    "normalize_image": "image_utils",
    "denormalize_image": "image_utils",
    "create_standard_transforms": "image_utils",
    "create_denoising_transforms": "image_utils",
    "prepare_for_visualization": "image_utils",
    "batch_normalize": "image_utils",
    "batch_denormalize": "image_utils",
    "get_dataset_stats": "image_utils",
    # Positional encoding
    "SinusoidalPositionalEncoding": "positional_encoding",
    "LearnablePositionalEmbedding": "positional_encoding",
    "RelativePositionalEncoding": "positional_encoding",
    "create_causal_mask": "positional_encoding",
    "create_padding_mask": "positional_encoding",
    "create_attention_mask": "positional_encoding",
    "get_positional_encoding": "positional_encoding",
}

__all__ = list(_SYMBOL_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_SYMBOL_MODULES])

"""
General utility functions for notebooks.

Common helper functions used across multiple notebooks.
"""

import torch
import numpy as np
import random


def get_device(
    verbose: bool = True,
    prefer_cpu: bool = False,
    show_mps_warning: bool = True
) -> torch.device:
    """
    Get best available device with configurable strategy.

    Supports two device selection strategies:
    1. Standard (prefer_cpu=False): MPS > CUDA > CPU
    2. Safe mode (prefer_cpu=True): CUDA > CPU (avoids MPS)

    Safe mode is recommended for notebooks using:
    - nn.Transformer or nested tensor operations
    - Advanced PyTorch operations with known MPS issues

    For MPS-compatible notebooks, use standard mode for best performance.

    Args:
        verbose: Whether to print device information
        prefer_cpu: If True, avoid MPS even if available (safe mode for Transformers)
        show_mps_warning: If True and MPS is available but not used, show fallback instructions

    Returns:
        torch.device object

    Examples:
        >>> # Standard usage (MPS-compatible notebooks)
        >>> device = get_device()
        Using MPS (Metal Performance Shaders) for GPU acceleration

        >>> # Safe mode (Transformer notebooks)
        >>> device = get_device(prefer_cpu=True)
        Using CPU
        Note: MPS is available but not used due to compatibility issues
        To use MPS with CPU fallback, run: PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab

        >>> # Quiet mode
        >>> device = get_device(verbose=False)
    """
    if not prefer_cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
            if show_mps_warning and prefer_cpu and torch.backends.mps.is_available():
                print("Note: MPS is available but not used due to compatibility issues")
                print("To use MPS with CPU fallback, run: PYTORCH_ENABLE_MPS_FALLBACK=1 jupyter lab")

    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters

    Example:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture and parameters.

    Args:
        model: PyTorch model

    Example:
        >>> print_model_summary(model)
        Model: MyModel
        Total parameters: 1,234,567
        Trainable parameters: 1,234,567
        Non-trainable parameters: 0
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")


def load_pretrained_gpt2_weights(
    model: torch.nn.Module,
    model_name: str = 'openai-community/gpt2',
    n_layer: int = 12,
    verbose: bool = True
) -> torch.nn.Module:
    """
    Load pretrained weights from HuggingFace GPT-2 into a custom GPT-2 model.
    
    This function maps weights from the HuggingFace GPT2LMHeadModel to a custom
    implementation that follows the standard GPT-2 architecture with:
    - token_embedding_table: nn.Embedding for token embeddings
    - position_embedding_table: nn.Embedding for position embeddings  
    - blocks: nn.ModuleList of transformer blocks, each containing:
        - ln1, ln2: LayerNorm layers
        - head: Multi-head attention with W_qkv and W_o Linear layers
        - ffn: FeedForward with ffn1 and ffn2 Linear layers
    - ln_final: Final layer norm
    - lm_head: Language model head (typically weight-tied to token_embedding_table)
    
    Args:
        model: Custom Transformer model instance with the expected structure
        model_name: HuggingFace model identifier (default: 'openai-community/gpt2')
        n_layer: Number of transformer layers in the model
        verbose: Whether to print progress messages
    
    Returns:
        The model with loaded weights
        
    Note:
        HuggingFace uses Conv1D for attention and FFN layers which store weights
        transposed compared to nn.Linear. This function handles the transposition.
        
    Example:
        >>> model = Transformer()  # Your custom GPT-2 implementation
        >>> model = load_pretrained_gpt2_weights(model, n_layer=12)
        >>> model.eval()
        >>> output = model.generate("Hello, world")
    """
    from transformers import GPT2LMHeadModel
    
    if verbose:
        print(f"Loading pretrained weights from {model_name}...")
    
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_state_dict = hf_model.state_dict()
    
    our_state_dict = model.state_dict()
    
    # 1. Token and position embeddings
    our_state_dict['token_embedding_table.weight'].copy_(hf_state_dict['transformer.wte.weight'])
    our_state_dict['position_embedding_table.weight'].copy_(hf_state_dict['transformer.wpe.weight'])
    
    # 2. Final layer norm
    our_state_dict['ln_final.weight'].copy_(hf_state_dict['transformer.ln_f.weight'])
    our_state_dict['ln_final.bias'].copy_(hf_state_dict['transformer.ln_f.bias'])
    
    # 3. Load weights for each transformer block
    for i in range(n_layer):
        # Layer norm 1 (pre-attention)
        our_state_dict[f'blocks.{i}.ln1.weight'].copy_(hf_state_dict[f'transformer.h.{i}.ln_1.weight'])
        our_state_dict[f'blocks.{i}.ln1.bias'].copy_(hf_state_dict[f'transformer.h.{i}.ln_1.bias'])
        
        # Attention QKV projection (combined)
        # HuggingFace Conv1D stores as (768, 2304), PyTorch Linear expects (2304, 768)
        hf_qkv_weight = hf_state_dict[f'transformer.h.{i}.attn.c_attn.weight']
        hf_qkv_bias = hf_state_dict[f'transformer.h.{i}.attn.c_attn.bias']
        our_state_dict[f'blocks.{i}.head.W_qkv.weight'].copy_(hf_qkv_weight.T)
        our_state_dict[f'blocks.{i}.head.W_qkv.bias'].copy_(hf_qkv_bias)
        
        # Attention output projection
        hf_o_weight = hf_state_dict[f'transformer.h.{i}.attn.c_proj.weight']
        hf_o_bias = hf_state_dict[f'transformer.h.{i}.attn.c_proj.bias']
        our_state_dict[f'blocks.{i}.head.W_o.weight'].copy_(hf_o_weight.T)
        our_state_dict[f'blocks.{i}.head.W_o.bias'].copy_(hf_o_bias)
        
        # Layer norm 2 (pre-FFN)
        our_state_dict[f'blocks.{i}.ln2.weight'].copy_(hf_state_dict[f'transformer.h.{i}.ln_2.weight'])
        our_state_dict[f'blocks.{i}.ln2.bias'].copy_(hf_state_dict[f'transformer.h.{i}.ln_2.bias'])
        
        # FFN first layer (expansion)
        hf_fc_weight = hf_state_dict[f'transformer.h.{i}.mlp.c_fc.weight']
        hf_fc_bias = hf_state_dict[f'transformer.h.{i}.mlp.c_fc.bias']
        our_state_dict[f'blocks.{i}.ffn.ffn1.weight'].copy_(hf_fc_weight.T)
        our_state_dict[f'blocks.{i}.ffn.ffn1.bias'].copy_(hf_fc_bias)
        
        # FFN second layer (projection)
        hf_proj_weight = hf_state_dict[f'transformer.h.{i}.mlp.c_proj.weight']
        hf_proj_bias = hf_state_dict[f'transformer.h.{i}.mlp.c_proj.bias']
        our_state_dict[f'blocks.{i}.ffn.ffn2.weight'].copy_(hf_proj_weight.T)
        our_state_dict[f'blocks.{i}.ffn.ffn2.bias'].copy_(hf_proj_bias)
    
    if verbose:
        print(f"✓ Successfully loaded {n_layer} transformer blocks")
        print("✓ Weight tying preserved (lm_head shares weights with token embeddings)")
    
    return model

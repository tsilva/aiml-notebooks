"""
Hardware detection and configuration utilities.

Automatically detect hardware capabilities and configure optimal training settings
for NVIDIA GPUs (CUDA), Apple Silicon (MPS), or CPU.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import psutil
import torch


@dataclass
class HardwareConfig:
    """
    Hardware configuration with optimal training settings.

    Attributes:
        device_type: Type of device ('cuda', 'mps', or 'cpu')
        device: PyTorch device object
        device_name: Human-readable device name
        memory_gb: Available memory in GB
        precision: Lightning precision string ('bf16-mixed', '16-mixed', '32-true')
        batch_size: Recommended batch size based on available memory
        pin_memory: Whether to pin memory for DataLoader
        use_compile: Whether torch.compile is recommended
        use_fused_optimizer: Whether fused optimizers are available
        use_flash_attention: Whether Flash Attention can be used
        gradient_accumulation_steps: Recommended gradient accumulation steps
        compute_capability: CUDA compute capability (None if not CUDA)
    """
    device_type: Literal['cuda', 'mps', 'cpu']
    device: torch.device
    device_name: str
    memory_gb: float
    precision: Literal['bf16-mixed', '16-mixed', '32-true']
    batch_size: int
    pin_memory: bool
    use_compile: bool
    use_fused_optimizer: bool
    use_flash_attention: bool
    gradient_accumulation_steps: int
    compute_capability: Optional[tuple[int, int]] = None

    def __str__(self) -> str:
        """Human-readable configuration summary."""
        lines = [
            f"Hardware Configuration:",
            f"  Device: {self.device_type.upper()} - {self.device_name}",
            f"  Memory: {self.memory_gb:.1f} GB",
            f"  Precision: {self.precision}",
            f"  Batch size: {self.batch_size}",
            f"  Pin memory: {self.pin_memory}",
        ]

        if self.device_type == 'cuda' and self.compute_capability:
            lines.insert(2, f"  Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}")

        optimizations = []
        if self.use_compile:
            optimizations.append("torch.compile")
        if self.use_fused_optimizer:
            optimizations.append("Fused optimizer")
        if self.use_flash_attention:
            optimizations.append("Flash Attention")
        if self.gradient_accumulation_steps > 1:
            optimizations.append(f"Gradient accumulation (×{self.gradient_accumulation_steps})")

        if optimizations:
            lines.append(f"  Optimizations: {', '.join(optimizations)}")

        return "\n".join(lines)


def detect_hardware(
    base_batch_size: int = 64,
    enable_tf32: bool = True,
    enable_mps_fallback: bool = True,
    verbose: bool = True
) -> HardwareConfig:
    """
    Detect hardware and return optimal training configuration.

    Automatically detects CUDA GPUs, Apple Silicon (MPS), or falls back to CPU,
    then configures optimal settings for each platform.

    CUDA (NVIDIA GPUs):
    - Uses BF16 on Ampere+ (compute capability >= 8.0), FP16 on older GPUs
    - Enables TF32 for additional speedup on Ampere+
    - Large batch sizes for datacenter GPUs (40GB+ memory)
    - Enables fused optimizers, torch.compile, Flash Attention support
    - Pin memory for faster CPU→GPU transfers

    MPS (Apple Silicon):
    - Uses FP16 mixed precision (BF16 not well-supported)
    - Adaptive batch sizing based on unified memory (8-128GB)
    - Gradient accumulation for low-memory devices
    - No pin memory (unified memory architecture)
    - CPU fallback enabled for unsupported operations

    CPU:
    - Full FP32 precision
    - Conservative batch sizes
    - No special optimizations

    Args:
        base_batch_size: Base batch size to scale from (default: 64)
        enable_tf32: Enable TF32 on Ampere+ GPUs (default: True)
        enable_mps_fallback: Enable CPU fallback for MPS (default: True)
        verbose: Print detection results (default: True)

    Returns:
        HardwareConfig object with optimal settings

    Example:
        >>> config = detect_hardware()
        🚀 NVIDIA GPU Detected: NVIDIA A100-SXM4-80GB
           Compute Capability: 8.0
           Memory: 80.0 GB
           ✓ Ampere+ architecture detected
           ✓ TF32 enabled for matmul and cuDNN

        📊 Final Configuration:
           Device: cuda
           Precision: bf16-mixed
           Batch size: 256
           Mixed precision: ✓
           Flash Attention: ✓ (if installed)
           torch.compile: ✓
           Fused optimizer: ✓

        >>> # Use configuration
        >>> model = MyModel().to(config.device)
        >>> trainer = L.Trainer(
        ...     precision=config.precision,
        ...     accelerator='auto',
        ...     devices=1
        ... )
    """
    # CUDA Detection
    if torch.cuda.is_available():
        device_type = 'cuda'
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if verbose:
            print(f"🚀 NVIDIA GPU Detected: {device_name}")
            print(f"   Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            print(f"   Memory: {memory_gb:.1f} GB")

        # Ampere or newer (A100, H100, RTX 30XX/40XX)
        if compute_capability[0] >= 8:
            if verbose:
                print("   ✓ Ampere+ architecture detected")
            precision = 'bf16-mixed'

            # Enable TF32 for additional speedup
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if verbose:
                    print("   ✓ TF32 enabled for matmul and cuDNN")
        else:
            if verbose:
                print("   ℹ Pre-Ampere architecture (using FP16 instead of BF16)")
            precision = '16-mixed'

        # Batch size scaling based on memory
        if memory_gb >= 40:  # Datacenter GPUs
            batch_size = base_batch_size * 4  # 256 for base_batch_size=64
        elif memory_gb >= 24:  # RTX 3090/4090, A5000
            batch_size = base_batch_size * 2  # 128
        elif memory_gb >= 12:  # RTX 3060, 4060 Ti
            batch_size = base_batch_size  # 64
        else:  # <12GB
            batch_size = base_batch_size // 2  # 32

        pin_memory = True
        use_compile = True
        use_fused_optimizer = True
        use_flash_attention = True  # Will check availability separately
        gradient_accumulation_steps = 1

    # Apple Silicon (MPS) Detection
    elif torch.backends.mps.is_available():
        import platform

        device_type = 'mps'
        device = torch.device('mps')
        device_name = f"Apple Silicon ({platform.machine()})"
        compute_capability = None
        memory_gb = psutil.virtual_memory().total / 1e9

        if verbose:
            print(f"🍎 Apple Silicon Detected")
            print(f"   Architecture: {platform.machine()}")
            print(f"   macOS: {platform.mac_ver()[0]}")
            print(f"   PyTorch: {torch.__version__}")
            print(f"   Unified Memory: {memory_gb:.1f} GB")

        # Adaptive batch sizing for unified memory
        if memory_gb >= 32:  # M1/M2 Max, M1 Ultra, M3 Pro+
            batch_size = base_batch_size  # 64
            gradient_accumulation_steps = 1
            if verbose:
                print(f"   → High memory config: batch_size={batch_size}")
        elif memory_gb >= 16:  # M1/M2 Pro
            batch_size = base_batch_size // 2  # 32
            gradient_accumulation_steps = 1
            if verbose:
                print(f"   → Medium memory config: batch_size={batch_size}")
        else:  # Base M1/M2
            batch_size = base_batch_size // 4  # 16
            gradient_accumulation_steps = 4
            if verbose:
                print(f"   → Low memory config: batch_size={batch_size} with gradient accumulation")

        precision = '16-mixed'  # FP16 for MPS
        pin_memory = False  # No benefit with unified memory
        use_compile = False  # Limited MPS support
        use_fused_optimizer = False  # Not available on MPS
        use_flash_attention = False  # Not available on MPS

        # Enable MPS fallback for unsupported operations
        if enable_mps_fallback:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            if verbose:
                print("   ✓ CPU fallback enabled for unsupported MPS operations")

    # CPU Fallback
    else:
        device_type = 'cpu'
        device = torch.device('cpu')
        device_name = "CPU"
        compute_capability = None
        memory_gb = psutil.virtual_memory().total / 1e9

        if verbose:
            print("⚠️  CPU Mode (no GPU acceleration detected)")
            print("   Training will be slow. Consider using a GPU for better performance.")

        batch_size = base_batch_size // 2  # 32
        precision = '32-true'
        pin_memory = False
        use_compile = False
        use_fused_optimizer = False
        use_flash_attention = False
        gradient_accumulation_steps = 1

    # Create configuration
    config = HardwareConfig(
        device_type=device_type,
        device=device,
        device_name=device_name,
        memory_gb=memory_gb,
        precision=precision,
        batch_size=batch_size,
        pin_memory=pin_memory,
        use_compile=use_compile,
        use_fused_optimizer=use_fused_optimizer,
        use_flash_attention=use_flash_attention,
        gradient_accumulation_steps=gradient_accumulation_steps,
        compute_capability=compute_capability
    )

    if verbose:
        print(f"\n📊 Final Configuration:")
        print(f"   Device: {device}")
        print(f"   Precision: {precision}")
        print(f"   Batch size: {batch_size}")
        print(f"   Mixed precision: {'✓' if precision != '32-true' else '✗'}")
        print(f"   Flash Attention: {'✓ (if installed)' if use_flash_attention else '✗'}")
        print(f"   torch.compile: {'✓' if use_compile else '✗'}")
        print(f"   Fused optimizer: {'✓' if use_fused_optimizer else '✗'}")
        if gradient_accumulation_steps > 1:
            print(f"   Gradient accumulation: ✓ ({gradient_accumulation_steps} steps)")

    return config


def check_flash_attention() -> bool:
    """
    Check if Flash Attention 2 is available.

    Flash Attention 2 provides 2-4x speedup for attention computation
    on CUDA GPUs with reduced memory usage. Only works on CUDA.

    Returns:
        True if Flash Attention 2 is installed and usable, False otherwise

    Example:
        >>> if check_flash_attention():
        ...     print("✓ Flash Attention 2 is available")
        ...     from flash_attn import flash_attn_func
        ... else:
        ...     print("Flash Attention not available")
        ...     print("Install with: pip install flash-attn")
    """
    try:
        from flash_attn import flash_attn_func
        return True
    except ImportError:
        return False


def configure_cuda_optimizations(
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True
) -> None:
    """
    Configure CUDA-specific optimizations.

    Enables performance optimizations for NVIDIA GPUs:
    - TF32: Faster matmul on Ampere+ GPUs with minimal accuracy loss
    - cuDNN benchmark: Auto-tune cuDNN kernels for your input sizes

    Args:
        enable_tf32: Enable TF32 for matmul and cuDNN (default: True)
        enable_cudnn_benchmark: Enable cuDNN benchmarking (default: True)

    Example:
        >>> configure_cuda_optimizations()
        ✓ TF32 enabled
        ✓ cuDNN benchmarking enabled
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping CUDA optimizations")
        return

    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled")

    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmarking enabled")


def apply_torch_compile(
    model: torch.nn.Module,
    mode: str = 'max-autotune',
    verbose: bool = True,
    fallback_on_error: bool = True
) -> torch.nn.Module:
    """
    Apply torch.compile with automatic version detection and error handling.

    torch.compile (PyTorch 2.0+) provides 1.5-2x speedup through JIT compilation.
    Works best on CUDA, limited support on MPS, not recommended for CPU.

    Args:
        model: PyTorch model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            - 'default': Balanced compilation
            - 'reduce-overhead': Minimize Python overhead
            - 'max-autotune': Maximum optimization (best for production)
        verbose: Print compilation status (default: True)
        fallback_on_error: Return original model if compilation fails (default: True)

    Returns:
        Compiled model if successful, original model if compilation fails/unsupported

    Example:
        >>> model = MyModel()
        >>> model = apply_torch_compile(model)
        Applying torch.compile (may take a minute on first run)...
        ✓ torch.compile applied with mode='max-autotune'

        >>> # Or with error handling disabled
        >>> model = apply_torch_compile(model, fallback_on_error=False)
    """
    # Check PyTorch version
    pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])

    if pytorch_version < (2, 0):
        if verbose:
            print(f"ℹ torch.compile requires PyTorch 2.0+, found {torch.__version__}")
            print("  Skipping compilation")
        return model

    try:
        if verbose:
            print(f"Applying torch.compile with mode='{mode}'...")
            print("  (This may take a minute on first run)")

        compiled_model = torch.compile(model, mode=mode)

        if verbose:
            print(f"✓ torch.compile applied successfully")

        return compiled_model

    except Exception as e:
        if fallback_on_error:
            if verbose:
                print(f"⚠ torch.compile failed: {e}")
                print("  Continuing without compilation")
            return model
        else:
            raise


def flash_attention_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    softmax_scale: Optional[float] = None
) -> torch.Tensor:
    """
    Unified flash attention interface with automatic fallback.

    Automatically uses Flash Attention 2 if available, otherwise falls back
    to optimized PyTorch implementation. Provides a single interface for both.

    Expected tensor shapes:
    - q, k, v: (batch_size, seq_len, num_heads, head_dim)

    Args:
        q: Query tensor (B, T, H, D)
        k: Key tensor (B, T, H, D)
        v: Value tensor (B, T, H, D)
        dropout_p: Dropout probability (default: 0.0)
        causal: Apply causal mask (default: False)
        softmax_scale: Scale for softmax (default: 1/sqrt(head_dim))

    Returns:
        Output tensor (B, T, H, D)

    Example:
        >>> # Works the same regardless of whether Flash Attention is installed
        >>> q = torch.randn(2, 128, 8, 64)  # (B, T, H, D)
        >>> k = torch.randn(2, 128, 8, 64)
        >>> v = torch.randn(2, 128, 8, 64)
        >>> out = flash_attention_func(q, k, v, causal=True)
        >>> out.shape
        torch.Size([2, 128, 8, 64])
    """
    if check_flash_attention():
        # Use Flash Attention 2
        from flash_attn import flash_attn_func as flash_attn_func_impl
        return flash_attn_func_impl(q, k, v, dropout_p=dropout_p, causal=causal, softmax_scale=softmax_scale)

    else:
        # Fallback to standard PyTorch implementation
        B, T, H, D = q.shape

        # Transpose to (B, H, T, D) for standard attention
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scale = softmax_scale if softmax_scale is not None else (D ** -0.5)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply causal mask if requested
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask, float('-inf'))

        # Softmax and dropout
        att = torch.nn.functional.softmax(att, dim=-1)
        if dropout_p > 0.0:
            att = torch.nn.functional.dropout(att, p=dropout_p)

        # Apply attention to values
        out = att @ v  # (B, H, T, D)

        # Transpose back to (B, T, H, D)
        out = out.transpose(1, 2).contiguous()

        return out

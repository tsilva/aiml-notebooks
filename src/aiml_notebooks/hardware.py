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


def auto_optimizer(hw_config: Optional[HardwareConfig] = None) -> Literal['adamw', 'adamw-fused']:
    """
    Auto-detect best optimizer based on hardware.

    Returns 'adamw-fused' for CUDA GPUs (2x faster), 'adamw' for MPS/CPU.

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        'adamw-fused' if fused optimizer is supported, 'adamw' otherwise

    Example:
        >>> hw = detect_hardware()
        >>> optimizer_name = auto_optimizer(hw)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(optimizer_name == 'adamw-fused'))
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return 'adamw-fused' if hw_config.use_fused_optimizer else 'adamw'


def auto_precision(hw_config: Optional[HardwareConfig] = None) -> Literal['bf16-mixed', '16-mixed', '32-true']:
    """
    Auto-detect best precision based on hardware.

    Returns:
        - 'bf16-mixed' for CUDA Ampere+ GPUs (A100, H100, RTX 30XX/40XX)
        - '16-mixed' for older CUDA GPUs and Apple Silicon (MPS)
        - '32-true' for CPU

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        Optimal precision string for Lightning Trainer

    Example:
        >>> hw = detect_hardware()
        >>> precision = auto_precision(hw)
        >>> trainer = L.Trainer(precision=precision, ...)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return hw_config.precision


def auto_attention_backend(hw_config: Optional[HardwareConfig] = None) -> Literal['flash', 'custom']:
    """
    Auto-detect best attention backend based on hardware and available packages.

    Flash Attention 2 provides 2-4x speedup on CUDA GPUs with compute capability >= 8.0.
    Falls back to custom implementation for MPS/CPU or when flash-attn is not installed.

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        'flash' if Flash Attention is available and supported, 'custom' otherwise

    Example:
        >>> hw = detect_hardware()
        >>> backend = auto_attention_backend(hw)
        >>> model = GPTModel(attention_backend=backend)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    # Check both hardware support and package availability
    if hw_config.use_flash_attention and check_flash_attention():
        return 'flash'
    else:
        return 'custom'


def auto_compile_model(
    hw_config: Optional[HardwareConfig] = None,
    mode: str = 'default'
) -> Literal['default', 'reduce-overhead', 'max-autotune', False]:
    """
    Auto-detect whether to use torch.compile based on hardware.

    torch.compile provides 1.5-2x speedup on CUDA GPUs. Limited support on MPS,
    not recommended for CPU.

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.
        mode: Compilation mode to return if supported ('default', 'reduce-overhead', 'max-autotune')

    Returns:
        Compilation mode string if supported, False otherwise

    Example:
        >>> hw = detect_hardware()
        >>> compile_setting = auto_compile_model(hw, mode='max-autotune')
        >>> if compile_setting:
        ...     model = apply_torch_compile(model, mode=compile_setting)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return mode if hw_config.use_compile else False


def auto_pin_memory(hw_config: Optional[HardwareConfig] = None) -> bool:
    """
    Auto-detect whether to use pinned memory for DataLoader.

    Pinned memory provides faster CPU→GPU transfers on CUDA. Not beneficial
    for MPS (unified memory) or CPU.

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        True if pinned memory should be used, False otherwise

    Example:
        >>> hw = detect_hardware()
        >>> pin_memory = auto_pin_memory(hw)
        >>> dataloader = DataLoader(dataset, batch_size=32, pin_memory=pin_memory)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return hw_config.pin_memory


def auto_num_workers(hw_config: Optional[HardwareConfig] = None, max_workers: int = 16, streaming: bool = False) -> int:
    """
    Auto-detect optimal number of DataLoader workers based on hardware.

    Returns optimal worker count for data loading:
    - CUDA: 4-12 workers for parallel data loading (more for streaming)
    - MPS: 0 workers (multiprocessing issues on some PyTorch/MPS versions)
    - CPU: 2 workers for basic parallelism

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.
        max_workers: Maximum number of workers to return (default: 16)
        streaming: Whether using streaming dataset (default: False)

    Returns:
        Optimal number of DataLoader workers

    Example:
        >>> hw = detect_hardware()
        >>> num_workers = auto_num_workers(hw)
        >>> dataloader = DataLoader(dataset, batch_size=32, num_workers=num_workers)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    if hw_config.device_type == 'cuda':
        # CUDA benefits from parallel data loading
        # Use more workers for streaming datasets (more I/O bound)
        if streaming:
            return min(max_workers, 12)
        else:
            return min(max_workers, 8)
    elif hw_config.device_type == 'mps':
        # MPS can have issues with multiprocessing workers in some PyTorch versions
        # Using 0 workers (main process) is safest
        return 0
    else:  # CPU
        # CPU can benefit from some parallelism but keep it modest
        return min(max_workers, 2)


def auto_persistent_workers(hw_config: Optional[HardwareConfig] = None, num_workers: Optional[int] = None) -> bool:
    """
    Auto-detect whether to use persistent workers in DataLoader.

    Persistent workers keep worker processes alive between epochs, avoiding
    startup overhead. Only works when num_workers > 0.

    Returns:
        - True for CUDA with num_workers > 0 (faster epoch transitions)
        - False for MPS (num_workers=0) or num_workers=0
        - False for CPU (minimal benefit)

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.
        num_workers: Number of workers (if None, will auto-detect)

    Returns:
        True if persistent workers should be used, False otherwise

    Example:
        >>> hw = detect_hardware()
        >>> num_workers = auto_num_workers(hw)
        >>> persistent_workers = auto_persistent_workers(hw, num_workers)
        >>> dataloader = DataLoader(dataset, num_workers=num_workers,
        ...                         persistent_workers=persistent_workers)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    if num_workers is None:
        num_workers = auto_num_workers(hw_config)

    # Persistent workers only work with num_workers > 0
    if num_workers == 0:
        return False

    # Enable for CUDA (faster epoch transitions)
    if hw_config.device_type == 'cuda':
        return True

    # Disable for MPS and CPU (minimal benefit or not supported)
    return False


def auto_prefetch_factor(hw_config: Optional[HardwareConfig] = None, num_workers: Optional[int] = None, streaming: bool = False) -> Optional[int]:
    """
    Auto-detect optimal prefetch factor for DataLoader.

    Prefetch factor controls how many batches each worker pre-loads.
    Higher values use more memory but reduce waiting time.
    Only applies when num_workers > 0.

    Returns:
        - 4 for CUDA with fast GPUs (overlap compute and I/O)
        - 2 for streaming datasets (more I/O intensive)
        - None for num_workers=0 (parameter not applicable)

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.
        num_workers: Number of workers (if None, will auto-detect)
        streaming: Whether using streaming dataset (default: False)

    Returns:
        Prefetch factor (int) or None if num_workers=0

    Example:
        >>> hw = detect_hardware()
        >>> num_workers = auto_num_workers(hw)
        >>> prefetch_factor = auto_prefetch_factor(hw, num_workers)
        >>> if prefetch_factor is not None:
        ...     dataloader = DataLoader(dataset, num_workers=num_workers,
        ...                            prefetch_factor=prefetch_factor)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    if num_workers is None:
        num_workers = auto_num_workers(hw_config, streaming=streaming)

    # Prefetch factor only applies when num_workers > 0
    if num_workers == 0:
        return None

    if hw_config.device_type == 'cuda':
        # Use more prefetching for streaming (I/O bound)
        if streaming:
            return 2
        else:
            return 4

    # Conservative for CPU
    return 2


def auto_batch_size(hw_config: Optional[HardwareConfig] = None) -> int:
    """
    Auto-detect optimal batch size based on hardware memory.

    Returns hardware-optimized batch size:
    - CUDA: Scaled based on GPU memory (32-256)
    - MPS: Scaled based on unified memory (16-64)
    - CPU: Conservative batch size (32)

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        Optimal batch size for the hardware

    Example:
        >>> hw = detect_hardware()
        >>> batch_size = auto_batch_size(hw)
        >>> dataloader = DataLoader(dataset, batch_size=batch_size)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return hw_config.batch_size


def auto_gradient_accumulation_steps(hw_config: Optional[HardwareConfig] = None) -> int:
    """
    Auto-detect optimal gradient accumulation steps based on hardware.

    Gradient accumulation simulates larger batch sizes by accumulating gradients
    across multiple forward/backward passes before updating weights.

    Returns:
    - CUDA with high memory: 1 (no accumulation needed)
    - MPS with low memory: 2-4 steps (simulate larger batches)
    - CPU: 1 (no accumulation needed)

    Args:
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.

    Returns:
        Optimal gradient accumulation steps

    Example:
        >>> hw = detect_hardware()
        >>> accum_steps = auto_gradient_accumulation_steps(hw)
        >>> trainer = L.Trainer(accumulate_grad_batches=accum_steps)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    return hw_config.gradient_accumulation_steps


def auto_learning_rate(
    base_lr: float,
    base_batch_size: int,
    hw_config: Optional[HardwareConfig] = None,
    scaling_rule: Literal['linear', 'sqrt'] = 'linear'
) -> float:
    """
    Auto-scale learning rate based on actual batch size and gradient accumulation.

    When batch size changes, learning rate should be adjusted to maintain training
    dynamics. Two common scaling rules:

    - Linear scaling (default): LR scales proportionally with effective batch size
      Effective batch size = batch_size * gradient_accumulation_steps
      Scaled LR = base_lr * (effective_batch_size / base_batch_size)
      Best for most cases, especially large models

    - Square root scaling: LR scales with sqrt of effective batch size
      Scaled LR = base_lr * sqrt(effective_batch_size / base_batch_size)
      Can be better for small models or very large batch sizes

    Args:
        base_lr: Base learning rate for base_batch_size
        base_batch_size: Batch size that base_lr was tuned for
        hw_config: HardwareConfig from detect_hardware(). If None, will auto-detect.
        scaling_rule: 'linear' (default) or 'sqrt' scaling

    Returns:
        Scaled learning rate for actual hardware configuration

    Example:
        >>> hw = detect_hardware(base_batch_size=64)
        >>> # Base LR was tuned for batch_size=64
        >>> scaled_lr = auto_learning_rate(base_lr=3e-4, base_batch_size=64, hw_config=hw)
        >>> # If hw.batch_size=32 and gradient_accumulation=2, scaled_lr = 3e-4 (same effective batch)
        >>> # If hw.batch_size=128, scaled_lr = 6e-4 (2x effective batch → 2x LR)
    """
    if hw_config is None:
        hw_config = detect_hardware(verbose=False)

    # Calculate effective batch sizes
    actual_batch_size = hw_config.batch_size
    gradient_accumulation = hw_config.gradient_accumulation_steps
    effective_batch_size = actual_batch_size * gradient_accumulation

    # Calculate scaling factor
    ratio = effective_batch_size / base_batch_size

    if scaling_rule == 'linear':
        scaled_lr = base_lr * ratio
    elif scaling_rule == 'sqrt':
        scaled_lr = base_lr * (ratio ** 0.5)
    else:
        raise ValueError(f"Unknown scaling_rule '{scaling_rule}'. Must be 'linear' or 'sqrt'")

    return scaled_lr


# ============================================================================
# Training Memory Estimation
# ============================================================================

def estimate_training_memory_gb(
    config: dict,
    vocab_size: int = 50257,
    dtype: str = "float32",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
    include_buffers: bool = True,
) -> dict:
    """
    Comprehensive memory estimator for training a GPT-2 style transformer.
    
    This estimates PEAK memory usage during training, which typically occurs
    during the backward pass through the cross-entropy loss.
    
    Args:
        config: Dictionary with model configuration:
            - n_positions: sequence length / block size
            - batch_size: micro-batch size (actual batch per forward pass)
            - n_embd: embedding dimension
            - n_head: number of attention heads
            - n_layer: number of transformer blocks
            - gradient_accumulation_steps: (optional, for display only)
        vocab_size: vocabulary size (default: 50257 for GPT-2)
        dtype: "float32", "float16", "bfloat16", or "mixed" (mixed precision)
        optimizer: "adamw", "adam", "sgd", "sgd_momentum", or "adafactor"
        gradient_checkpointing: if True, trades compute for memory
        include_buffers: include causal masks and other buffers
    
    Returns:
        Dictionary with detailed memory breakdown in GB
    
    Memory Components:
        1. Model Parameters: weights of all layers
        2. Gradients: same size as parameters (stored for optimizer step)
        3. Optimizer States: depends on optimizer (AdamW = 2x params)
        4. Activations: stored during forward pass for backprop
           - This is where most memory goes for large batch/sequence!
        5. Peak Backward Memory: temporary tensors during backward pass
        6. Framework Overhead: autograd graph, memory fragmentation, metadata
        7. Buffers: causal masks, position indices (small but present)
    
    Key Insight:
        Gradient accumulation does NOT multiply activation memory!
        - Forward pass 1 → store activations → backward pass 1 → free activations
        - Forward pass 2 → store activations → backward pass 2 → free activations
        - ... (gradients accumulate, but activations are freed each step)
        - Optimizer step (uses accumulated gradients)
        
        Peak memory occurs during a SINGLE forward-backward pass, not across
        accumulation steps.

    Example:
        >>> config = {"n_positions": 1024, "batch_size": 32, "n_embd": 768, 
        ...           "n_head": 12, "n_layer": 12}
        >>> estimate = estimate_training_memory_gb(config)
        >>> print(f"Total memory: {estimate['total_memory_gb']:.2f} GB")
    """
    # Extract config
    n_positions = config["n_positions"]  # T (sequence length)
    batch_size = config["batch_size"]     # B (micro-batch size)
    n_embd = config["n_embd"]             # C (embedding dimension)
    n_head = config["n_head"]             # H (number of heads)
    n_layer = config["n_layer"]           # L (number of layers)
    head_size = n_embd // n_head          # D (dimension per head)
    
    # Bytes per element based on dtype
    DTYPE_BYTES = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "mixed": 2,  # Activations in fp16, master weights in fp32
    }
    bytes_per_param = DTYPE_BYTES.get(dtype, 4)
    bytes_per_activation = 2 if dtype == "mixed" else bytes_per_param
    
    # =========================================================================
    # 1. MODEL PARAMETERS (weights + biases)
    # =========================================================================
    
    # Token embeddings: (vocab_size, n_embd)
    token_emb_params = vocab_size * n_embd
    
    # Position embeddings: (n_positions, n_embd)
    pos_emb_params = n_positions * n_embd
    
    # Per transformer block:
    # - QKV projection: Linear(n_embd, 3 * n_embd) → weight + bias
    attn_qkv_params = n_embd * 3 * n_embd + 3 * n_embd  # weight + bias
    # - Output projection: Linear(n_embd, n_embd) → weight + bias
    attn_out_params = n_embd * n_embd + n_embd
    # - FFN up: Linear(n_embd, 4 * n_embd)
    ffn_up_params = n_embd * 4 * n_embd + 4 * n_embd
    # - FFN down: Linear(4 * n_embd, n_embd)
    ffn_down_params = 4 * n_embd * n_embd + n_embd
    # - Layer norms: 2 per block, each has weight + bias of size n_embd
    ln_params_per_block = 2 * (n_embd + n_embd)
    
    params_per_block = attn_qkv_params + attn_out_params + ffn_up_params + ffn_down_params + ln_params_per_block
    
    # Final layer norm
    final_ln_params = n_embd + n_embd
    
    # LM head: Linear(n_embd, vocab_size) - usually tied to token embeddings
    lm_head_params = 0  # Assuming weight tying
    
    total_params = token_emb_params + pos_emb_params + (n_layer * params_per_block) + final_ln_params + lm_head_params
    
    # =========================================================================
    # 2. GRADIENTS
    # =========================================================================
    gradient_elements = total_params
    
    # =========================================================================
    # 3. OPTIMIZER STATES
    # =========================================================================
    OPTIMIZER_MULTIPLIER = {
        "adamw": 2,
        "adam": 2,
        "sgd": 0,
        "sgd_momentum": 1,
        "adafactor": 0.5,
    }
    optimizer_multiplier = OPTIMIZER_MULTIPLIER.get(optimizer, 2)
    optimizer_state_elements = total_params * optimizer_multiplier
    
    # =========================================================================
    # 4. ACTIVATIONS (Forward Pass - stored for backprop)
    # =========================================================================
    if gradient_checkpointing:
        activation_elements = (
            batch_size * n_positions * n_embd * (n_layer + 2)
        )
    else:
        # Input embeddings: (B, T, C)
        input_emb_act = batch_size * n_positions * n_embd
        
        # Per transformer block activations:
        attn_input = batch_size * n_positions * n_embd
        qkv_act = 3 * batch_size * n_head * n_positions * head_size
        attn_scores = batch_size * n_head * n_positions * n_positions
        attn_weights = batch_size * n_head * n_positions * n_positions
        attn_output_pre = batch_size * n_head * n_positions * head_size
        attn_output_post = batch_size * n_positions * n_embd
        
        attn_act_per_block = attn_input + qkv_act + attn_scores + attn_weights + attn_output_pre + attn_output_post
        
        # FFN:
        ffn_input = batch_size * n_positions * n_embd
        ffn_expanded = batch_size * n_positions * 4 * n_embd
        ffn_gelu_input = batch_size * n_positions * 4 * n_embd
        ffn_output = batch_size * n_positions * n_embd
        
        ffn_act_per_block = ffn_input + ffn_expanded + ffn_gelu_input + ffn_output
        
        # Residual stream saved at each block
        residual_per_block = 2 * batch_size * n_positions * n_embd
        
        act_per_block = attn_act_per_block + ffn_act_per_block + residual_per_block
        
        # Final layer norm input: (B, T, C)
        final_ln_input = batch_size * n_positions * n_embd
        
        # LM head output / logits: (B, T, V) - THIS IS MASSIVE!
        logits_act = batch_size * n_positions * vocab_size
        
        activation_elements = input_emb_act + (n_layer * act_per_block) + final_ln_input + logits_act
    
    # =========================================================================
    # 5. PEAK BACKWARD MEMORY
    # =========================================================================
    cross_entropy_peak = 2 * batch_size * n_positions * vocab_size
    
    attn_backward_peak = (
        3 * batch_size * n_head * n_positions * head_size +
        batch_size * n_head * n_positions * n_positions
    )
    
    backward_peak = max(cross_entropy_peak, attn_backward_peak)
    
    # =========================================================================
    # 6. BUFFERS
    # =========================================================================
    if include_buffers:
        causal_mask = n_positions * n_positions * n_layer
        position_indices = n_positions * 2
        buffer_elements = causal_mask + position_indices
    else:
        buffer_elements = 0
    
    # =========================================================================
    # CONVERT TO BYTES AND GB
    # =========================================================================
    def to_gb(elements, bytes_each):
        return (elements * bytes_each) / (1024**3)
    
    # Model weights
    model_memory_gb = to_gb(total_params, bytes_per_param)
    if dtype == "mixed":
        model_memory_gb = to_gb(total_params, 2) + to_gb(total_params, 4)
    
    # Gradients
    gradient_memory_gb = to_gb(gradient_elements, bytes_per_param)
    
    # Optimizer states (always fp32)
    optimizer_memory_gb = to_gb(optimizer_state_elements, 4)
    
    # Activations
    activation_memory_gb = to_gb(activation_elements, bytes_per_activation)
    
    # Peak backward memory
    backward_peak_gb = to_gb(backward_peak, bytes_per_activation)
    
    # Buffers
    buffer_memory_gb = to_gb(buffer_elements, 4)
    
    # =========================================================================
    # TOTAL MEMORY CALCULATION
    # =========================================================================
    static_memory_gb = model_memory_gb + gradient_memory_gb + optimizer_memory_gb + buffer_memory_gb
    dynamic_memory_gb = activation_memory_gb + backward_peak_gb * 0.7
    
    # Framework overhead: ~10%
    subtotal_gb = static_memory_gb + dynamic_memory_gb
    framework_overhead_gb = subtotal_gb * 0.10
    
    total_memory_gb = subtotal_gb + framework_overhead_gb
    
    # =========================================================================
    # DETAILED BREAKDOWN
    # =========================================================================
    logits_memory_gb = to_gb(batch_size * n_positions * vocab_size, bytes_per_activation)
    attn_matrix_per_block_gb = to_gb(2 * batch_size * n_head * n_positions * n_positions, bytes_per_activation)
    
    return {
        # Summary
        "total_memory_gb": total_memory_gb,
        "total_params": total_params,
        
        # Static memory
        "model_memory_gb": model_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "buffer_memory_gb": buffer_memory_gb,
        "static_memory_gb": static_memory_gb,
        
        # Dynamic memory
        "activation_memory_gb": activation_memory_gb,
        "backward_peak_gb": backward_peak_gb,
        "dynamic_memory_gb": dynamic_memory_gb,
        
        # Framework overhead
        "framework_overhead_gb": framework_overhead_gb,
        
        # Detailed breakdown
        "logits_memory_gb": logits_memory_gb,
        "cross_entropy_peak_gb": to_gb(cross_entropy_peak, bytes_per_activation),
        "attn_matrix_per_block_gb": attn_matrix_per_block_gb,
        
        # Config echo
        "config": {
            "batch_size": batch_size,
            "n_positions": n_positions,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "vocab_size": vocab_size,
            "dtype": dtype,
            "optimizer": optimizer,
            "gradient_checkpointing": gradient_checkpointing,
        },
        
        # Parameter breakdown
        "param_breakdown": {
            "token_embeddings": token_emb_params,
            "position_embeddings": pos_emb_params,
            "transformer_blocks": n_layer * params_per_block,
            "final_layer_norm": final_ln_params,
            "lm_head": lm_head_params,
        },
    }


def print_memory_estimate(estimate: dict) -> None:
    """
    Pretty print a memory estimate from estimate_training_memory_gb.
    
    Args:
        estimate: Dictionary returned by estimate_training_memory_gb
        
    Example:
        >>> estimate = estimate_training_memory_gb(config)
        >>> print_memory_estimate(estimate)
    """
    print("=" * 75)
    print("  TRAINING MEMORY ESTIMATE")
    print("=" * 75)
    
    cfg = estimate["config"]
    print(f"\n📋 Configuration:")
    print(f"   Model: {cfg['n_layer']}L-{cfg['n_head']}H-{cfg['n_embd']}D, vocab={cfg['vocab_size']:,}")
    print(f"   Training: batch={cfg['batch_size']}, seq_len={cfg['n_positions']}, dtype={cfg['dtype']}")
    print(f"   Optimizer: {cfg['optimizer']}, checkpointing={cfg['gradient_checkpointing']}")
    print(f"   Parameters: {estimate['total_params']:,} ({estimate['total_params']/1e6:.2f}M)")
    
    print(f"\n📊 Memory Breakdown:")
    print(f"   ┌─ Static Memory (always allocated) ──────────────────────────────┐")
    print(f"   │  Model weights:      {estimate['model_memory_gb']:>8.3f} GB                          │")
    print(f"   │  Gradients:          {estimate['gradient_memory_gb']:>8.3f} GB                          │")
    print(f"   │  Optimizer states:   {estimate['optimizer_memory_gb']:>8.3f} GB                          │")
    print(f"   │  Buffers:            {estimate['buffer_memory_gb']:>8.3f} GB                          │")
    print(f"   │  ─────────────────────────────────────                          │")
    print(f"   │  Static subtotal:    {estimate['static_memory_gb']:>8.3f} GB                          │")
    print(f"   └─────────────────────────────────────────────────────────────────┘")
    
    print(f"   ┌─ Dynamic Memory (forward/backward pass) ────────────────────────┐")
    print(f"   │  Activations:        {estimate['activation_memory_gb']:>8.3f} GB                          │")
    print(f"   │  Backward peak:      {estimate['backward_peak_gb']:>8.3f} GB                          │")
    print(f"   │  ─────────────────────────────────────                          │")
    print(f"   │  Dynamic subtotal:   {estimate['dynamic_memory_gb']:>8.3f} GB                          │")
    print(f"   └─────────────────────────────────────────────────────────────────┘")
    
    print(f"   ┌─ Overhead ───────────────────────────────────────────────────────┐")
    print(f"   │  Framework (10%):    {estimate['framework_overhead_gb']:>8.3f} GB                          │")
    print(f"   └─────────────────────────────────────────────────────────────────┘")
    
    print(f"\n🎯 TOTAL PEAK MEMORY:     {estimate['total_memory_gb']:>8.3f} GB")
    
    print(f"\n⚠️  Largest Memory Consumers:")
    print(f"   • Logits tensor (B×T×V):       {estimate['logits_memory_gb']:.3f} GB")
    print(f"   • Cross-entropy backward:      {estimate['cross_entropy_peak_gb']:.3f} GB")
    print(f"   • Attention matrices (per L):  {estimate['attn_matrix_per_block_gb']:.3f} GB")
    
    # Recommendations
    total = estimate['total_memory_gb']
    print(f"\n💡 Recommendations for your {cfg['batch_size']}×{cfg['n_positions']} config:")
    if total > 14:
        print(f"   ❌ Too large for 16GB! Reduce batch_size or use gradient accumulation")
        safe_batch = max(1, int(cfg['batch_size'] * 12 / total))
        print(f"   → Try batch_size={safe_batch} with gradient_accumulation_steps={cfg['batch_size']//safe_batch}")
    elif total > 12:
        print(f"   ⚠️  Tight fit for 16GB. May OOM under memory pressure.")
        print(f"   → Consider reducing batch_size slightly or using gradient checkpointing")
    elif total > 8:
        print(f"   ✅ Should fit in 16GB with some headroom")
    else:
        print(f"   ✅ Comfortable fit in 16GB")
        if cfg['batch_size'] < 128:
            print(f"   → You could likely increase batch_size for faster training")
    
    print("=" * 75)


def find_max_batch_size(
    config: dict,
    memory_budget_gb: float = 14.0,
    target_effective_batch: Optional[int] = None,
    vocab_size: int = 50257,
    dtype: str = "float32",
    optimizer: str = "adamw",
    gradient_checkpointing: bool = False,
) -> dict:
    """
    Find the maximum batch size that fits in memory, with gradient accumulation.
    
    Uses binary search to find the largest batch size that fits within the
    memory budget, then calculates gradient accumulation steps if a target
    effective batch size is specified.
    
    Args:
        config: Base config dict (batch_size will be overridden during search)
        memory_budget_gb: Available GPU memory (default 14GB for 16GB card with OS overhead)
        target_effective_batch: Desired effective batch size (batch_size × grad_accum)
        vocab_size: Vocabulary size
        dtype: Data type for training
        optimizer: Optimizer type
        gradient_checkpointing: Whether using gradient checkpointing
    
    Returns:
        Dictionary with recommended batch_size and gradient_accumulation_steps
        
    Example:
        >>> config = {"n_positions": 1024, "n_embd": 768, "n_head": 12, 
        ...           "n_layer": 12, "batch_size": 32}
        >>> result = find_max_batch_size(config, memory_budget_gb=14.0, 
        ...                               target_effective_batch=512)
        >>> print(f"Use batch_size={result['batch_size']} with "
        ...       f"gradient_accumulation_steps={result['gradient_accumulation_steps']}")
    """
    # Binary search for max batch size
    low, high = 1, 2048
    max_batch = 1
    
    test_config = config.copy()
    
    while low <= high:
        mid = (low + high) // 2
        test_config["batch_size"] = mid
        
        estimate = estimate_training_memory_gb(
            test_config,
            vocab_size=vocab_size,
            dtype=dtype,
            optimizer=optimizer,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        if estimate["total_memory_gb"] <= memory_budget_gb:
            max_batch = mid
            low = mid + 1
        else:
            high = mid - 1
    
    # Calculate gradient accumulation steps if target effective batch specified
    if target_effective_batch:
        grad_accum = max(1, target_effective_batch // max_batch)
        effective_batch = max_batch * grad_accum
    else:
        grad_accum = 1
        effective_batch = max_batch
    
    # Get final memory estimate
    test_config["batch_size"] = max_batch
    final_estimate = estimate_training_memory_gb(
        test_config,
        vocab_size=vocab_size,
        dtype=dtype,
        optimizer=optimizer,
        gradient_checkpointing=gradient_checkpointing,
    )
    
    return {
        "batch_size": max_batch,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "estimated_memory_gb": final_estimate["total_memory_gb"],
        "memory_budget_gb": memory_budget_gb,
        "headroom_gb": memory_budget_gb - final_estimate["total_memory_gb"],
    }


# ============================================================================
# Memory Management
# ============================================================================

def free_memory(verbose: bool = True, aggressive: bool = False) -> int:
    """
    Free up memory by deleting large objects and clearing caches.
    
    Use this when you need to free memory without restarting the kernel.
    
    This function:
    1. Deletes common large variables (model, optimizer, data loaders, etc.)
    2. Clears MPS/CUDA caches multiple times
    3. Forces multiple rounds of garbage collection
    4. Shows memory before/after (if verbose)
    
    Args:
        verbose: Show detailed output
        aggressive: If True, also delete large objects found automatically
    
    Returns:
        Number of variables deleted
        
    Example:
        >>> free_memory()  # Standard cleanup
        >>> free_memory(aggressive=True)  # More aggressive cleanup
    """
    import gc
    import sys
    
    # Get memory before cleanup
    mem_before = None
    if verbose:
        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024**3)  # GB
            print(f"\n🧹 Memory cleanup starting...")
            print(f"   Memory before: {mem_before:.2f} GB")
        except Exception:
            print(f"\n🧹 Memory cleanup starting...")
    
    # Get all namespaces (Jupyter notebooks use user_ns)
    namespaces = [globals()]
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            namespaces.append(ipython.user_ns)
    except Exception:
        pass
    
    # List of common variable names to delete
    vars_to_delete = [
        'model', 'model_pretrained',
        'optimizer',
        'train_loader', 'val_loader', 'test_loader',
        'train_dataset', 'val_dataset', 'test_dataset',
        'x', 'y', 'logits', 'loss', 'losses', 'grad_norms',
        'stats', 'output_tokens', 'tokens_t',
        'attention', 'Q', 'K', 'V', 'QKt', 'QKt_scaled', 'QKt_masked',
        'mha_out', 'ffn_out', 'out',
        'memory_estimate',
        'token_embedding_table', 'position_embedding_table',
        'padded', 'tokens_emb', 'positions_emb',
    ]
    
    deleted_count = 0
    deleted_vars = []
    
    # Delete from all namespaces
    for namespace in namespaces:
        for var_name in vars_to_delete:
            if var_name in namespace:
                try:
                    del namespace[var_name]
                    deleted_count += 1
                    deleted_vars.append(var_name)
                except Exception:
                    pass
    
    # Aggressive mode: find and delete large objects
    if aggressive:
        if verbose:
            print(f"   🔍 Aggressive mode: searching for large objects...")
        
        for namespace in namespaces:
            for name, obj in list(namespace.items()):
                if name.startswith('_'):
                    continue
                
                try:
                    # Check if it's a large PyTorch tensor or model
                    size = 0
                    if hasattr(obj, 'element_size') and hasattr(obj, 'nelement'):
                        size = obj.element_size() * obj.nelement()
                    elif hasattr(obj, 'parameters'):
                        size = sum(p.numel() * (p.element_size() if hasattr(p, 'element_size') else 4)
                                  for p in obj.parameters())
                    elif isinstance(obj, (list, tuple)) and len(obj) > 1000:
                        size = sys.getsizeof(obj)
                    
                    # Delete if > 10MB
                    if size > 10 * 1024 * 1024:
                        try:
                            del namespace[name]
                            deleted_count += 1
                            deleted_vars.append(name)
                            if verbose:
                                print(f"      Deleted large object: {name} ({size / (1024**2):.1f} MB)")
                        except Exception:
                            pass
                except Exception:
                    pass
    
    # Clear PyTorch caches multiple times
    if torch.backends.mps.is_available():
        for _ in range(3):
            torch.mps.empty_cache()
        if verbose:
            print(f"   ✓ Cleared MPS cache (3x)")
    elif torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
        if verbose:
            print(f"   ✓ Cleared CUDA cache (3x)")
    
    # Multiple rounds of garbage collection
    total_collected = 0
    for _ in range(3):
        collected = gc.collect()
        total_collected += collected
    
    if verbose:
        print(f"   ✓ Garbage collected {total_collected} objects (3 rounds)")
        print(f"   ✓ Deleted {deleted_count} variables: {', '.join(deleted_vars[:10])}")
        if len(deleted_vars) > 10:
            print(f"      ... and {len(deleted_vars) - 10} more")
    
    # Get memory after cleanup
    if verbose and mem_before is not None:
        try:
            process = psutil.Process()
            mem_after = process.memory_info().rss / (1024**3)
            freed = mem_before - mem_after
            print(f"   Memory after:  {mem_after:.2f} GB")
            print(f"   Memory freed:  {freed:.2f} GB")
            if freed < 0.1:
                print(f"   ⚠️  Warning: Little memory freed. Try aggressive=True or manual cleanup.")
            print(f"   ✅ Memory cleanup complete!")
        except Exception:
            print(f"   ✅ Memory cleanup complete!")
    
    return deleted_count


def show_memory_usage() -> None:
    """
    Show memory usage of large variables in the current namespace.

    Helps identify what's consuming memory by listing all variables
    larger than 1MB sorted by size.

    Example:
        >>> show_memory_usage()
        ====================
        MEMORY USAGE BY VARIABLE
        ====================
        Variable                           Size (MB)       Size (GB)
        model                                 512.00           0.500
        train_loader                          256.00           0.250
        ...
    """
    import sys

    print("=" * 70)
    print("MEMORY USAGE BY VARIABLE")
    print("=" * 70)

    # Get all variables in global scope
    vars_dict = globals()

    # Calculate size of each variable
    var_sizes = []
    for name, obj in vars_dict.items():
        if not name.startswith('_'):
            try:
                size = sys.getsizeof(obj)
                # For PyTorch tensors, get actual memory
                if hasattr(obj, 'element_size') and hasattr(obj, 'nelement'):
                    size = obj.element_size() * obj.nelement()
                elif hasattr(obj, '__dict__'):
                    # For models, estimate size
                    if hasattr(obj, 'parameters'):
                        size = sum(p.numel() * p.element_size() if hasattr(p, 'numel') else 0
                                  for p in obj.parameters())

                if size > 1024 * 1024:  # Only show variables > 1MB
                    var_sizes.append((name, size / (1024**2)))  # MB
            except Exception:
                pass

    # Sort by size
    var_sizes.sort(key=lambda x: x[1], reverse=True)

    if var_sizes:
        print(f"{'Variable':<30} {'Size (MB)':>15} {'Size (GB)':>15}")
        print("-" * 70)
        for name, size_mb in var_sizes[:20]:  # Top 20
            size_gb = size_mb / 1024
            print(f"{name:<30} {size_mb:>15.2f} {size_gb:>15.3f}")
    else:
        print("No large variables found (>1MB)")

    print("=" * 70)
    print("\n💡 To free memory, run: free_memory()")
    print("💡 For aggressive cleanup: free_memory(aggressive=True)")
    print("=" * 70)


# ============================================================================
# Configuration Initialization
# ============================================================================

def init_training(config: dict, verbose: bool = True) -> dict:
    """
    Initialize training environment with automatic hardware detection and seeding.

    This function:
    1. Sets random seed for reproducibility using Lightning's seed_everything
    2. Detects hardware capabilities (CUDA, MPS, or CPU)
    3. Resolves "auto" fields in config based on hardware detection
    4. Prints the final configuration
    5. Returns a new config dict with all "auto" fields resolved

    Supported "auto" fields:
        - device: "auto" -> "cuda"/"mps"/"cpu"
        - precision: "auto" -> "bf16-mixed"/"16-mixed"/"32-true"
        - batch_size: "auto" -> hardware-optimized batch size
        - pin_memory: "auto" -> True for CUDA, False for MPS/CPU
        - num_workers: "auto" -> 0-12 based on hardware and dataset mode
        - persistent_workers: "auto" -> True for CUDA (if num_workers > 0), False otherwise
        - prefetch_factor: "auto" -> 2-4 based on hardware, None if num_workers=0
        - gradient_accumulation_steps: "auto" -> 1-4 based on memory
        - use_compile: "auto" -> True for CUDA, False for MPS/CPU
        - use_flash_attention: "auto" -> True if available on CUDA
        - use_fused_optimizer: "auto" -> True for CUDA, False otherwise
        - torch_compile: "auto" -> "max-autotune" for CUDA, False for MPS/CPU
        - use_fused_adamw: "auto" -> True for CUDA, False otherwise

    Args:
        config: Configuration dictionary with training hyperparameters
                Must include "seed" key for reproducibility
        verbose: Print configuration details (default: True)

    Returns:
        New configuration dictionary with all "auto" fields resolved

    Example:
        >>> CONFIG = {
        ...     "seed": 42,
        ...     "device": "auto",
        ...     "precision": "auto",
        ...     "batch_size": "auto",
        ...     "learning_rate": 1e-3,
        ... }
        >>> CONFIG = init_training(CONFIG)

        🌱 Initializing Training Environment
        ═══════════════════════════════════════════════════════════════

        🎲 Random Seed: 42
           ✓ Seed set using Lightning's seed_everything()

        🍎 Apple Silicon Detected
           Architecture: arm64
           ...

        📊 Final Configuration:
           Device: mps
           Precision: 16-mixed
           Batch size: 32
           ...

        >>> # All auto fields are now resolved
        >>> CONFIG["device"]  # "mps" instead of "auto"
        >>> CONFIG["precision"]  # "16-mixed" instead of "auto"
    """
    try:
        import lightning as L
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required for init_training. "
            "Install with: uv pip install lightning"
        )

    if verbose:
        print("\n🌱 Initializing Training Environment")
        print("═" * 75)

    # Create a copy to avoid modifying the original
    resolved_config = config.copy()

    # 1. Set random seed
    if "seed" not in config:
        raise ValueError("Config must include 'seed' field for reproducibility")

    seed = config["seed"]
    L.seed_everything(seed, workers=True)

    if verbose:
        print(f"\n🎲 Random Seed: {seed}")
        print(f"   ✓ Seed set using Lightning's seed_everything()")

    # 2. Check if we need to detect hardware (any "auto" fields present)
    auto_fields = {
        k: v for k, v in config.items()
        if isinstance(v, str) and v == "auto"
    }

    if not auto_fields:
        if verbose:
            print("\n✓ No 'auto' fields detected. Configuration ready.")
            print("═" * 75)
        return resolved_config

    # 3. Detect hardware
    if verbose:
        print("\n🔍 Detecting hardware for auto-resolution...")
        print()

    hw_config = detect_hardware(
        base_batch_size=config.get("base_batch_size", 64),
        enable_tf32=config.get("enable_tf32", True),
        enable_mps_fallback=config.get("enable_mps_fallback", True),
        verbose=verbose
    )

    # 4. Resolve auto fields
    if verbose:
        print(f"\n⚙️  Resolving {len(auto_fields)} 'auto' field(s):")

    # Check if using streaming mode
    is_streaming = resolved_config.get("_use_streaming", False)

    # Resolve num_workers first (needed for other dataloader settings)
    num_workers = None
    if "num_workers" in auto_fields:
        num_workers = auto_num_workers(hw_config, streaming=is_streaming)
        resolved_config["num_workers"] = num_workers

    field_mapping = {
        "device": lambda: str(hw_config.device),
        "precision": lambda: hw_config.precision,
        "batch_size": lambda: hw_config.batch_size,
        "pin_memory": lambda: hw_config.pin_memory,
        "num_workers": lambda: auto_num_workers(hw_config, streaming=is_streaming),
        "persistent_workers": lambda: auto_persistent_workers(hw_config, num_workers),
        "prefetch_factor": lambda: auto_prefetch_factor(hw_config, num_workers, streaming=is_streaming),
        "gradient_accumulation_steps": lambda: hw_config.gradient_accumulation_steps,
        "use_compile": lambda: hw_config.use_compile,
        "use_flash_attention": lambda: hw_config.use_flash_attention,
        "use_fused_optimizer": lambda: hw_config.use_fused_optimizer,
        "torch_compile": lambda: "max-autotune" if hw_config.use_compile else False,
        "use_fused_adamw": lambda: hw_config.use_fused_optimizer,
    }

    for field in auto_fields:
        if field in field_mapping:
            resolved_value = field_mapping[field]()
            resolved_config[field] = resolved_value
            if verbose:
                print(f"   {field}: auto → {resolved_value}")
        else:
            if verbose:
                print(f"   ⚠️  Unknown auto field '{field}' - leaving as 'auto'")

    # 5. Print final configuration summary
    if verbose:
        print("\n" + "═" * 75)
        print("✓ Initialization complete!")
        print("═" * 75)

    return resolved_config

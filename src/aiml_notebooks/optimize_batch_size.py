"""
Optimal Batch Size Finder for Transformer Training

Automatically finds the best batch_size and gradient_accumulation_steps
for your model, hardware, and training objectives.

Usage:
    from aiml_notebooks.optimize_batch_size import find_optimal_batch_config

    config = find_optimal_batch_config(
        model=my_model,
        sample_input=(batch_size, seq_len),  # Shape for dummy data
        target_effective_batch=512,
        device='auto'
    )

    # Use the recommended config
    batch_size = config['batch_size']
    grad_accum_steps = config['gradient_accumulation_steps']
"""

import torch
import torch.nn as nn
import time
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List, Dict, Any
import psutil
import os
from tqdm import tqdm


@dataclass
class BatchConfig:
    """Configuration for batch size and gradient accumulation."""
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int

    # Performance metrics
    updates_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    time_per_update: float = 0.0
    memory_gb: float = 0.0

    # Quality metrics
    score: float = 0.0
    fits_in_memory: bool = True

    # Hardware info
    platform: str = ""

    def __repr__(self):
        return (
            f"BatchConfig(batch={self.batch_size}, "
            f"accum={self.gradient_accumulation_steps}, "
            f"effective={self.effective_batch_size}, "
            f"updates/s={self.updates_per_sec:.2f}, "
            f"score={self.score:.2f})"
        )


class BatchSizeOptimizer:
    """Find optimal batch size and gradient accumulation for transformer training."""

    def __init__(
        self,
        model: nn.Module,
        sample_input_shape: Tuple[int, ...],
        vocab_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        optimizer_class: type = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize the batch size optimizer.

        Args:
            model: PyTorch model to optimize for
            sample_input_shape: Shape of input tensor (batch_size, seq_len) or (batch_size, channels, height, width)
            vocab_size: Vocabulary size for language models (if applicable)
            seq_len: Sequence length (auto-detected from sample_input_shape if not provided)
            device: Target device ('auto' to auto-detect, or specific device)
            optimizer_class: Optimizer to use (default: AdamW)
            optimizer_kwargs: Kwargs for optimizer (default: {'lr': 3e-4})
        """
        self.model = model
        self.sample_input_shape = sample_input_shape
        self.vocab_size = vocab_size
        self.seq_len = seq_len or (sample_input_shape[1] if len(sample_input_shape) > 1 else None)
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 3e-4}

        # Auto-detect device
        if device is None or device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.platform = 'mps'
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.platform = 'cuda'
            else:
                self.device = torch.device('cpu')
                self.platform = 'cpu'
        else:
            self.device = device
            self.platform = str(device.type)

        # Move model to device
        self.model = self.model.to(self.device)

        # Calculate model parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())

    def _get_memory_usage_gb(self) -> float:
        """Get current process memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)

    def _sync_device(self):
        """Synchronize device (for accurate timing)."""
        if self.platform == 'mps':
            torch.mps.synchronize()
        elif self.platform == 'cuda':
            torch.cuda.synchronize()

    def _clear_cache(self):
        """Clear device cache."""
        if self.platform == 'mps':
            torch.mps.empty_cache()
        elif self.platform == 'cuda':
            torch.cuda.empty_cache()

    def _create_dummy_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy input/target batch for testing."""
        if self.vocab_size is not None:
            # Language model: (batch_size, seq_len)
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
            y = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        else:
            # General case: use sample_input_shape
            shape = (batch_size,) + self.sample_input_shape[1:]
            x = torch.randn(shape, device=self.device)
            y = torch.randint(0, 2, (batch_size,), device=self.device)  # Binary classification
        return x, y

    def _estimate_target_effective_batch(self) -> int:
        """
        Estimate optimal effective batch size based on model size.

        Based on research:
        - Small models (<10M params): 128-256
        - Medium models (10M-100M): 256-512
        - Large models (100M-1B): 512-2048
        - Very large (>1B): 2048+
        """
        if self.num_params < 10e6:
            return 256
        elif self.num_params < 100e6:
            return 512
        elif self.num_params < 1e9:
            return 1024
        else:
            return 2048

    def profile_configuration(
        self,
        batch_size: int,
        gradient_accumulation_steps: int,
        num_warmup: int = 3,
        num_measure: int = 20,
        show_progress: bool = False,
    ) -> Optional[BatchConfig]:
        """
        Profile a specific batch size + gradient accumulation configuration.

        Args:
            batch_size: Batch size per forward/backward pass
            gradient_accumulation_steps: Number of accumulation steps
            num_warmup: Number of warmup iterations
            num_measure: Number of measurement iterations

        Returns:
            BatchConfig with performance metrics, or None if OOM
        """
        effective_batch_size = batch_size * gradient_accumulation_steps

        try:
            # Create fresh optimizer for this test
            model = self.model
            model.train()
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)

            # Warmup
            warmup_iter = tqdm(range(num_warmup), desc="Warmup", leave=False, disable=not show_progress)
            for _ in warmup_iter:
                optimizer.zero_grad()
                for _ in range(gradient_accumulation_steps):
                    x, y = self._create_dummy_batch(batch_size)

                    # Forward pass
                    if self.vocab_size is not None:
                        # Language model
                        logits = model(x)
                        if logits.dim() == 3:  # (batch, seq, vocab)
                            logits = logits.permute(0, 2, 1)
                            loss = torch.nn.functional.cross_entropy(logits, y)
                        else:
                            loss = torch.nn.functional.cross_entropy(logits, y)
                    else:
                        # General model
                        output = model(x)
                        loss = torch.nn.functional.cross_entropy(output, y)

                    # Backward pass (scaled by accumulation steps)
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                optimizer.step()

            # Measure
            self._sync_device()
            start_time = time.time()

            num_updates = num_measure
            measure_iter = tqdm(range(num_updates), desc="Measuring", leave=False, disable=not show_progress)
            for _ in measure_iter:
                optimizer.zero_grad()
                for _ in range(gradient_accumulation_steps):
                    x, y = self._create_dummy_batch(batch_size)

                    if self.vocab_size is not None:
                        logits = model(x)
                        if logits.dim() == 3:
                            logits = logits.permute(0, 2, 1)
                            loss = torch.nn.functional.cross_entropy(logits, y)
                        else:
                            loss = torch.nn.functional.cross_entropy(logits, y)
                    else:
                        output = model(x)
                        loss = torch.nn.functional.cross_entropy(output, y)

                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                optimizer.step()

            self._sync_device()
            elapsed = time.time() - start_time

            # Calculate metrics
            updates_per_sec = num_updates / elapsed
            samples_per_sec = (num_updates * effective_batch_size) / elapsed
            tokens_per_sec = samples_per_sec * (self.seq_len or 1)
            time_per_update = elapsed / num_updates
            memory_gb = self._get_memory_usage_gb()

            # Cleanup
            del optimizer
            self._clear_cache()

            return BatchConfig(
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                effective_batch_size=effective_batch_size,
                updates_per_sec=updates_per_sec,
                samples_per_sec=samples_per_sec,
                tokens_per_sec=tokens_per_sec,
                time_per_update=time_per_update,
                memory_gb=memory_gb,
                fits_in_memory=True,
                platform=self.platform,
            )

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
                self._clear_cache()
                return BatchConfig(
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    effective_batch_size=effective_batch_size,
                    fits_in_memory=False,
                    platform=self.platform,
                )
            else:
                raise

    def _test_oom_only(self, batch_size: int) -> bool:
        """
        Fast OOM test - just try one forward+backward pass.
        Much faster than full profiling for finding max batch size.

        Args:
            batch_size: Batch size to test

        Returns:
            True if batch size fits in memory, False if OOM
        """
        try:
            model = self.model
            model.train()
            optimizer = self.optimizer_class(model.parameters(), **self.optimizer_kwargs)

            # Single forward+backward pass
            x, y = self._create_dummy_batch(batch_size)

            if self.vocab_size is not None:
                logits = model(x)
                if logits.dim() == 3:
                    logits = logits.permute(0, 2, 1)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y)
            else:
                output = model(x)
                loss = torch.nn.functional.cross_entropy(output, y)

            loss.backward()
            optimizer.step()

            # Cleanup
            del optimizer, x, y, logits, loss
            self._clear_cache()

            return True

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
                self._clear_cache()
                return False
            else:
                raise

    def find_max_batch_size(
        self,
        min_batch: int = 8,
        max_batch: int = 4096,
        show_progress: bool = True,
    ) -> int:
        """
        Binary search to find maximum batch size that fits in memory.

        Args:
            min_batch: Minimum batch size to test
            max_batch: Maximum batch size to test
            show_progress: Show progress bar during search

        Returns:
            Maximum batch size that fits in memory
        """
        left, right = min_batch, max_batch
        max_fits = min_batch

        # Estimate number of iterations for progress bar
        import math
        max_iterations = math.ceil(math.log2(max_batch / min_batch)) + 1

        pbar = tqdm(total=max_iterations, desc="Finding max batch size", disable=not show_progress)

        while left <= right:
            mid = (left + right) // 2

            # Fast OOM-only test (much faster than full profiling)
            fits = self._test_oom_only(mid)

            if fits:
                max_fits = mid
                left = mid + 1
                pbar.set_postfix({"current_max": max_fits})
            else:
                right = mid - 1

            pbar.update(1)

        pbar.close()
        return max_fits

    def score_configuration(
        self,
        config: BatchConfig,
        target_effective_batch: int,
        weight_speed: float = 0.3,
        weight_quality: float = 0.7,
    ) -> float:
        """
        Score a configuration based on speed and training quality.

        Args:
            config: Configuration to score
            target_effective_batch: Target effective batch size for optimal training
            weight_speed: Weight for speed component (0-1)
            weight_quality: Weight for quality component (0-1)

        Returns:
            Score (higher is better)
        """
        if not config.fits_in_memory:
            return 0.0

        # Speed component: normalized updates per second
        # (will be normalized against max in find_optimal_config)
        speed_score = config.updates_per_sec

        # Quality component: how close to target effective batch size
        # Use gaussian-like scoring: peak at target, decays on both sides
        ratio = config.effective_batch_size / target_effective_batch
        if ratio < 1:
            # Penalize being too small (noisy gradients)
            quality_score = ratio ** 2
        else:
            # Penalize being too large (diminishing returns)
            quality_score = 1.0 / (1.0 + (ratio - 1) ** 2)

        return speed_score * weight_speed + quality_score * weight_quality * 100

    def find_optimal_config(
        self,
        target_effective_batch: Optional[int] = None,
        test_batch_sizes: Optional[List[int]] = None,
        test_accumulation_steps: Optional[List[int]] = None,
        weight_speed: float = 0.3,
        weight_quality: float = 0.7,
        top_k: int = 5,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Find optimal batch size and gradient accumulation configuration.

        Args:
            target_effective_batch: Target effective batch size (auto-estimated if None)
            test_batch_sizes: List of batch sizes to test (auto-generated if None)
            test_accumulation_steps: List of accumulation steps to test
            weight_speed: Weight for speed vs quality (0=all quality, 1=all speed)
            weight_quality: Weight for training quality
            top_k: Number of top configurations to return

        Returns:
            Dictionary with:
                - 'recommended': Best configuration
                - 'alternatives': List of alternative configurations
                - 'all_results': All tested configurations
                - 'summary': Summary statistics
        """
        # Auto-estimate target effective batch if not provided
        if target_effective_batch is None:
            target_effective_batch = self._estimate_target_effective_batch()

        print(f"🔍 Optimizing for {self.platform.upper()} | Model: {self.num_params/1e6:.1f}M params")
        print(f"🎯 Target effective batch size: {target_effective_batch}")
        print(f"⚖️  Optimization weights: {weight_speed:.0%} speed, {weight_quality:.0%} quality\n")

        # Find maximum batch size if not provided
        if test_batch_sizes is None:
            # Smart estimate: ballpark based on model size, then verify
            # GPU memory scales roughly with: params * seq_len * batch_size * 4 (float32)
            # Plus gradients (2x) and optimizer states (2x) = ~12x multiplier
            # MPS M1 Pro 16GB has ~10GB usable for tensors

            seq_len_for_calc = self.seq_len or 128  # Default if not set
            model_memory_gb = (self.num_params * seq_len_for_calc * 4 * 12) / (1024**3)
            estimated_max_batch = int(10.0 / model_memory_gb)  # Conservative estimate

            # Clamp to reasonable range
            estimated_max_batch = max(64, min(estimated_max_batch, 2048))

            # Round to nearest power of 2
            import math
            estimated_max_batch = 2 ** int(math.log2(estimated_max_batch))

            print(f"📏 Estimating maximum batch size (model: {self.num_params/1e6:.1f}M params)...")
            print(f"   💡 Predicted: ~{estimated_max_batch} (verifying...)")

            # Quick verify estimate works, try 1.5x if it does
            if self._test_oom_only(estimated_max_batch):
                max_batch = estimated_max_batch
                # Try going higher
                test_higher = int(estimated_max_batch * 1.5)
                if self._test_oom_only(test_higher):
                    max_batch = test_higher
                    # Try even higher
                    test_higher = estimated_max_batch * 2
                    if self._test_oom_only(test_higher):
                        max_batch = test_higher
            else:
                # Estimate was too high, binary search down
                max_batch = self.find_max_batch_size(
                    min_batch=estimated_max_batch // 4,
                    max_batch=estimated_max_batch,
                    show_progress=False
                )

            print(f"   ✓ Maximum batch size: {max_batch}\n")

            # Generate test batch sizes (powers of 2, plus some in-between values)
            test_batch_sizes = []
            batch = 8
            while batch <= max_batch:
                test_batch_sizes.append(batch)
                if batch * 1.5 <= max_batch:
                    test_batch_sizes.append(int(batch * 1.5))
                batch *= 2
            test_batch_sizes = sorted(set(test_batch_sizes))

        # Generate accumulation steps to test
        if test_accumulation_steps is None:
            test_accumulation_steps = [1, 2, 4, 8, 16]

        # Smart configuration generation: only test combos near target effective batch
        print("⚡ Profiling configurations (focused on optimal range)...")
        all_configs = []

        # Generate smart configurations around target effective batch
        target_multiples = [0.5, 0.75, 1.0, 1.5, 2.0]  # Test 0.5x, 0.75x, 1x, 1.5x, 2x target
        smart_configs = set()

        for multiple in target_multiples:
            target_eff = int(target_effective_batch * multiple)

            # Find all batch/accum combinations that achieve this effective batch
            for batch_size in test_batch_sizes:
                if target_eff % batch_size == 0:  # Exact match possible
                    accum_needed = target_eff // batch_size
                    if accum_needed in test_accumulation_steps:
                        smart_configs.add((batch_size, accum_needed))

            # Also add nearby configurations
            for batch_size in test_batch_sizes:
                for accum in test_accumulation_steps:
                    eff = batch_size * accum
                    # Within 20% of target
                    if abs(eff - target_eff) / target_eff < 0.2:
                        smart_configs.add((batch_size, accum))

        # Convert to sorted list
        configs_to_test = sorted(smart_configs, key=lambda x: x[0] * x[1])

        print(f"   📊 Testing {len(configs_to_test)} targeted configurations (vs {len(test_batch_sizes) * len(test_accumulation_steps)} exhaustive)")

        # Create progress bar
        pbar = tqdm(total=len(configs_to_test), desc="Testing configs", disable=not show_progress)

        for batch_size, grad_accum in configs_to_test:
            effective = batch_size * grad_accum

            config = self.profile_configuration(
                batch_size, grad_accum, num_measure=15, show_progress=False
            )

            if config and config.fits_in_memory:
                # Score this configuration
                config.score = self.score_configuration(
                    config, target_effective_batch, weight_speed, weight_quality
                )
                all_configs.append(config)

                # Print progress (use tqdm.write to avoid interfering with progress bar)
                accum_str = f"×{grad_accum}" if grad_accum > 1 else "native"
                result_str = (f"   batch={batch_size:<4} {accum_str:<8} → "
                              f"effective={effective:<5} | "
                              f"{config.updates_per_sec:>5.1f} upd/s | "
                              f"score={config.score:>5.1f}")
                if show_progress:
                    tqdm.write(result_str)
                else:
                    print(result_str)

            pbar.update(1)
            pbar.set_postfix({"tested": len(all_configs), "fits": len(all_configs)})

        pbar.close()

        if not all_configs:
            raise RuntimeError("No valid configurations found!")

        # Normalize scores
        max_speed = max(c.updates_per_sec for c in all_configs)
        for config in all_configs:
            speed_normalized = config.updates_per_sec / max_speed

            ratio = config.effective_batch_size / target_effective_batch
            if ratio < 1:
                quality_score = ratio ** 2
            else:
                quality_score = 1.0 / (1.0 + (ratio - 1) ** 2)

            config.score = speed_normalized * weight_speed + quality_score * weight_quality

        # Sort by score
        all_configs.sort(key=lambda c: c.score, reverse=True)

        # Get top k configurations
        recommended = all_configs[0]
        alternatives = all_configs[1:top_k]

        # Print results
        print("\n" + "="*80)
        print("🏆 RECOMMENDED CONFIGURATION:")
        print("="*80)
        self._print_config(recommended)

        if alternatives:
            print("\n📋 ALTERNATIVE CONFIGURATIONS:")
            for i, config in enumerate(alternatives, 1):
                print(f"\n{i}. Alternative (score={config.score:.3f}):")
                self._print_config(config, detailed=False)

        return {
            'recommended': asdict(recommended),
            'alternatives': [asdict(c) for c in alternatives],
            'all_results': [asdict(c) for c in all_configs],
            'summary': {
                'target_effective_batch': target_effective_batch,
                'num_configs_tested': len(all_configs),
                'platform': self.platform,
                'model_params': self.num_params,
            }
        }

    def _print_config(self, config: BatchConfig, detailed: bool = True):
        """Pretty print a configuration."""
        print(f"  batch_size: {config.batch_size}")
        print(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
        print(f"  effective_batch_size: {config.effective_batch_size}")

        if detailed:
            print(f"\n  Performance:")
            print(f"    • {config.updates_per_sec:.2f} updates/second")
            print(f"    • {config.samples_per_sec:.0f} samples/second")
            if self.seq_len:
                print(f"    • {config.tokens_per_sec:.0f} tokens/second")
            print(f"    • {config.time_per_update*1000:.1f}ms per update")
            print(f"    • {config.memory_gb:.2f} GB memory")


def find_optimal_batch_config(
    model: nn.Module,
    sample_input_shape: Optional[Tuple[int, ...]] = None,
    vocab_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    target_effective_batch: Optional[int] = None,
    device: Optional[torch.device] = None,
    weight_speed: float = 0.3,
    weight_quality: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to find optimal batch configuration.

    Args:
        model: PyTorch model
        sample_input_shape: Shape of input tensor (excluding batch dimension).
            - For language models: Can omit if seq_len is provided (auto-inferred as (1, seq_len))
            - For image models: (channels, height, width) e.g., (3, 224, 224)
            - For other models: Full shape excluding batch dimension
        vocab_size: Vocabulary size (for language models)
        seq_len: Sequence length (for language models). If provided without sample_input_shape,
            automatically infers sample_input_shape=(1, seq_len)
        target_effective_batch: Target effective batch size (auto-estimated if None)
        device: Target device ('auto' or specific device)
        weight_speed: Weight for speed vs quality (0-1)
        weight_quality: Weight for training quality (0-1)
        **kwargs: Additional args for BatchSizeOptimizer

    Returns:
        Dictionary with recommended configuration and alternatives

    Examples:
        Language model (simplified):
        >>> model = GPT2()
        >>> result = find_optimal_batch_config(
        ...     model=model,
        ...     vocab_size=50257,
        ...     seq_len=128,  # No need for sample_input_shape!
        ... )

        Language model (explicit):
        >>> result = find_optimal_batch_config(
        ...     model=model,
        ...     sample_input_shape=(1, 128),
        ...     vocab_size=50257,
        ... )

        Image model:
        >>> model = ResNet50()
        >>> result = find_optimal_batch_config(
        ...     model=model,
        ...     sample_input_shape=(1, 3, 224, 224),
        ... )
    """
    # Auto-infer sample_input_shape for language models
    if sample_input_shape is None:
        if seq_len is not None:
            sample_input_shape = (1, seq_len)
        else:
            raise ValueError(
                "Must provide either sample_input_shape or seq_len. "
                "For language models, just pass seq_len and we'll infer the shape."
            )

    # Extract show_progress before passing kwargs to optimizer
    show_progress_val = kwargs.pop('show_progress', True)

    optimizer = BatchSizeOptimizer(
        model=model,
        sample_input_shape=sample_input_shape,
        vocab_size=vocab_size,
        seq_len=seq_len,
        device=device,
        **kwargs
    )

    return optimizer.find_optimal_config(
        target_effective_batch=target_effective_batch,
        weight_speed=weight_speed,
        weight_quality=weight_quality,
        show_progress=show_progress_val,
    )

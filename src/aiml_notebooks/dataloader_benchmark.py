"""
DataLoader benchmarking utilities for finding optimal configuration.

This module provides tools to automatically find the best DataLoader settings
for maximum throughput by testing different combinations of:
- num_workers: Number of parallel data loading processes
- persistent_workers: Whether to keep workers alive between epochs
- batch_size: Number of samples per batch

The benchmark measures wall-time throughput and returns the optimal configuration.
"""

import time
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any
import itertools
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    batch_size: int
    num_workers: int
    persistent_workers: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """String representation."""
        return f"batch_size={self.batch_size}, num_workers={self.num_workers}, persistent_workers={self.persistent_workers}"


@dataclass
class BenchmarkResult:
    """Result from benchmarking a DataLoader configuration."""
    config: DataLoaderConfig
    samples_per_second: float
    batches_per_second: float
    avg_batch_time: float
    total_time: float

    def __str__(self) -> str:
        """String representation."""
        return (
            f"Config: {self.config}\n"
            f"  Throughput: {self.samples_per_second:.1f} samples/sec, "
            f"{self.batches_per_second:.1f} batches/sec\n"
            f"  Avg batch time: {self.avg_batch_time*1000:.2f}ms"
        )


def benchmark_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    num_batches: int = 100,
    warmup_batches: int = 10,
    device: Optional[torch.device] = None,
    **dataloader_kwargs
) -> BenchmarkResult:
    """
    Benchmark a single DataLoader configuration.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        persistent_workers: Whether to keep workers alive
        num_batches: Number of batches to benchmark (after warmup)
        warmup_batches: Number of warmup batches (not counted)
        device: Device to transfer data to (optional)
        **dataloader_kwargs: Additional DataLoader arguments

    Returns:
        BenchmarkResult with throughput metrics
    """
    # Don't use persistent_workers if num_workers is 0
    if num_workers == 0:
        persistent_workers = False

    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        **dataloader_kwargs
    )

    # Warmup phase
    loader_iter = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(loader_iter)
            if device is not None:
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

    # Benchmark phase
    batch_times = []
    total_samples = 0

    start_time = time.perf_counter()

    loader_iter = iter(loader)
    for i in range(num_batches):
        try:
            batch_start = time.perf_counter()
            batch = next(loader_iter)

            # Transfer to device if specified
            if device is not None:
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(device)

            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)

            # Count samples in batch
            if isinstance(batch, (list, tuple)):
                batch_size_actual = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
            elif isinstance(batch, torch.Tensor):
                batch_size_actual = batch.shape[0]
            else:
                batch_size_actual = batch_size

            total_samples += batch_size_actual

        except StopIteration:
            # Restart if we run out of data
            loader_iter = iter(loader)
            batch = next(loader_iter)
            # Count this batch too
            if isinstance(batch, (list, tuple)):
                batch_size_actual = batch[0].shape[0] if isinstance(batch[0], torch.Tensor) else len(batch[0])
            elif isinstance(batch, torch.Tensor):
                batch_size_actual = batch.shape[0]
            else:
                batch_size_actual = batch_size
            total_samples += batch_size_actual

    total_time = time.perf_counter() - start_time

    # Calculate metrics
    samples_per_second = total_samples / total_time
    batches_per_second = num_batches / total_time
    avg_batch_time = np.mean(batch_times)

    # Clean up
    del loader

    return BenchmarkResult(
        config=config,
        samples_per_second=samples_per_second,
        batches_per_second=batches_per_second,
        avg_batch_time=avg_batch_time,
        total_time=total_time
    )


def find_optimal_dataloader_config(
    dataset: Dataset,
    batch_sizes: Optional[List[int]] = None,
    num_workers_list: Optional[List[int]] = None,
    test_persistent_workers: bool = True,
    num_batches: int = 100,
    warmup_batches: int = 10,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    **dataloader_kwargs
) -> Tuple[DataLoaderConfig, List[BenchmarkResult]]:
    """
    Find optimal DataLoader configuration by benchmarking different settings.

    This function tests various combinations of batch_size, num_workers, and
    persistent_workers to find the configuration that maximizes throughput.

    Args:
        dataset: Dataset to benchmark
        batch_sizes: List of batch sizes to test (default: [32, 64, 128, 256])
        num_workers_list: List of num_workers to test (default: [0, 2, 4, 8])
        test_persistent_workers: Whether to test persistent_workers (default: True)
        num_batches: Number of batches per benchmark (default: 100)
        warmup_batches: Number of warmup batches (default: 10)
        device: Device to transfer data to (optional, for realistic benchmarking)
        verbose: Whether to print progress (default: True)
        **dataloader_kwargs: Additional DataLoader arguments (e.g., pin_memory, collate_fn)

    Returns:
        Tuple of (best_config, all_results):
        - best_config: DataLoaderConfig with highest throughput
        - all_results: List of all BenchmarkResults, sorted by throughput (descending)

    Example:
        >>> from aiml_notebooks import create_dataset, find_optimal_dataloader_config
        >>>
        >>> # Load dataset
        >>> train_dataset, _ = create_dataset(dataset_id='mnist')
        >>>
        >>> # Find optimal config
        >>> best_config, all_results = find_optimal_dataloader_config(
        ...     dataset=train_dataset,
        ...     batch_sizes=[64, 128, 256],
        ...     num_workers_list=[0, 2, 4],
        ...     verbose=True
        ... )
        >>>
        >>> # Use best config
        >>> from torch.utils.data import DataLoader
        >>> train_loader = DataLoader(train_dataset, **best_config.to_dict())
    """
    # Default values
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256]

    if num_workers_list is None:
        # Auto-detect reasonable range based on CPU count
        import os
        cpu_count = os.cpu_count() or 4
        num_workers_list = [0, 2, min(4, cpu_count), min(8, cpu_count)]
        num_workers_list = sorted(set(num_workers_list))  # Remove duplicates

    # Generate configurations to test
    configs_to_test = []
    for batch_size, num_workers in itertools.product(batch_sizes, num_workers_list):
        if num_workers == 0:
            # Don't test persistent_workers with num_workers=0
            configs_to_test.append((batch_size, num_workers, False))
        else:
            configs_to_test.append((batch_size, num_workers, False))
            if test_persistent_workers:
                configs_to_test.append((batch_size, num_workers, True))

    if verbose:
        print("=" * 70)
        print("DataLoader Configuration Benchmark")
        print("=" * 70)
        print(f"Dataset size: {len(dataset)}")
        print(f"Configurations to test: {len(configs_to_test)}")
        print(f"Batches per test: {num_batches} (+ {warmup_batches} warmup)")
        if device is not None:
            print(f"Device: {device}")
        print()

    # Benchmark all configurations
    results = []
    for i, (batch_size, num_workers, persistent_workers) in enumerate(configs_to_test, 1):
        if verbose:
            print(f"[{i}/{len(configs_to_test)}] Testing: "
                  f"batch_size={batch_size}, num_workers={num_workers}, "
                  f"persistent_workers={persistent_workers}...", end=" ", flush=True)

        try:
            result = benchmark_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                num_batches=num_batches,
                warmup_batches=warmup_batches,
                device=device,
                **dataloader_kwargs
            )
            results.append(result)

            if verbose:
                print(f"{result.samples_per_second:.1f} samples/sec")

        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")

    # Sort by throughput (descending)
    results.sort(key=lambda r: r.samples_per_second, reverse=True)

    if verbose:
        print("\n" + "=" * 70)
        print("Results (sorted by throughput):")
        print("=" * 70)
        for i, result in enumerate(results[:5], 1):  # Show top 5
            print(f"\n{i}. {result.samples_per_second:.1f} samples/sec "
                  f"({result.batches_per_second:.1f} batches/sec)")
            print(f"   {result.config}")

        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more configurations")

        print("\n" + "=" * 70)
        print("Recommendation:")
        print("=" * 70)
        best_result = results[0]
        print(f"Use: {best_result.config}")
        print(f"Expected throughput: {best_result.samples_per_second:.1f} samples/sec")
        print("=" * 70)

    return results[0].config, results


def quick_benchmark_dataloader(
    dataset: Dataset,
    verbose: bool = True,
    **dataloader_kwargs
) -> DataLoaderConfig:
    """
    Quick benchmark with sensible defaults (fewer configurations tested).

    This is a convenience function that tests fewer configurations for faster results.

    Args:
        dataset: Dataset to benchmark
        verbose: Whether to print results (default: True)
        **dataloader_kwargs: Additional DataLoader arguments

    Returns:
        Best DataLoaderConfig

    Example:
        >>> config = quick_benchmark_dataloader(train_dataset)
        >>> train_loader = DataLoader(train_dataset, **config.to_dict())
    """
    import os
    cpu_count = os.cpu_count() or 4

    best_config, _ = find_optimal_dataloader_config(
        dataset=dataset,
        batch_sizes=[128],  # Test only one batch size
        num_workers_list=[0, min(4, cpu_count)],  # Test 0 and 4 workers
        test_persistent_workers=False,  # Skip persistent_workers
        num_batches=50,  # Fewer batches
        warmup_batches=5,
        verbose=verbose,
        **dataloader_kwargs
    )

    return best_config

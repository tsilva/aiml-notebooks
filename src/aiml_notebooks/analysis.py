"""
Latent space analysis and visualization utilities.

This module provides tools for analyzing learned representations:
- Dimensionality reduction (t-SNE, PCA, UMAP)
- Latent space visualization
- Cluster analysis
- Latent traversal and interpolation visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, Subset


def extract_latent_representations(
    model: torch.nn.Module,
    dataset: Union[Dataset, DataLoader],
    encode_fn: Optional[callable] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
    return_labels: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract latent representations from a model for a dataset.

    Args:
        model: Model with encode method
        dataset: Dataset or DataLoader to encode
        encode_fn: Optional custom encoding function (model, batch) -> latent
        max_samples: Maximum number of samples to encode (None = all)
        batch_size: Batch size for encoding
        device: Device to run on
        return_labels: Whether to return labels along with latents

    Returns:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Labels [n_samples] (if return_labels=True and available)

    Example:
        >>> latents, labels = extract_latent_representations(
        ...     vae, test_dataset, max_samples=5000
        ... )
        >>> print(f"Extracted {len(latents)} latent vectors")
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Default encoding function
    if encode_fn is None:
        encode_fn = lambda m, x: m.encode(x) if hasattr(m, 'encode') else m.encoder(x)

    # Create DataLoader if needed
    if isinstance(dataset, Dataset):
        if max_samples is not None and max_samples < len(dataset):
            dataset = Subset(dataset, range(max_samples))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        loader = dataset

    latents = []
    labels = [] if return_labels else None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents"):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    data, label = batch
                    if return_labels:
                        if isinstance(label, torch.Tensor):
                            labels.append(label.cpu().numpy())
                        else:
                            labels.append(np.array(label))
                else:
                    data = batch[0]
            elif isinstance(batch, dict):
                data = batch.get('input_ids', batch.get('images', batch.get('data')))
                if return_labels and 'labels' in batch:
                    labels.append(batch['labels'].cpu().numpy())
            else:
                data = batch

            # Move to device
            data = data.to(device)

            # Encode
            z = encode_fn(model, data)

            # Handle VAE outputs (mu, logvar)
            if isinstance(z, tuple):
                z = z[0]  # Take mean

            latents.append(z.cpu().numpy())

            # Limit samples if specified
            if max_samples is not None and len(latents) * batch_size >= max_samples:
                break

    # Concatenate results
    latents = np.concatenate(latents, axis=0)
    if return_labels and labels:
        labels = np.concatenate(labels, axis=0)

    # Flatten if needed (e.g., for CNNs with spatial dimensions)
    if latents.ndim > 2:
        latents = latents.reshape(latents.shape[0], -1)

    return (latents, labels) if return_labels else (latents, None)


def reduce_dimensions(
    latents: np.ndarray,
    method: str = 'tsne',
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensionality of latent vectors.

    Args:
        latents: High-dimensional latent vectors [n_samples, latent_dim]
        method: Reduction method ('tsne', 'pca', or 'umap')
        n_components: Number of dimensions to reduce to
        **kwargs: Additional arguments for the reduction method

    Returns:
        Reduced latent vectors [n_samples, n_components]

    Example:
        >>> latents_2d = reduce_dimensions(latents, method='tsne', perplexity=30)
        >>> plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels)
    """
    if method == 'tsne':
        default_kwargs = {'random_state': 42, 'perplexity': 30}
        default_kwargs.update(kwargs)
        reducer = TSNE(n_components=n_components, **default_kwargs)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
    elif method == 'umap':
        try:
            from umap import UMAP
            default_kwargs = {'random_state': 42, 'n_neighbors': 15}
            default_kwargs.update(kwargs)
            reducer = UMAP(n_components=n_components, **default_kwargs)
        except ImportError:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Running {method.upper()} dimensionality reduction...")
    latents_reduced = reducer.fit_transform(latents)

    return latents_reduced


def visualize_latent_space(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    n_components: int = 2,
    figsize: tuple = (12, 10),
    cmap: str = 'tab10',
    alpha: float = 0.6,
    s: float = 5,
    title: Optional[str] = None,
    **reduction_kwargs
):
    """
    Visualize latent space with dimensionality reduction.

    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Optional labels for coloring points
        class_names: Optional class names for legend
        method: Reduction method ('tsne', 'pca', 'umap')
        n_components: Number of dimensions (2 or 3)
        figsize: Figure size
        cmap: Color map
        alpha: Point transparency
        s: Point size
        title: Plot title
        **reduction_kwargs: Additional arguments for reduction method

    Example:
        >>> visualize_latent_space(
        ...     latents, labels, class_names=['Cat', 'Dog'],
        ...     method='tsne', perplexity=30
        ... )
    """
    # Reduce dimensions if needed
    if latents.shape[1] > n_components:
        latents_reduced = reduce_dimensions(
            latents, method=method, n_components=n_components, **reduction_kwargs
        )
    else:
        latents_reduced = latents

    # Create plot
    if n_components == 2:
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            latents_reduced[:, 0],
            latents_reduced[:, 1],
            c=labels if labels is not None else 'blue',
            cmap=cmap,
            alpha=alpha,
            s=s
        )

        if labels is not None:
            cbar = plt.colorbar(scatter)
            if class_names:
                cbar.set_label('Class')
            else:
                cbar.set_label('Label')

        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')

        # Add explained variance for PCA
        if method == 'pca' and hasattr(reduce_dimensions, '_last_reducer'):
            reducer = reduce_dimensions._last_reducer
            if hasattr(reducer, 'explained_variance_ratio_'):
                var1, var2 = reducer.explained_variance_ratio_[:2]
                plt.xlabel(f'PC1 ({var1*100:.1f}% var)')
                plt.ylabel(f'PC2 ({var2*100:.1f}% var)')

    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(
            latents_reduced[:, 0],
            latents_reduced[:, 1],
            latents_reduced[:, 2],
            c=labels if labels is not None else 'blue',
            cmap=cmap,
            alpha=alpha,
            s=s
        )

        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_zlabel(f'{method.upper()} Dimension 3')

        if labels is not None:
            plt.colorbar(scatter, label='Class' if class_names else 'Label')

    else:
        raise ValueError("n_components must be 2 or 3")

    if title is None:
        title = f'Latent Space Visualization ({method.upper()})'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print statistics
    if labels is not None:
        print(f"\nLatent space statistics:")
        print(f"  Total samples: {len(latents)}")
        print(f"  Unique classes: {len(np.unique(labels))}")
        if class_names:
            for i, name in enumerate(class_names):
                count = np.sum(labels == i)
                print(f"  {name}: {count} samples")


def analyze_latent_clusters(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    method: str = 'kmeans'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform clustering analysis on latent space.

    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Optional true labels for comparison
        n_clusters: Number of clusters (None = use number of unique labels)
        method: Clustering method ('kmeans', 'dbscan', 'gmm')

    Returns:
        cluster_labels: Cluster assignments [n_samples]
        cluster_centers: Cluster centers [n_clusters, latent_dim]

    Example:
        >>> cluster_labels, centers = analyze_latent_clusters(
        ...     latents, labels, n_clusters=10
        ... )
        >>> print(f"Found {len(centers)} clusters")
    """
    if n_clusters is None and labels is not None:
        n_clusters = len(np.unique(labels))
    elif n_clusters is None:
        n_clusters = 10  # Default

    if method == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(latents)
        cluster_centers = clusterer.cluster_centers_

    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(latents)
        # DBSCAN doesn't have explicit centers
        cluster_centers = None

    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(latents)
        cluster_centers = clusterer.means_

    else:
        raise ValueError(f"Unknown method: {method}")

    # Print statistics
    print(f"\nClustering Results ({method}):")
    print(f"  Number of clusters: {len(np.unique(cluster_labels))}")
    for i in np.unique(cluster_labels):
        count = np.sum(cluster_labels == i)
        print(f"  Cluster {i}: {count} samples")

    # Compare with true labels if available
    if labels is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        print(f"\n  Adjusted Rand Index: {ari:.3f}")
        print(f"  Normalized Mutual Info: {nmi:.3f}")

    return cluster_labels, cluster_centers


def latent_traversal(
    model: torch.nn.Module,
    base_latent: torch.Tensor,
    dim_idx: int,
    n_steps: int = 10,
    range_scale: float = 3.0,
    decode_fn: Optional[callable] = None,
    device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Traverse a single dimension in latent space.

    Useful for understanding what each latent dimension controls.

    Args:
        model: Model with decode method
        base_latent: Base latent vector [latent_dim]
        dim_idx: Index of dimension to traverse
        n_steps: Number of steps
        range_scale: Range to traverse (in standard deviations)
        decode_fn: Optional custom decoding function
        device: Device to run on

    Returns:
        List of decoded outputs

    Example:
        >>> # See what dimension 0 controls
        >>> outputs = latent_traversal(vae, base_z, dim_idx=0, n_steps=10)
        >>> plot_image_grid(outputs)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if decode_fn is None:
        decode_fn = lambda m, z: m.decode(z) if hasattr(m, 'decode') else m.decoder(z)

    base_latent = base_latent.to(device)
    if base_latent.dim() == 1:
        base_latent = base_latent.unsqueeze(0)

    outputs = []
    values = np.linspace(-range_scale, range_scale, n_steps)

    with torch.no_grad():
        for val in values:
            z = base_latent.clone()
            z[0, dim_idx] = val
            output = decode_fn(model, z)
            outputs.append(output.squeeze(0).cpu())

    return outputs


def compute_latent_statistics(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> dict:
    """
    Compute statistics about latent space.

    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Optional labels for per-class statistics

    Returns:
        Dictionary with statistics

    Example:
        >>> stats = compute_latent_statistics(latents, labels)
        >>> print(f"Mean norm: {stats['mean_norm']:.3f}")
        >>> print(f"Std per dim: {stats['std_per_dim']}")
    """
    stats = {
        'mean': np.mean(latents, axis=0),
        'std': np.std(latents, axis=0),
        'mean_norm': np.mean(np.linalg.norm(latents, axis=1)),
        'std_norm': np.std(np.linalg.norm(latents, axis=1)),
        'min_per_dim': np.min(latents, axis=0),
        'max_per_dim': np.max(latents, axis=0),
        'std_per_dim': np.std(latents, axis=0),
    }

    # Per-class statistics
    if labels is not None:
        stats['per_class'] = {}
        for label in np.unique(labels):
            mask = labels == label
            class_latents = latents[mask]
            stats['per_class'][int(label)] = {
                'mean': np.mean(class_latents, axis=0),
                'std': np.std(class_latents, axis=0),
                'count': np.sum(mask)
            }

    return stats


def track_block_activations(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer_norm_enabled: bool = True
) -> List[dict]:
    """
    Track activation statistics at the output of each transformer block.
    
    Useful for diagnosing training issues like vanishing/exploding activations.
    Monitors mean, standard deviation, and maximum absolute values through the
    forward pass of a transformer model.
    
    The model is expected to have:
    - token_embedding_table: nn.Embedding for token embeddings
    - position_embedding_table: nn.Embedding for position embeddings
    - blocks: nn.ModuleList of transformer blocks
    - ln_final: Optional final layer norm
    
    Args:
        model: Transformer model with the expected structure
        x: Input tensor of token ids with shape (batch_size, seq_len)
        layer_norm_enabled: Whether final layer norm should be applied
    
    Returns:
        List of dictionaries, each containing:
        - layer: Name of the layer ('input', 'block_0', 'block_1', ..., 'final'/'final_ln')
        - mean: Mean activation value
        - std: Standard deviation of activations
        - max_abs: Maximum absolute activation value
        
    Example:
        >>> model = Transformer()
        >>> x = torch.randint(0, 50257, (1, 128))  # Random tokens
        >>> stats = track_block_activations(model, x)
        >>> for s in stats:
        ...     print(f"{s['layer']}: mean={s['mean']:.4f}, std={s['std']:.4f}")
        
        >>> # Visualize activation growth
        >>> import matplotlib.pyplot as plt
        >>> max_abs = [s['max_abs'] for s in stats]
        >>> plt.semilogy(max_abs, marker='o')
        >>> plt.xlabel('Layer')
        >>> plt.ylabel('Max Absolute Activation')
        >>> plt.show()
    """
    stats = []
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        x_emb = model.token_embedding_table(x)
        pos = torch.arange(x_emb.size(1), device=x.device)
        pos_emb = model.position_embedding_table(pos).unsqueeze(0)
        hidden = x_emb + pos_emb
        
        # Track initial state
        stats.append({
            'layer': 'input',
            'mean': hidden.mean().item(),
            'std': hidden.std().item(),
            'max_abs': hidden.abs().max().item()
        })
        
        # Track through each block
        for i, block in enumerate(model.blocks):
            result = block(hidden)
            # Handle blocks that return (output, kv_cache) tuple
            hidden = result[0] if isinstance(result, tuple) else result
            stats.append({
                'layer': f'block_{i}',
                'mean': hidden.mean().item(),
                'std': hidden.std().item(),
                'max_abs': hidden.abs().max().item()
            })
        
        # Apply final layer norm if enabled
        if layer_norm_enabled and hasattr(model, 'ln_final'):
            hidden = model.ln_final(hidden)
            stats.append({
                'layer': 'final_ln',
                'mean': hidden.mean().item(),
                'std': hidden.std().item(),
                'max_abs': hidden.abs().max().item()
            })
        else:
            stats.append({
                'layer': 'final',
                'mean': hidden.mean().item(),
                'std': hidden.std().item(),
                'max_abs': hidden.abs().max().item()
            })
    
    return stats

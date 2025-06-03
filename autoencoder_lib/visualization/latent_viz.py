"""
Latent Space Visualization Functions

Functions for visualizing and analyzing latent space representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
from typing import List, Tuple, Optional, Union, Dict
import seaborn as sns


def visualize_latent_space_2d(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    figure_size: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Visualize latent space in 2D using t-SNE or PCA.
    
    Args:
        latent_vectors: Latent representations (N, latent_dim)
        labels: Corresponding labels
        class_names: Names for each class
        method: 'tsne' or 'pca'
        figure_size: Size of the figure
        title: Title for the plot
        save_path: Path to save the plot
        
    Returns:
        Tuple of (2D embedding, silhouette score)
    """
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        embedding_2d = reducer.fit_transform(latent_vectors)
        method_name = 't-SNE'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(latent_vectors)
        method_name = f'PCA (explained variance: {reducer.explained_variance_ratio_.sum():.3f})'
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    # Calculate silhouette score
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(embedding_2d, labels)
    else:
        silhouette = 0.0
    
    # Create visualization
    plt.figure(figsize=figure_size)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=[colors[i]], label=label_name, alpha=0.7, s=50)
    
    if title is None:
        title = f'{method_name} Visualization of Latent Space\nSilhouette Score: {silhouette:.4f}'
    
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel(f'{method_name} Component 1')
    plt.ylabel(f'{method_name} Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return embedding_2d, silhouette


def compare_latent_spaces(
    latent_vectors_list: List[Union[torch.Tensor, np.ndarray]],
    labels_list: List[Union[torch.Tensor, np.ndarray]],
    model_names: List[str],
    class_names: Optional[List[str]] = None,
    method: str = 'tsne',
    figure_size: Tuple[int, int] = (18, 6)
) -> List[float]:
    """
    Compare latent spaces from different models.
    
    Args:
        latent_vectors_list: List of latent representations from different models
        labels_list: List of corresponding labels
        model_names: Names of the models
        class_names: Names for each class
        method: 'tsne' or 'pca'
        figure_size: Size of the figure
        
    Returns:
        List of silhouette scores for each model
    """
    num_models = len(latent_vectors_list)
    fig, axes = plt.subplots(1, num_models, figsize=figure_size)
    
    if num_models == 1:
        axes = [axes]
    
    silhouette_scores = []
    
    for i, (latent_vectors, labels, model_name) in enumerate(zip(latent_vectors_list, labels_list, model_names)):
        # Convert to numpy if needed
        if isinstance(latent_vectors, torch.Tensor):
            latent_vectors = latent_vectors.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
            embedding_2d = reducer.fit_transform(latent_vectors)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(latent_vectors)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(embedding_2d, labels)
        else:
            silhouette = 0.0
        silhouette_scores.append(silhouette)
        
        # Plot
        ax = axes[i]
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
            
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                      c=[colors[j]], label=label_name, alpha=0.7, s=30)
        
        ax.set_title(f'{model_name}\nSilhouette: {silhouette:.4f}', fontsize=12)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
    
    plt.suptitle(f'{method.upper()} Comparison of Latent Spaces', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return silhouette_scores


def plot_latent_distribution(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    max_dims: int = 6,
    figure_size: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot distributions of latent dimensions.
    
    Args:
        latent_vectors: Latent representations (N, latent_dim)
        labels: Corresponding labels
        class_names: Names for each class
        max_dims: Maximum number of dimensions to plot
        figure_size: Size of the figure
    """
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    latent_dim = latent_vectors.shape[1]
    dims_to_plot = min(max_dims, latent_dim)
    
    # Create subplots
    cols = min(3, dims_to_plot)
    rows = (dims_to_plot + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for dim in range(dims_to_plot):
        ax = axes[dim]
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
            
            ax.hist(latent_vectors[mask, dim], bins=20, alpha=0.6, 
                   color=colors[i], label=label_name, density=True)
        
        ax.set_title(f'Latent Dimension {dim}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        if dim == 0:
            ax.legend()
    
    # Hide unused subplots
    for i in range(dims_to_plot, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Latent Space Distributions by Dimension', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_latent_interpolation(
    model: torch.nn.Module,
    start_latent: Union[torch.Tensor, np.ndarray],
    end_latent: Union[torch.Tensor, np.ndarray],
    num_steps: int = 8,
    device: torch.device = torch.device('cpu'),
    figure_size: Tuple[int, int] = (16, 4)
) -> None:
    """
    Visualize interpolation between two points in latent space.
    
    Args:
        model: Trained autoencoder model
        start_latent: Starting latent vector
        end_latent: Ending latent vector
        num_steps: Number of interpolation steps
        device: Device to run model on
        figure_size: Size of the figure
    """
    # Convert to tensors if needed
    if isinstance(start_latent, np.ndarray):
        start_latent = torch.from_numpy(start_latent).float()
    if isinstance(end_latent, np.ndarray):
        end_latent = torch.from_numpy(end_latent).float()
    
    # Move to device
    start_latent = start_latent.to(device)
    end_latent = end_latent.to(device)
    model = model.to(device)
    model.eval()
    
    # Create interpolation
    alphas = np.linspace(0, 1, num_steps)
    interpolated_latents = []
    
    for alpha in alphas:
        interpolated = (1 - alpha) * start_latent + alpha * end_latent
        interpolated_latents.append(interpolated)
    
    # Generate reconstructions
    reconstructions = []
    with torch.no_grad():
        for latent in interpolated_latents:
            if hasattr(model, 'decoder'):
                reconstruction = model.decoder(latent.unsqueeze(0))
            else:
                # If model doesn't have separate decoder, assume it's the full model
                reconstruction = model(latent.unsqueeze(0))
            reconstructions.append(reconstruction.squeeze(0))
    
    # Plot interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=figure_size)
    
    for i, (alpha, reconstruction) in enumerate(zip(alphas, reconstructions)):
        ax = axes[i]
        
        # Convert to numpy and handle different image formats
        img = reconstruction.detach().cpu().numpy()
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        
        ax.set_title(f'α={alpha:.2f}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_latent_clustering(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (15, 10)
) -> Dict[str, float]:
    """
    Analyze clustering quality in latent space.
    
    Args:
        latent_vectors: Latent representations
        labels: Corresponding labels
        class_names: Names for each class
        figure_size: Size of the figure
        
    Returns:
        Dictionary of clustering metrics
    """
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans
    
    # Calculate various clustering metrics
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Silhouette score
    silhouette = silhouette_score(latent_vectors, labels)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predicted_labels = kmeans.fit_predict(latent_vectors)
    
    # Clustering metrics
    ari = adjusted_rand_score(labels, predicted_labels)
    nmi = normalized_mutual_info_score(labels, predicted_labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Original clustering (t-SNE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
    embedding_2d = tsne.fit_transform(latent_vectors)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # True labels
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        axes[0, 0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                          c=[colors[i]], label=label_name, alpha=0.7, s=30)
    
    axes[0, 0].set_title(f'True Labels\nSilhouette: {silhouette:.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # K-means clustering
    for i, label in enumerate(unique_labels):
        mask = predicted_labels == label
        axes[0, 1].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                          c=[colors[i]], alpha=0.7, s=30)
    
    axes[0, 1].set_title(f'K-means Clustering\nARI: {ari:.4f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics comparison
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = axes[1, 0].bar(range(len(metric_names)), metric_values, alpha=0.7)
    axes[1, 0].set_title('Clustering Metrics')
    axes[1, 0].set_xticks(range(len(metric_names)))
    axes[1, 0].set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{value:.3f}', ha='center', va='bottom')
    
    # Cluster centers analysis
    if latent_vectors.shape[1] >= 2:
        # Show cluster centers in first 2 dimensions
        class_centers = []
        for label in unique_labels:
            mask = labels == label
            center = np.mean(latent_vectors[mask], axis=0)
            class_centers.append(center)
        
        class_centers = np.array(class_centers)
        
        # Plot first two dimensions
        for i, (label, center) in enumerate(zip(unique_labels, class_centers)):
            label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
            axes[1, 1].scatter(center[0], center[1], c=[colors[i]], s=200, 
                             marker='*', label=label_name, edgecolors='black', linewidth=2)
        
        axes[1, 1].set_title('Class Centers (First 2 Dims)')
        axes[1, 1].set_xlabel('Latent Dim 0')
        axes[1, 1].set_ylabel('Latent Dim 1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.suptitle('Latent Space Clustering Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return metrics


def plot_latent_variance_analysis(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (15, 8)
) -> None:
    """
    Analyze variance in different latent dimensions.
    
    Args:
        latent_vectors: Latent representations
        labels: Corresponding labels
        class_names: Names for each class
        figure_size: Size of the figure
    """
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    latent_dim = latent_vectors.shape[1]
    unique_labels = np.unique(labels)
    
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    
    # Overall variance per dimension
    variances = np.var(latent_vectors, axis=0)
    axes[0].bar(range(latent_dim), variances, alpha=0.7)
    axes[0].set_title('Variance per Latent Dimension')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Variance')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Variance by class
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_variances = np.var(latent_vectors[mask], axis=0)
        label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        axes[1].plot(range(latent_dim), class_variances, 
                    color=colors[i], marker='o', label=label_name, alpha=0.7)
    
    axes[1].set_title('Variance by Class')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative explained variance (PCA-style)
    pca = PCA()
    pca.fit(latent_vectors)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    axes[2].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                'o-', linewidth=2, markersize=6)
    axes[2].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
    axes[2].axhline(y=0.99, color='r', linestyle=':', alpha=0.7, label='99% threshold')
    
    axes[2].set_title('Cumulative Explained Variance')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Cumulative Explained Variance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 
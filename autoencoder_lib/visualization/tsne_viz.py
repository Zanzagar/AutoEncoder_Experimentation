"""
t-SNE Visualization Functions

This module contains functions for creating t-SNE projections and visualizations:
- Raw data t-SNE projections
- Latent space t-SNE projections  
- Comparative t-SNE visualizations
- Interactive t-SNE exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from PIL import Image
import torch
import os
import time
from typing import Dict, List, Tuple, Optional, Union, Any

# Import data loading utility
from ..data.layered_geological import load_layered_dataset


def visualize_raw_data_tsne(
    dataset_info: Optional[Dict] = None,
    dataset_path: Optional[str] = None,
    random_state: int = 42,
    perplexity: Optional[int] = None,
    max_iter: int = 1000,
    max_samples: Optional[int] = None,
    figure_size: Tuple[int, int] = (12, 10)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Visualize raw dataset using t-SNE.
    
    Args:
        dataset_info: Pre-loaded dataset info dictionary
        dataset_path: Path to dataset (alternative to dataset_info)
        random_state: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter (auto-calculated if None)
        max_iter: Number of t-SNE iterations
        max_samples: Maximum samples to use (for performance)
        figure_size: Size of the figure
        
    Returns:
        Tuple of (t-SNE embedding, labels, silhouette score)
    """
    print(f"📊 Visualizing raw dataset with t-SNE...")
    
    # Load dataset info if not provided
    if dataset_info is None:
        if dataset_path is None:
            raise ValueError("Either dataset_info or dataset_path must be provided")
        dataset_info = load_layered_dataset(dataset_path)
    
    # Extract data from the dataset structure
    filenames = dataset_info['filenames']
    labels = dataset_info['labels']
    class_names = dataset_info.get('label_names', dataset_info.get('class_names', None))
    
    n_total_samples = len(filenames)
    
    # Limit samples if specified
    if max_samples and n_total_samples > max_samples:
        indices = np.random.RandomState(random_state).choice(n_total_samples, max_samples, replace=False)
        selected_filenames = [filenames[i] for i in indices]
        selected_labels = labels[indices]
        print(f"📉 Using {max_samples} random samples from {n_total_samples} total")
    else:
        selected_filenames = filenames
        selected_labels = labels
    
    n_samples = len(selected_filenames)
    
    # Load and flatten the images
    print(f"Loading {n_samples} images...")
    images = []
    final_labels = []
    
    for i, (filename, label) in enumerate(zip(selected_filenames, selected_labels)):
        try:
            # Handle both absolute and relative paths
            if not os.path.exists(filename):
                # Try with dataset_path as base
                if dataset_path:
                    filename = os.path.join(dataset_path, os.path.basename(filename))
            
            if os.path.exists(filename):
                img = Image.open(filename).convert('L')  # Convert to grayscale
                img_array = np.array(img).flatten() / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                final_labels.append(label)
            else:
                print(f"Warning: Image {filename} not found, skipping...")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    if len(images) == 0:
        raise ValueError("No images could be loaded from the dataset")
    
    # Convert to numpy arrays
    flattened_data = np.array(images)
    all_labels = np.array(final_labels)
    n_samples = len(flattened_data)
    
    # Auto-calculate perplexity if not provided
    if perplexity is None:
        perplexity = min(30, n_samples - 1)
    
    print(f"Computing t-SNE with perplexity={perplexity} on {n_samples} samples...")
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter,
        init='pca',
        learning_rate='auto'
    )
    
    embedding = tsne.fit_transform(flattened_data)
    
    # Calculate silhouette score
    silhouette = None
    if len(np.unique(all_labels)) > 1:
        try:
            silhouette = silhouette_score(embedding, all_labels)
            print(f"Silhouette score: {silhouette:.6f}")
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
    
    # Create visualization
    plt.figure(figsize=figure_size)
    
    # Get unique labels and colors using rainbow colormap like the reference notebook
    unique_labels = np.unique(all_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        plt.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Create title with data count like the reference notebook: "t-SNE Visualization of All Raw Image Data (n={len(X)})"
    title_with_info = f"t-SNE Visualization of All Raw Image Data (n={n_samples})"
    if silhouette is not None:
        title_with_info += f"\nSilhouette Score: {silhouette:.3f}"
    
    plt.title(title_with_info, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return embedding, all_labels, silhouette if silhouette is not None else 0.0


def visualize_latent_tsne(
    latent_representations: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
    perplexity: Optional[int] = None,
    max_iter: int = 1000,
    figure_size: Tuple[int, int] = (12, 10),
    title: str = "t-SNE Visualization of Latent Space"
) -> Tuple[np.ndarray, float]:
    """
    Create a t-SNE visualization of latent representations.
    
    Args:
        latent_representations: Latent space representations (N, latent_dim)
        labels: Class labels for each representation
        class_names: Names for each class
        random_state: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter (auto-calculated if None)
        max_iter: Number of t-SNE iterations
        figure_size: Size of the figure
        title: Title for the plot
        
    Returns:
        Tuple of (t-SNE embedding, silhouette score)
    """
    n_samples = len(latent_representations)
    
    # Auto-calculate perplexity if not provided
    if perplexity is None:
        perplexity = min(30, n_samples - 1)
    
    print(f"Computing t-SNE with perplexity={perplexity} on {n_samples} samples...")
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter,
        init='pca',
        learning_rate='auto'
    )
    
    embedding = tsne.fit_transform(latent_representations)
    
    # Calculate silhouette score
    silhouette = None
    if len(np.unique(labels)) > 1:
        try:
            silhouette = silhouette_score(embedding, labels)
            print(f"Silhouette score: {silhouette:.6f}")
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
    
    # Create visualization
    plt.figure(figsize=figure_size)
    
    # Get unique labels and colors using rainbow colormap like the reference notebook
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        plt.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            c=[colors[i]],
            label=label_name,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add data point count to title like the reference notebook
    title_with_count = f"{title} (n={n_samples})"
    if silhouette is not None:
        title_with_count += f"\nSilhouette Score: {silhouette:.3f}"
    
    plt.title(title_with_count, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return embedding, silhouette if silhouette is not None else 0.0


def compare_tsne_embeddings(
    embeddings_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    titles: List[str],
    class_names: Optional[List[str]] = None,
    figure_size: Tuple[int, int] = (18, 6)
) -> None:
    """
    Compare multiple t-SNE embeddings side by side.
    
    Args:
        embeddings_list: List of 2D t-SNE embeddings
        labels_list: List of corresponding labels
        titles: List of titles for each plot
        class_names: List of class names for the legend
        figure_size: Size of the figure
    """
    n_plots = len(embeddings_list)
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=figure_size)
    if n_plots == 1:
        axes = [axes]
    
    for i, (embeddings, labels, base_title) in enumerate(zip(embeddings_list, labels_list, titles)):
        # Add data count to title
        n_samples = len(embeddings)
        title_with_count = f"{base_title} (n={n_samples})"
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            try:
                silhouette = silhouette_score(embeddings, labels)
                title_with_count += f"\nSilhouette Score: {silhouette:.3f}"
            except Exception:
                pass
        
        plot_with_labels(embeddings, labels, class_names, title_with_count, axes[i])
    
    plt.tight_layout()
    plt.show()


def interactive_tsne_exploration(
    data: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    perplexity_range: Tuple[int, int] = (5, 50),
    n_perplexities: int = 4,
    random_state: int = 42,
    figure_size: Tuple[int, int] = (16, 12)
) -> None:
    """
    Explore t-SNE with different perplexity values.
    
    Args:
        data: Input data for t-SNE
        labels: Corresponding labels
        class_names: Names for each class
        perplexity_range: Range of perplexity values to explore
        n_perplexities: Number of different perplexity values to try
        random_state: Random seed for reproducibility
        figure_size: Size of the figure
    """
    perplexities = np.linspace(perplexity_range[0], perplexity_range[1], n_perplexities, dtype=int)
    
    plt.figure(figsize=figure_size)
    
    for i, perplexity in enumerate(perplexities):
        print(f"Computing t-SNE with perplexity={perplexity}...")
        
        # Ensure perplexity is valid
        max_perplexity = min(perplexity, len(data) - 1)
        
        # Use max_iter instead of deprecated n_iter
        tsne = TSNE(n_components=2, perplexity=max_perplexity, max_iter=1000, random_state=random_state)
        embedding = tsne.fit_transform(data)
        
        plt.subplot(2, 2, i + 1)
        
        # Create class names if not provided
        if class_names is None:
            current_class_names = [f"Class {j}" for j in sorted(np.unique(labels))]
        else:
            current_class_names = class_names
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for j, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[colors[j]],
                label=current_class_names[label] if label < len(current_class_names) else f"Class {label}",
                alpha=0.7,
                s=30,
                edgecolors='none'
            )
        
        # Calculate silhouette score
        if len(unique_labels) > 1:
            silhouette = silhouette_score(embedding, labels)
            title = f"Perplexity={max_perplexity}\nSilhouette: {silhouette:.3f}"
        else:
            title = f"Perplexity={max_perplexity}"
        
        plt.title(title, fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Only show legend on first subplot
        if i == 0:
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle("t-SNE Exploration with Different Perplexity Values", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_with_labels(
    low_dim_embs: np.ndarray, 
    labels: np.ndarray, 
    class_names: Optional[List[str]] = None, 
    title: str = "t-SNE Visualization", 
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot t-SNE data with color-coded labels on a given axis.
    
    Args:
        low_dim_embs: Low-dimensional embeddings from t-SNE (N, 2)
        labels: Class labels for each point (N,)
        class_names: List of class names for the legend
        title: Plot title
        ax: Matplotlib axis to plot on (if None, creates a new figure)
        
    Returns:
        The matplotlib axis used for plotting
    """
    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
    unique_labels = np.unique(labels)
    # Use rainbow colormap for consistent visualization
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            low_dim_embs[mask, 0], 
            low_dim_embs[mask, 1],
            c=[colors[i]],
            label=class_names[label] if class_names is not None else f'Class {label}',
            alpha=0.7,
            s=50,
            edgecolors='none'
        )
    
    ax.legend()
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    return ax


def visualize_side_by_side_latent_spaces(
    model,
    train_data: Union[torch.Tensor, np.ndarray],
    train_labels: Union[torch.Tensor, np.ndarray],
    comparison_data: Union[torch.Tensor, np.ndarray],
    comparison_labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    title_suffix: str = "",
    orig_silhouette: Optional[float] = None,
    max_samples: int = 500,
    device: str = 'cpu',
    figure_size: Tuple[int, int] = (20, 16),
    grid_layout: str = "2x2",
    verbose: bool = False,
    # Legacy parameters for backward compatibility
    test_data: Union[torch.Tensor, np.ndarray, None] = None,
    test_labels: Union[torch.Tensor, np.ndarray, None] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute and visualize t-SNE projections of train and comparison data with both latent space and reconstructed images.
    
    This function can be used in two contexts:
    1. During training: comparing train vs validation data for monitoring
    2. Final evaluation: comparing train vs test data for unbiased assessment
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels tensor
        comparison_data: Comparison data tensor (validation or test data)
        comparison_labels: Comparison labels tensor (validation or test labels)
        class_names: List of class names for visualization
        title_suffix: Additional text for plot title (should specify context: validation/test)
        orig_silhouette: Original silhouette score from training
        max_samples: Maximum samples to use for t-SNE (for performance)
        device: Device to run model on
        figure_size: Size of the figure
        grid_layout: Layout type ("1x2" for side-by-side latent only, "2x2" for latent + reconstructed)
        verbose: Whether to print progress messages (default: False for cleaner output)
        test_data: Legacy parameter for backward compatibility (use comparison_data instead)
        test_labels: Legacy parameter for backward compatibility (use comparison_labels instead)
        
    Returns:
        Tuple of (train_silhouette, comparison_silhouette) scores
    """
    import torch
    from sklearn.metrics import silhouette_score
    
    # Handle legacy parameters for backward compatibility
    if test_data is not None and comparison_data is None:
        comparison_data = test_data
    if test_labels is not None and comparison_labels is None:
        comparison_labels = test_labels
    
    model.eval()
    
    # Convert to tensors if needed
    if isinstance(train_data, np.ndarray):
        train_data = torch.FloatTensor(train_data)
    if isinstance(comparison_data, np.ndarray):
        comparison_data = torch.FloatTensor(comparison_data)
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.detach().cpu().numpy()
    if isinstance(comparison_labels, torch.Tensor):
        comparison_labels = comparison_labels.detach().cpu().numpy()
    
    with torch.no_grad():
        # Process train data
        train_data_device = train_data.to(device)
        try:
            encoded_train, _ = model(train_data_device)
        except:
            # Try alternative interface
            encoded_train = model.encode(train_data_device)
        
        encoded_train = encoded_train.view(encoded_train.size(0), -1).detach().cpu().numpy()
        
        # Process comparison data
        comparison_data_device = comparison_data.to(device)
        try:
            encoded_comparison, _ = model(comparison_data_device)
        except:
            # Try alternative interface
            encoded_comparison = model.encode(comparison_data_device)
        
        encoded_comparison = encoded_comparison.view(encoded_comparison.size(0), -1).detach().cpu().numpy()
        
        # Restrict to max samples for performance
        train_plot_only = min(max_samples, encoded_train.shape[0])
        comparison_plot_only = min(max_samples, encoded_comparison.shape[0])
        
        if verbose:
            print(f"Running t-SNE on {train_plot_only} train samples and {comparison_plot_only} comparison samples...")
            print(f"Train latent space: shape={encoded_train.shape}, std={encoded_train.std():.4f}, mean={encoded_train.mean():.4f}")
            print(f"Comparison latent space: shape={encoded_comparison.shape}, std={encoded_comparison.std():.4f}, mean={encoded_comparison.mean():.4f}")
            print(f"Train perplexity: {train_perplexity}, Comparison perplexity: {comparison_perplexity}")
            print(f"Train classes: {np.unique(train_plot_labels, return_counts=True)}")
            print(f"Comparison classes: {np.unique(comparison_plot_labels, return_counts=True)}")
        
        # Apply t-SNE with consistent parameters
        # Train data t-SNE - improved perplexity calculation
        train_perplexity = min(30, max(5, len(encoded_train[:train_plot_only])//4))
        train_tsne = TSNE(perplexity=train_perplexity, 
                         n_components=2, init='pca', max_iter=5000, random_state=42)
        train_low_dim_embs = train_tsne.fit_transform(encoded_train[:train_plot_only])
        train_plot_labels = train_labels[:train_plot_only]
        
        # Comparison data t-SNE - use different random seed and improved perplexity
        comparison_perplexity = min(30, max(5, len(encoded_comparison[:comparison_plot_only])//4))
        comparison_tsne = TSNE(perplexity=comparison_perplexity, 
                        n_components=2, init='pca', max_iter=5000, random_state=123)  # Different seed
        comparison_low_dim_embs = comparison_tsne.fit_transform(encoded_comparison[:comparison_plot_only])
        comparison_plot_labels = comparison_labels[:comparison_plot_only]
        
        # Calculate silhouette scores
        train_silhouette = None
        if len(np.unique(train_plot_labels)) > 1:
            try:
                train_silhouette = silhouette_score(train_low_dim_embs, train_plot_labels)
                if verbose:
                    print(f"Train data silhouette score: {train_silhouette:.6f}")
            except Exception as e:
                if verbose:
                    print(f"Could not calculate train silhouette score: {e}")
        
        comparison_silhouette = None
        if len(np.unique(comparison_plot_labels)) > 1:
            try:
                comparison_silhouette = silhouette_score(comparison_low_dim_embs, comparison_plot_labels)
                if verbose:
                    print(f"Comparison data silhouette score: {comparison_silhouette:.6f}")
                
                # Check if it matches original silhouette
                if orig_silhouette is not None and verbose:
                    similarity = 100 - abs(comparison_silhouette - orig_silhouette) * 100
                    print(f"Similarity to original score: {similarity:.2f}%")
            except Exception as e:
                if verbose:
                    print(f"Could not calculate comparison silhouette score: {e}")
        
        # Determine context from title_suffix for proper labeling
        comparison_type = "Comparison"  # Default fallback
        if "validation" in title_suffix.lower():
            comparison_type = "Validation"
        elif "test" in title_suffix.lower():
            comparison_type = "Test"
        
        # Determine layout based on grid_layout parameter
        if grid_layout == "2x2":
            # Generate reconstructed images for both train and comparison data
            if verbose:
                print("Generating reconstructed images for t-SNE analysis...")
            
            with torch.no_grad():
                # Get reconstructed train data
                try:
                    _, train_reconstructed = model(train_data_device)
                except:
                    train_reconstructed = model.decode(encoded_train)
                train_reconstructed = train_reconstructed.view(train_reconstructed.size(0), -1).detach().cpu().numpy()
                
                # Get reconstructed comparison data  
                try:
                    _, comparison_reconstructed = model(comparison_data_device)
                except:
                    comparison_reconstructed = model.decode(encoded_comparison)
                comparison_reconstructed = comparison_reconstructed.view(comparison_reconstructed.size(0), -1).detach().cpu().numpy()
            
            # Compute t-SNE for reconstructed images
            if verbose:
                print("Computing t-SNE for reconstructed images...")
            train_recon_perplexity = min(30, max(5, len(train_reconstructed[:train_plot_only])//4))
            train_recon_tsne = TSNE(perplexity=train_recon_perplexity, 
                                   n_components=2, init='pca', max_iter=5000, random_state=456)  # Different seed
            train_recon_low_dim = train_recon_tsne.fit_transform(train_reconstructed[:train_plot_only])
            
            comparison_recon_perplexity = min(30, max(5, len(comparison_reconstructed[:comparison_plot_only])//4))
            comparison_recon_tsne = TSNE(perplexity=comparison_recon_perplexity, 
                                         n_components=2, init='pca', max_iter=5000, random_state=789)  # Different seed
            comparison_recon_low_dim = comparison_recon_tsne.fit_transform(comparison_reconstructed[:comparison_plot_only])
            
            # Create 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=figure_size)
            axes = axes.flatten()
            
            # Get colors once
            unique_train_labels = np.unique(train_plot_labels)
            unique_comparison_labels = np.unique(comparison_plot_labels)
            train_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_train_labels)))
            comparison_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_comparison_labels)))
            
            # Plot 1: Train Latent Space
            for i, label in enumerate(unique_train_labels):
                mask = train_plot_labels == label
                axes[0].scatter(
                    train_low_dim_embs[mask, 0], train_low_dim_embs[mask, 1],
                    c=[train_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            train_title = f"Train Latent Space ({len(train_data)} images)"
            if train_silhouette is not None:
                train_title += f"\nSilhouette Score: {train_silhouette:.3f}"
            axes[0].set_title(train_title, fontsize=12)
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')
            
            # Plot 2: Comparison Latent Space (with proper context)
            for i, label in enumerate(unique_comparison_labels):
                mask = comparison_plot_labels == label
                axes[1].scatter(
                    comparison_low_dim_embs[mask, 0], comparison_low_dim_embs[mask, 1],
                    c=[comparison_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            comparison_title = f"{comparison_type} Latent Space ({len(comparison_data)} images)"
            if comparison_silhouette is not None:
                comparison_title += f"\nSilhouette Score: {comparison_silhouette:.3f}"
            if orig_silhouette is not None:
                comparison_title += f" (Original: {orig_silhouette:.3f})"
            axes[1].set_title(comparison_title, fontsize=14)
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
            
            # Plot 3: Train Reconstructed Images
            for i, label in enumerate(unique_train_labels):
                mask = train_plot_labels == label
                axes[2].scatter(
                    train_recon_low_dim[mask, 0], train_recon_low_dim[mask, 1],
                    c=[train_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            axes[2].set_title(f"Train Reconstructed Images ({len(train_data)} images)", fontsize=12)
            axes[2].legend(fontsize=8)
            axes[2].grid(alpha=0.3)
            axes[2].set_xlabel('t-SNE Component 1')
            axes[2].set_ylabel('t-SNE Component 2')
            
            # Plot 4: Comparison Reconstructed Images (with proper context)
            for i, label in enumerate(unique_comparison_labels):
                mask = comparison_plot_labels == label
                axes[3].scatter(
                    comparison_recon_low_dim[mask, 0], comparison_recon_low_dim[mask, 1],
                    c=[comparison_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            axes[3].set_title(f"{comparison_type} Reconstructed Images ({len(comparison_data)} images)", fontsize=12)
            axes[3].legend(fontsize=8)
            axes[3].grid(alpha=0.3)
            axes[3].set_xlabel('t-SNE Component 1')
            axes[3].set_ylabel('t-SNE Component 2')
            
        else:
            # Original 1x2 layout (side-by-side latent spaces only)
            fig, axes = plt.subplots(1, 2, figsize=figure_size)
            
            # Plot train data
            unique_train_labels = np.unique(train_plot_labels)
            train_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_train_labels)))
            
            for i, label in enumerate(unique_train_labels):
                mask = train_plot_labels == label
                axes[0].scatter(
                    train_low_dim_embs[mask, 0], 
                    train_low_dim_embs[mask, 1],
                    c=[train_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7,
                    s=50,
                    edgecolors='none'
                )
            
            train_title = f"Train Data Latent Space ({len(train_data)} images)"
            if train_silhouette is not None:
                train_title += f"\nSilhouette Score: {train_silhouette:.3f}"
            axes[0].set_title(train_title, fontsize=14)
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')
            
            # Plot comparison data
            unique_comparison_labels = np.unique(comparison_plot_labels)
            comparison_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_comparison_labels)))
            
            for i, label in enumerate(unique_comparison_labels):
                mask = comparison_plot_labels == label
                axes[1].scatter(
                    comparison_low_dim_embs[mask, 0], 
                    comparison_low_dim_embs[mask, 1],
                    c=[comparison_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7,
                    s=50,
                    edgecolors='none'
                )
            
            comparison_title = f"{comparison_type} Latent Space ({len(comparison_data)} images)"
            if comparison_silhouette is not None:
                comparison_title += f"\nSilhouette Score: {comparison_silhouette:.3f}"
            if orig_silhouette is not None:
                comparison_title += f" (Original: {orig_silhouette:.3f})"
            axes[1].set_title(comparison_title, fontsize=14)
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
        
        # Add overall title
        if title_suffix:
            fig.suptitle(title_suffix, fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return train_silhouette, comparison_silhouette 


def plot_tsne_visualization(
    X: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE Visualization",
    random_state: int = 42,
    perplexity: int = 30,
    n_iter: int = 1000,
    save_path: Optional[str] = None,
    figure_size: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a t-SNE visualization of high-dimensional data.
    This implementation matches the AutoEncoderJupyterTest.ipynb reference patterns.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        labels: Class labels for each sample
        class_names: Optional list of class names for the legend
        title: Title for the plot
        random_state: Random state for reproducible results
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE optimization
        save_path: Path to save the figure (optional)
        figure_size: Size of the figure
    """
    from sklearn.manifold import TSNE
    
    # Apply t-SNE with fixed random state for consistency
    print(f"Computing t-SNE projection with perplexity={perplexity}, random_state={random_state}...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
        verbose=0
    )
    X_tsne = tsne.fit_transform(X)
    
    # Get unique labels and colors using rainbow colormap like the reference notebook
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Create the plot
    plt.figure(figsize=figure_size)
    
    # Plot each class with consistent colors
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        
        plt.scatter(
            X_tsne[mask, 0], 
            X_tsne[mask, 1],
            c=[colors[i]], 
            label=class_name,
            alpha=0.7,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add data point count to title like the reference notebook
    n_samples = len(X)
    full_title = f"{title} (n={n_samples})"
    
    plt.title(full_title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Make the plot look cleaner
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE visualization to: {save_path}")
    
    plt.show() 


def visualize_three_way_latent_spaces(
    model,
    train_data: Union[torch.Tensor, np.ndarray],
    train_labels: Union[torch.Tensor, np.ndarray],
    validation_data: Union[torch.Tensor, np.ndarray],
    validation_labels: Union[torch.Tensor, np.ndarray],
    test_data: Union[torch.Tensor, np.ndarray],
    test_labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    title_suffix: str = "Final Evaluation",
    max_samples: int = 500,
    device: str = 'cpu',
    figure_size: Tuple[int, int] = (20, 12),
    verbose: bool = False
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Visualize train, validation, and test data with reconstructions and latent spaces.
    
    Creates a 2x3 grid:
    - Top row: Train, Validation, Test latent spaces (t-SNE)
    - Bottom row: Train, Validation, Test reconstructed images
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels
        validation_data: Validation data tensor
        validation_labels: Validation labels
        test_data: Test data tensor
        test_labels: Test labels
        class_names: Optional list of class names
        title_suffix: Title for the overall visualization
        max_samples: Maximum samples per dataset to visualize
        device: Device to run computations on
        figure_size: Figure size for the plot
        verbose: Whether to print detailed information
        
    Returns:
        Tuple of (train_silhouette, validation_silhouette, test_silhouette)
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    
    # Convert to tensors if needed
    train_data = torch.tensor(train_data).float() if not isinstance(train_data, torch.Tensor) else train_data.float()
    validation_data = torch.tensor(validation_data).float() if not isinstance(validation_data, torch.Tensor) else validation_data.float()
    test_data = torch.tensor(test_data).float() if not isinstance(test_data, torch.Tensor) else test_data.float()
    
    train_labels = torch.tensor(train_labels).long() if not isinstance(train_labels, torch.Tensor) else train_labels.long()
    validation_labels = torch.tensor(validation_labels).long() if not isinstance(validation_labels, torch.Tensor) else validation_labels.long()
    test_labels = torch.tensor(test_labels).long() if not isinstance(test_labels, torch.Tensor) else test_labels.long()
    
    # Move to device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    train_data_device = train_data.to(device_obj)
    validation_data_device = validation_data.to(device_obj)
    test_data_device = test_data.to(device_obj)
    
    # Limit samples for visualization performance
    train_plot_only = min(max_samples, len(train_data))
    validation_plot_only = min(max_samples, len(validation_data))
    test_plot_only = min(max_samples, len(test_data))
    
    model.eval()
    with torch.no_grad():
        # Get latent representations and reconstructions for all three datasets
        try:
            train_encoded, train_reconstructed = model(train_data_device[:train_plot_only])
        except:
            train_encoded = model.encode(train_data_device[:train_plot_only])
            train_reconstructed = model.decode(train_encoded)
        train_encoded = train_encoded.detach().cpu().numpy()
        train_reconstructed = train_reconstructed.view(train_reconstructed.size(0), -1).detach().cpu().numpy()
        
        try:
            validation_encoded, validation_reconstructed = model(validation_data_device[:validation_plot_only])
        except:
            validation_encoded = model.encode(validation_data_device[:validation_plot_only])
            validation_reconstructed = model.decode(validation_encoded)
        validation_encoded = validation_encoded.detach().cpu().numpy()
        validation_reconstructed = validation_reconstructed.view(validation_reconstructed.size(0), -1).detach().cpu().numpy()
        
        try:
            test_encoded, test_reconstructed = model(test_data_device[:test_plot_only])
        except:
            test_encoded = model.encode(test_data_device[:test_plot_only])
            test_reconstructed = model.decode(test_encoded)
        test_encoded = test_encoded.detach().cpu().numpy()
        test_reconstructed = test_reconstructed.view(test_reconstructed.size(0), -1).detach().cpu().numpy()
    
    # Apply t-SNE to latent representations with different seeds
    train_perplexity = min(30, max(5, len(train_encoded)//4))
    train_tsne = TSNE(perplexity=train_perplexity, n_components=2, init='pca', max_iter=5000, random_state=42)
    train_low_dim = train_tsne.fit_transform(train_encoded)
    train_plot_labels = train_labels[:train_plot_only].cpu().numpy()
    
    validation_perplexity = min(30, max(5, len(validation_encoded)//4))
    validation_tsne = TSNE(perplexity=validation_perplexity, n_components=2, init='pca', max_iter=5000, random_state=123)
    validation_low_dim = validation_tsne.fit_transform(validation_encoded)
    validation_plot_labels = validation_labels[:validation_plot_only].cpu().numpy()
    
    test_perplexity = min(30, max(5, len(test_encoded)//4))
    test_tsne = TSNE(perplexity=test_perplexity, n_components=2, init='pca', max_iter=5000, random_state=456)
    test_low_dim = test_tsne.fit_transform(test_encoded)
    test_plot_labels = test_labels[:test_plot_only].cpu().numpy()
    
    # Apply t-SNE to reconstructed images with different seeds
    train_recon_perplexity = min(30, max(5, len(train_reconstructed)//4))
    train_recon_tsne = TSNE(perplexity=train_recon_perplexity, n_components=2, init='pca', max_iter=5000, random_state=789)
    train_recon_low_dim = train_recon_tsne.fit_transform(train_reconstructed)
    
    validation_recon_perplexity = min(30, max(5, len(validation_reconstructed)//4))
    validation_recon_tsne = TSNE(perplexity=validation_recon_perplexity, n_components=2, init='pca', max_iter=5000, random_state=987)
    validation_recon_low_dim = validation_recon_tsne.fit_transform(validation_reconstructed)
    
    test_recon_perplexity = min(30, max(5, len(test_reconstructed)//4))
    test_recon_tsne = TSNE(perplexity=test_recon_perplexity, n_components=2, init='pca', max_iter=5000, random_state=654)
    test_recon_low_dim = test_recon_tsne.fit_transform(test_reconstructed)
    
    # Calculate silhouette scores for both latent spaces and reconstructed images
    def safe_silhouette(embeddings, labels):
        try:
            if len(np.unique(labels)) > 1:
                return silhouette_score(embeddings, labels)
        except:
            pass
        return None
    
    # Silhouette scores for latent spaces
    train_silhouette = safe_silhouette(train_low_dim, train_plot_labels)
    validation_silhouette = safe_silhouette(validation_low_dim, validation_plot_labels)
    test_silhouette = safe_silhouette(test_low_dim, test_plot_labels)
    
    # Silhouette scores for reconstructed images
    train_recon_silhouette = safe_silhouette(train_recon_low_dim, train_plot_labels)
    validation_recon_silhouette = safe_silhouette(validation_recon_low_dim, validation_plot_labels)
    test_recon_silhouette = safe_silhouette(test_recon_low_dim, test_plot_labels)
    
    # Create 2x3 grid: Top row = latent spaces, Bottom row = reconstructed images  
    fig, axes = plt.subplots(2, 3, figsize=figure_size)
    
    datasets = [
        ("Training", train_low_dim, train_recon_low_dim, train_plot_labels, train_silhouette, train_recon_silhouette),
        ("Validation", validation_low_dim, validation_recon_low_dim, validation_plot_labels, validation_silhouette, validation_recon_silhouette),
        ("Test", test_low_dim, test_recon_low_dim, test_plot_labels, test_silhouette, test_recon_silhouette)
    ]
    
    for i, (dataset_name, latent_embs, recon_embs, labels, latent_silhouette, recon_silhouette) in enumerate(datasets):
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Top row: Latent space t-SNE
        for j, label in enumerate(unique_labels):
            mask = labels == label
            axes[0, i].scatter(
                latent_embs[mask, 0], latent_embs[mask, 1],
                c=[colors[j]], alpha=0.7, s=50, edgecolors='none',
                label=class_names[label] if class_names is not None else f"Class {label}"
            )
        
        title = f"{dataset_name}: Latent Space ({len(latent_embs)} images)"
        if latent_silhouette is not None:
            title += f"\nSilhouette Score: {latent_silhouette:.3f}"
        axes[0, i].set_title(title, fontsize=12)
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(alpha=0.3)
        axes[0, i].set_xlabel('t-SNE Component 1')
        axes[0, i].set_ylabel('t-SNE Component 2')
        
        # Bottom row: Reconstructed images t-SNE
        for j, label in enumerate(unique_labels):
            mask = labels == label
            axes[1, i].scatter(
                recon_embs[mask, 0], recon_embs[mask, 1],
                c=[colors[j]], alpha=0.7, s=50, edgecolors='none',
                label=class_names[label] if class_names is not None else f"Class {label}"
            )
        
        title = f"{dataset_name}: Reconstructed Images ({len(recon_embs)} images)"
        if recon_silhouette is not None:
            title += f"\nSilhouette Score: {recon_silhouette:.3f}"
        axes[1, i].set_title(title, fontsize=12)
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(alpha=0.3)
        axes[1, i].set_xlabel('t-SNE Component 1')
        axes[1, i].set_ylabel('t-SNE Component 2')
    
    # Add overall title
    fig.suptitle(f"{title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return train_silhouette, validation_silhouette, test_silhouette 
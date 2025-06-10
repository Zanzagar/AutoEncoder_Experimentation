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
    test_data: Union[torch.Tensor, np.ndarray],
    test_labels: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    title_suffix: str = "",
    orig_silhouette: Optional[float] = None,
    max_samples: int = 500,
    device: str = 'cpu',
    figure_size: Tuple[int, int] = (20, 16),
    grid_layout: str = "2x2",
    verbose: bool = False
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute and visualize t-SNE projections of train and test data with both latent space and reconstructed images.
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels tensor
        test_data: Test data tensor
        test_labels: Test labels tensor
        class_names: List of class names for visualization
        title_suffix: Additional text for plot title
        orig_silhouette: Original silhouette score from training
        max_samples: Maximum samples to use for t-SNE (for performance)
        device: Device to run model on
        figure_size: Size of the figure
        grid_layout: Layout type ("1x2" for side-by-side latent only, "2x2" for latent + reconstructed)
        verbose: Whether to print progress messages (default: False for cleaner output)
        
    Returns:
        Tuple of (train_silhouette, test_silhouette) scores
    """
    import torch
    from sklearn.metrics import silhouette_score
    
    model.eval()
    
    # Convert to tensors if needed
    if isinstance(train_data, np.ndarray):
        train_data = torch.FloatTensor(train_data)
    if isinstance(test_data, np.ndarray):
        test_data = torch.FloatTensor(test_data)
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.detach().cpu().numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.detach().cpu().numpy()
    
    with torch.no_grad():
        # Process train data
        train_data_device = train_data.to(device)
        try:
            encoded_train, _ = model(train_data_device)
        except:
            # Try alternative interface
            encoded_train = model.encode(train_data_device)
        
        encoded_train = encoded_train.view(encoded_train.size(0), -1).detach().cpu().numpy()
        
        # Process test data
        test_data_device = test_data.to(device)
        try:
            encoded_test, _ = model(test_data_device)
        except:
            # Try alternative interface
            encoded_test = model.encode(test_data_device)
        
        encoded_test = encoded_test.view(encoded_test.size(0), -1).detach().cpu().numpy()
        
        # Restrict to max samples for performance
        train_plot_only = min(max_samples, encoded_train.shape[0])
        test_plot_only = min(max_samples, encoded_test.shape[0])
        
        if verbose:
            print(f"Running t-SNE on {train_plot_only} train samples and {test_plot_only} test samples...")
        
        # Apply t-SNE with consistent parameters
        # Train data t-SNE
        train_tsne = TSNE(perplexity=min(30, len(encoded_train[:train_plot_only])-1), 
                         n_components=2, init='pca', max_iter=5000, random_state=42)
        train_low_dim_embs = train_tsne.fit_transform(encoded_train[:train_plot_only])
        train_plot_labels = train_labels[:train_plot_only]
        
        # Test data t-SNE
        test_tsne = TSNE(perplexity=min(30, len(encoded_test[:test_plot_only])-1), 
                        n_components=2, init='pca', max_iter=5000, random_state=42)
        test_low_dim_embs = test_tsne.fit_transform(encoded_test[:test_plot_only])
        test_plot_labels = test_labels[:test_plot_only]
        
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
        
        test_silhouette = None
        if len(np.unique(test_plot_labels)) > 1:
            try:
                test_silhouette = silhouette_score(test_low_dim_embs, test_plot_labels)
                if verbose:
                    print(f"Test data silhouette score: {test_silhouette:.6f}")
                
                # Check if it matches original silhouette
                if orig_silhouette is not None and verbose:
                    similarity = 100 - abs(test_silhouette - orig_silhouette) * 100
                    print(f"Similarity to original score: {similarity:.2f}%")
            except Exception as e:
                if verbose:
                    print(f"Could not calculate test silhouette score: {e}")
        
        # Determine layout based on grid_layout parameter
        if grid_layout == "2x2":
            # Generate reconstructed images for both train and test data
            if verbose:
                print("Generating reconstructed images for t-SNE analysis...")
            
            with torch.no_grad():
                # Get reconstructed train data
                try:
                    _, train_reconstructed = model(train_data_device)
                except:
                    train_reconstructed = model.decode(encoded_train)
                train_reconstructed = train_reconstructed.view(train_reconstructed.size(0), -1).detach().cpu().numpy()
                
                # Get reconstructed test data  
                try:
                    _, test_reconstructed = model(test_data_device)
                except:
                    test_reconstructed = model.decode(encoded_test)
                test_reconstructed = test_reconstructed.view(test_reconstructed.size(0), -1).detach().cpu().numpy()
            
            # Compute t-SNE for reconstructed images
            if verbose:
                print("Computing t-SNE for reconstructed images...")
            train_recon_tsne = TSNE(perplexity=min(30, len(train_reconstructed[:train_plot_only])-1), 
                                   n_components=2, init='pca', max_iter=5000, random_state=42)
            train_recon_low_dim = train_recon_tsne.fit_transform(train_reconstructed[:train_plot_only])
            
            test_recon_tsne = TSNE(perplexity=min(30, len(test_reconstructed[:test_plot_only])-1), 
                                  n_components=2, init='pca', max_iter=5000, random_state=42)
            test_recon_low_dim = test_recon_tsne.fit_transform(test_reconstructed[:test_plot_only])
            
            # Create 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=figure_size)
            axes = axes.flatten()
            
            # Get colors once
            unique_train_labels = np.unique(train_plot_labels)
            unique_test_labels = np.unique(test_plot_labels)
            train_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_train_labels)))
            test_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_test_labels)))
            
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
            
            # Plot 2: Test Latent Space
            for i, label in enumerate(unique_test_labels):
                mask = test_plot_labels == label
                axes[1].scatter(
                    test_low_dim_embs[mask, 0], test_low_dim_embs[mask, 1],
                    c=[test_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            test_title = f"Test Latent Space ({len(test_data)} images)"
            if test_silhouette is not None:
                test_title += f"\nSilhouette Score: {test_silhouette:.3f}"
            if orig_silhouette is not None:
                test_title += f" (Original: {orig_silhouette:.3f})"
            axes[1].set_title(test_title, fontsize=12)
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
            
            # Plot 4: Test Reconstructed Images
            for i, label in enumerate(unique_test_labels):
                mask = test_plot_labels == label
                axes[3].scatter(
                    test_recon_low_dim[mask, 0], test_recon_low_dim[mask, 1],
                    c=[test_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7, s=50, edgecolors='none'
                )
            axes[3].set_title(f"Test Reconstructed Images ({len(test_data)} images)", fontsize=12)
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
            
            # Plot test data
            unique_test_labels = np.unique(test_plot_labels)
            test_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_test_labels)))
            
            for i, label in enumerate(unique_test_labels):
                mask = test_plot_labels == label
                axes[1].scatter(
                    test_low_dim_embs[mask, 0], 
                    test_low_dim_embs[mask, 1],
                    c=[test_colors[i]],
                    label=class_names[label] if class_names is not None else f"Class {label}",
                    alpha=0.7,
                    s=50,
                    edgecolors='none'
                )
            
            test_title = f"Test Data Latent Space ({len(test_data)} images)"
            if test_silhouette is not None:
                test_title += f"\nSilhouette Score: {test_silhouette:.3f}"
            if orig_silhouette is not None:
                test_title += f" (Original: {orig_silhouette:.3f})"
            axes[1].set_title(test_title, fontsize=14)
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
        
        # Add overall title
        if title_suffix:
            fig.suptitle(title_suffix, fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return train_silhouette, test_silhouette 


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
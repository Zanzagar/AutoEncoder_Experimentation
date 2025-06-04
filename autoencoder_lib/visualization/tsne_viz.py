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


def visualize_raw_data_tsne(
    dataset_info: Optional[Dict] = None,
    dataset_path: Optional[str] = None,
    random_state: int = 42,
    perplexity: Optional[int] = None,
    n_iter: int = 1000,
    max_samples: Optional[int] = None,
    figure_size: Tuple[int, int] = (12, 10)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create a t-SNE visualization of raw image data.
    
    Args:
        dataset_info: Dataset information dictionary (alternative to dataset_path)
        dataset_path: Path to the dataset info file
        random_state: Random seed for reproducibility
        perplexity: Perplexity parameter for t-SNE (auto-determined if None)
        n_iter: Number of iterations for t-SNE
        max_samples: Maximum number of samples to use (None for all)
        figure_size: Size of the figure to display
        
    Returns:
        Tuple of (t-SNE embedding, labels, silhouette score)
    """
    print("Loading dataset...")
    
    # Load dataset info
    if dataset_info is not None:
        pass  # Use provided dataset_info
    elif dataset_path is not None:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        dataset_info = np.load(dataset_path, allow_pickle=True).item()
    else:
        raise ValueError("Either dataset_info or dataset_path must be provided")
    
    # Get class names - handle both 'label_names' and 'class_names'
    class_names = dataset_info.get('label_names', 
                                  dataset_info.get('class_names',
                                  [f"Class {i}" for i in range(len(np.unique(dataset_info['labels'])))]))
    print(f"Dataset contains {len(class_names)} classes: {class_names}")
    
    # Determine samples to use
    total_samples = len(dataset_info.get('filenames', dataset_info.get('images', [])))
    if max_samples is not None and max_samples < total_samples:
        # Randomly sample
        np.random.seed(random_state)
        sample_indices = np.random.choice(total_samples, max_samples, replace=False)
        print(f"Using {max_samples} randomly selected samples out of {total_samples}")
    else:
        sample_indices = np.arange(total_samples)
        print(f"Using all {total_samples} samples")
    
    print(f"Preparing {len(sample_indices)} images for t-SNE...")
    
    # Load images and flatten them
    images = []
    labels = []
    
    # Check if we have direct image data or need to load from files
    if 'images' in dataset_info:
        # Direct image data
        for idx in sample_indices:
            img_array = dataset_info['images'][idx]
            if len(img_array.shape) > 2:
                img_array = img_array.squeeze()  # Remove extra dimensions
            images.append(img_array.flatten() / 255.0 if img_array.max() > 1 else img_array.flatten())
            labels.append(dataset_info['labels'][idx])
    else:
        # Load from file paths
        for idx in sample_indices:
            img_path = dataset_info['filenames'][idx]
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}, skipping...")
                continue
                
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img).flatten() / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                labels.append(dataset_info['labels'][idx])
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Running t-SNE on {X.shape[0]} images of dimension {X.shape[1]}...")
    print("This might take a while...")
    
    # Determine perplexity
    if perplexity is None:
        perplexity = min(30, len(X) - 1)
    
    # Apply t-SNE (use max_iter instead of deprecated n_iter)
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"t-SNE completed in {elapsed:.2f} seconds")
    
    # Calculate silhouette score on t-SNE results
    if len(np.unique(y)) > 1:  # Need at least 2 classes for silhouette score
        silhouette = float(silhouette_score(X_tsne, y))  # Convert to Python float
        print(f"Silhouette score on t-SNE embedding: {silhouette:.4f}")
    else:
        silhouette = 0.0
        print("Only one class found, silhouette score not applicable")
    
    # Visualize the t-SNE embedding
    plt.figure(figsize=figure_size)
    
    # Plot each class with a different color
    unique_labels = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            c=[colors[i]],
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.7,
            s=50,
            edgecolors='none'
        )
    
    # Add silhouette score to title
    title = f't-SNE Visualization of Raw Image Data (n={len(X)})'
    if silhouette > 0:
        title += f'\nSilhouette Score: {silhouette:.4f}'
    
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()
    
    return X_tsne, y, silhouette


def visualize_latent_tsne(
    latent_representations: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
    perplexity: Optional[int] = None,
    n_iter: int = 1000,
    figure_size: Tuple[int, int] = (12, 10),
    title: str = "t-SNE Visualization of Latent Space"
) -> Tuple[np.ndarray, float]:
    """
    Create a t-SNE visualization of latent space representations.
    
    Args:
        latent_representations: Latent space vectors (N, latent_dim)
        labels: Corresponding labels for each sample
        class_names: Names for each class
        random_state: Random seed for reproducibility
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        figure_size: Size of the figure
        title: Title for the plot
        
    Returns:
        Tuple of (t-SNE embedding, silhouette score)
    """
    print(f"Running t-SNE on {latent_representations.shape[0]} latent vectors of dimension {latent_representations.shape[1]}...")
    
    # Determine perplexity
    if perplexity is None:
        perplexity = min(30, len(latent_representations) - 1)
    
    # Apply t-SNE (use max_iter instead of deprecated n_iter)
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    latent_tsne = tsne.fit_transform(latent_representations)
    elapsed = time.time() - start_time
    print(f"t-SNE completed in {elapsed:.2f} seconds")
    
    # Calculate silhouette score
    if len(np.unique(labels)) > 1:
        silhouette = float(silhouette_score(latent_tsne, labels))  # Convert to Python float
        print(f"Silhouette score on latent t-SNE embedding: {silhouette:.4f}")
    else:
        silhouette = 0.0
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(np.unique(labels))]
    
    # Visualize
    plt.figure(figsize=figure_size)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            latent_tsne[mask, 0], latent_tsne[mask, 1],
            c=[colors[i]],
            label=class_names[label] if label < len(class_names) else f"Class {label}",
            alpha=0.7,
            s=50,
            edgecolors='none'
        )
    
    # Add silhouette score to title
    full_title = title
    if silhouette > 0:
        full_title += f'\nSilhouette Score: {silhouette:.4f}'
    
    plt.title(full_title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()
    
    return latent_tsne, silhouette


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
        embeddings_list: List of t-SNE embeddings to compare
        labels_list: List of corresponding labels
        titles: Titles for each embedding
        class_names: Names for each class
        figure_size: Size of the figure
    """
    n_embeddings = len(embeddings_list)
    
    plt.figure(figsize=figure_size)
    
    for i, (embedding, labels, title) in enumerate(zip(embeddings_list, labels_list, titles)):
        plt.subplot(1, n_embeddings, i + 1)
        
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
        
        # Calculate and add silhouette score
        if len(unique_labels) > 1:
            silhouette = silhouette_score(embedding, labels)
            title_with_score = f"{title}\nSilhouette: {silhouette:.3f}"
        else:
            title_with_score = title
        
        plt.title(title_with_score, fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Only show legend on first subplot
        if i == 0:
            plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
    figure_size: Tuple[int, int] = (20, 8)
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute and visualize t-SNE projections of train and test data side by side.
    
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
                print(f"Train data silhouette score: {train_silhouette:.6f}")
            except Exception as e:
                print(f"Could not calculate train silhouette score: {e}")
        
        test_silhouette = None
        if len(np.unique(test_plot_labels)) > 1:
            try:
                test_silhouette = silhouette_score(test_low_dim_embs, test_plot_labels)
                print(f"Test data silhouette score: {test_silhouette:.6f}")
                
                # Check if it matches original silhouette
                if orig_silhouette is not None:
                    similarity = 100 - abs(test_silhouette - orig_silhouette) * 100
                    print(f"Similarity to original score: {similarity:.2f}%")
            except Exception as e:
                print(f"Could not calculate test silhouette score: {e}")
        
        # Create side-by-side plots
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
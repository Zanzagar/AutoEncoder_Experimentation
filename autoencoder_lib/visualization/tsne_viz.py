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
        silhouette = silhouette_score(X_tsne, y)
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
        silhouette = silhouette_score(latent_tsne, labels)
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
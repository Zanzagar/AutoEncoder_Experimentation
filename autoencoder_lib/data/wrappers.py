"""
Dataset Generation and Visualization Wrappers

This module provides high-level wrapper functions for dataset generation and visualization
that can be easily used from Jupyter notebooks. These wrappers integrate the underlying
data generation and visualization capabilities into simple, user-friendly functions.

Supports both:
1. Original layered geological dataset format (from AutoEncoderJupyterTest.ipynb)
2. New geological dataset format (from the refactored framework)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
import logging
from .geological import LayeredGeologicalDataset
from .layered_geological import (
    generate_layered_dataset, 
    visualize_dataset_samples, 
    load_layered_dataset,
    get_dataset_statistics as get_layered_stats
)
from .preprocessing import calculate_data_statistics

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def generate_dataset(
    dataset_type: str = "layered_geological",
    output_dir: str = "generated_dataset",
    num_samples_per_class: int = 500,
    image_size: int = 64,
    random_seed: Optional[int] = None,
    visualize: bool = True,
    force_regenerate: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a dataset with automatic visualization and statistics.
    
    This wrapper supports multiple dataset types and provides a unified interface
    for dataset generation across the autoencoder experimentation framework.
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset to generate. Options:
        - "layered_geological": Original format from AutoEncoderJupyterTest.ipynb
        - "geological": New framework format with 5 geological pattern types
    output_dir : str
        Directory to save the generated dataset
    num_samples_per_class : int
        Number of samples to generate per class
    image_size : int
        Size of the square images (width and height)
    random_seed : int, optional
        Random seed for reproducible generation
    visualize : bool
        Whether to automatically display sample visualizations
    force_regenerate : bool
        If True, regenerate even if dataset already exists
    **kwargs
        Additional parameters specific to the dataset type
        
    Returns:
    --------
    dict
        Dictionary containing dataset information, format depends on dataset_type:
        
        For "layered_geological":
        - 'filenames': List of image file paths
        - 'labels': List of corresponding class labels (0 or 1)
        - 'label_names': List of class names ['consistent_layers', 'variable_layers']
        - 'params': Dictionary of generation parameters
        
        For "geological":
        - 'images': NumPy array of images
        - 'labels': NumPy array of labels
        - 'class_names': List of class names
        - 'metadata': Dictionary with generation parameters and statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
    
    logger.info(f"Generating {dataset_type} dataset...")
    logger.info(f"Parameters: {num_samples_per_class} samples per class, {image_size}x{image_size} images")
    
    if dataset_type == "layered_geological":
        # Use the original format from AutoEncoderJupyterTest.ipynb
        dataset_info = generate_layered_dataset(
            output_dir=output_dir,
            num_samples_per_class=num_samples_per_class,
            image_size=image_size,
            force_regenerate=force_regenerate
        )
        
        if dataset_info is None:
            raise RuntimeError("Failed to generate layered geological dataset")
        
        if visualize:
            logger.info("Displaying sample visualizations...")
            visualize_dataset_samples(dataset_info, samples_per_class=5)
            
            # Also show statistics
            stats = get_layered_stats(dataset_info)
            print("\nDataset Statistics:")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Number of classes: {stats['num_classes']}")
            print(f"Class distribution: {stats['class_distribution']}")
            print(f"Image size: {stats['image_size']}x{stats['image_size']}")
        
        return dataset_info
        
    elif dataset_type == "geological":
        # Use the new framework format
        # Extract geological-specific parameters
        num_layers_range = kwargs.get('num_layers_range', (3, 7))
        noise_level = kwargs.get('noise_level', 0.1)
        
        # Create the dataset
        dataset = LayeredGeologicalDataset(
            output_dir=output_dir,
            num_samples_per_class=num_samples_per_class,
            image_size=image_size,
            num_layers_range=num_layers_range,
            noise_level=noise_level,
            random_seed=random_seed
        )
        
        # Generate the dataset
        dataset_info = dataset.generate()
        
        if visualize:
            logger.info("Displaying sample visualizations...")
            _visualize_geological_samples(dataset_info)
            
            # Show statistics
            stats = calculate_data_statistics(dataset_info['images'])
            print("\nDataset Statistics:")
            print(f"Total samples: {len(dataset_info['images'])}")
            print(f"Number of classes: {len(dataset_info['class_names'])}")
            print(f"Image shape: {dataset_info['images'].shape}")
            print(f"Pixel value range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"Mean pixel value: {stats['mean']:.3f}")
            print(f"Std pixel value: {stats['std']:.3f}")
        
        return dataset_info
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: 'layered_geological', 'geological'")


def visualize_dataset(
    dataset_info: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None,
    dataset_type: str = "auto",
    tsne_perplexity: int = 30,
    tsne_random_state: int = 42,
    max_samples_for_tsne: int = 1000,
    show_statistics: bool = True,
    figure_size: Tuple[int, int] = (15, 10)
) -> Dict[str, Any]:
    """
    Visualize and analyze a dataset with comprehensive plots and statistics.
    
    This wrapper provides unified visualization for different dataset formats
    and automatically detects the format when possible.
    
    Parameters:
    -----------
    dataset_info : dict, optional
        Dataset information dictionary (if already loaded)
    dataset_path : str, optional
        Path to dataset file or directory (if loading from disk)
    dataset_type : str
        Type of dataset format. Options:
        - "auto": Automatically detect format
        - "layered_geological": Original format from AutoEncoderJupyterTest.ipynb
        - "geological": New framework format
    tsne_perplexity : int
        Perplexity parameter for t-SNE
    tsne_random_state : int
        Random state for reproducible t-SNE
    max_samples_for_tsne : int
        Maximum number of samples to use for t-SNE (for performance)
    show_statistics : bool
        Whether to display statistical summaries
    figure_size : tuple
        Size of the visualization figures
        
    Returns:
    --------
    dict
        Dictionary containing analysis results:
        - 'tsne_projection': t-SNE coordinates
        - 'pca_projection': PCA coordinates  
        - 'class_distribution': Class counts
        - 'statistics': Dataset statistics
    """
    # Load dataset if path provided
    if dataset_path is not None and dataset_info is None:
        if dataset_type == "auto":
            # Try to detect format
            if dataset_path.endswith('.npy') or Path(dataset_path).is_dir():
                dataset_type = "layered_geological"
            elif dataset_path.endswith('.json'):
                dataset_type = "geological"
            else:
                logger.warning("Could not auto-detect dataset type, assuming layered_geological")
                dataset_type = "layered_geological"
        
        if dataset_type == "layered_geological":
            dataset_info = load_layered_dataset(dataset_path)
        elif dataset_type == "geological":
            with open(dataset_path, 'r') as f:
                dataset_info = json.load(f)
                # Convert lists back to numpy arrays
                dataset_info['images'] = np.array(dataset_info['images'])
                dataset_info['labels'] = np.array(dataset_info['labels'])
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    if dataset_info is None:
        raise ValueError("Must provide either dataset_info or dataset_path")
    
    # Detect format if auto
    if dataset_type == "auto":
        if 'filenames' in dataset_info and 'label_names' in dataset_info:
            dataset_type = "layered_geological"
        elif 'images' in dataset_info and 'class_names' in dataset_info:
            dataset_type = "geological"
        else:
            raise ValueError("Could not auto-detect dataset format")
    
    logger.info(f"Analyzing {dataset_type} dataset...")
    
    if dataset_type == "layered_geological":
        return _analyze_layered_geological(
            dataset_info, tsne_perplexity, tsne_random_state, 
            max_samples_for_tsne, show_statistics, figure_size
        )
    elif dataset_type == "geological":
        return _analyze_geological(
            dataset_info, tsne_perplexity, tsne_random_state,
            max_samples_for_tsne, show_statistics, figure_size
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _analyze_layered_geological(
    dataset_info: Dict[str, Any],
    tsne_perplexity: int,
    tsne_random_state: int,
    max_samples_for_tsne: int,
    show_statistics: bool,
    figure_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Analyze layered geological dataset (original format)"""
    
    # Load images into memory for analysis
    images = []
    labels = []
    
    logger.info("Loading images for analysis...")
    sample_indices = list(range(len(dataset_info['filenames'])))
    if len(sample_indices) > max_samples_for_tsne:
        sample_indices = np.random.choice(sample_indices, max_samples_for_tsne, replace=False)
    
    for idx in sample_indices:
        img_path = dataset_info['filenames'][idx]
        img = plt.imread(img_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)  # Convert to grayscale if needed
        images.append(img.flatten())
        labels.append(dataset_info['labels'][idx])
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Perform dimensionality reduction
    logger.info("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_random_state)
    tsne_result = tsne.fit_transform(images)
    
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=tsne_random_state)
    pca_result = pca.fit_transform(images)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Sample visualization
    visualize_dataset_samples(dataset_info, samples_per_class=5, figure_size=(8, 4))
    
    # t-SNE plot
    scatter = axes[0, 0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('t-SNE Projection')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # PCA plot
    scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('PCA Projection')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Class distribution
    class_counts = {}
    for label in dataset_info['labels']:
        class_name = dataset_info['label_names'][label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    axes[1, 0].bar(class_counts.keys(), class_counts.values())
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Pixel intensity distribution
    pixel_values = images.flatten()
    axes[1, 1].hist(pixel_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Pixel Intensity Distribution')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats = get_layered_stats(dataset_info)
    
    if show_statistics:
        print("\n" + "="*50)
        print("DATASET ANALYSIS SUMMARY")
        print("="*50)
        print(f"Dataset Type: Layered Geological (Original Format)")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Number of Classes: {stats['num_classes']}")
        print(f"Image Size: {stats['image_size']}x{stats['image_size']}")
        print(f"Class Distribution: {stats['class_distribution']}")
        print(f"Pixel Value Range: [{np.min(pixel_values):.3f}, {np.max(pixel_values):.3f}]")
        print(f"Mean Pixel Value: {np.mean(pixel_values):.3f}")
        print(f"Std Pixel Value: {np.std(pixel_values):.3f}")
        print(f"PCA Explained Variance: {pca.explained_variance_ratio_[:2]}")
    
    return {
        'tsne_projection': tsne_result,
        'pca_projection': pca_result,
        'class_distribution': class_counts,
        'statistics': stats,
        'pca_explained_variance': pca.explained_variance_ratio_[:2],
        'pixel_statistics': {
            'min': np.min(pixel_values),
            'max': np.max(pixel_values),
            'mean': np.mean(pixel_values),
            'std': np.std(pixel_values)
        }
    }


def _analyze_geological(
    dataset_info: Dict[str, Any],
    tsne_perplexity: int,
    tsne_random_state: int,
    max_samples_for_tsne: int,
    show_statistics: bool,
    figure_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Analyze geological dataset (new framework format)"""
    
    images = dataset_info['images']
    labels = dataset_info['labels']
    class_names = dataset_info['class_names']
    
    # Sample data if too large
    if len(images) > max_samples_for_tsne:
        indices = np.random.choice(len(images), max_samples_for_tsne, replace=False)
        images_sample = images[indices]
        labels_sample = labels[indices]
    else:
        images_sample = images
        labels_sample = labels
    
    # Flatten images for dimensionality reduction
    images_flat = images_sample.reshape(len(images_sample), -1)
    
    # Perform dimensionality reduction
    logger.info("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_random_state)
    tsne_result = tsne.fit_transform(images_flat)
    
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=tsne_random_state)
    pca_result = pca.fit_transform(images_flat)
    
    # Create visualizations
    _visualize_geological_samples(dataset_info)
    
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # t-SNE plot
    scatter = axes[0, 0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_sample, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('t-SNE Projection')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # PCA plot
    scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], c=labels_sample, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title('PCA Projection')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_counts = {class_names[i]: counts[i] for i in unique_labels}
    
    axes[1, 0].bar(class_counts.keys(), class_counts.values())
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Pixel intensity distribution
    pixel_values = images.flatten()
    axes[1, 1].hist(pixel_values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Pixel Intensity Distribution')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats = calculate_data_statistics(images)
    
    if show_statistics:
        print("\n" + "="*50)
        print("DATASET ANALYSIS SUMMARY")
        print("="*50)
        print(f"Dataset Type: Geological (New Framework Format)")
        print(f"Total Samples: {len(images)}")
        print(f"Number of Classes: {len(class_names)}")
        print(f"Image Shape: {images.shape}")
        print(f"Class Distribution: {class_counts}")
        print(f"Pixel Value Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"Mean Pixel Value: {stats['mean']:.3f}")
        print(f"Std Pixel Value: {stats['std']:.3f}")
        print(f"PCA Explained Variance: {pca.explained_variance_ratio_[:2]}")
    
    return {
        'tsne_projection': tsne_result,
        'pca_projection': pca_result,
        'class_distribution': class_counts,
        'statistics': stats,
        'pca_explained_variance': pca.explained_variance_ratio_[:2]
    }


def _visualize_geological_samples(dataset_info: Dict[str, Any], samples_per_class: int = 5):
    """Visualize samples from geological dataset (new format)"""
    images = dataset_info['images']
    labels = dataset_info['labels']
    class_names = dataset_info['class_names']
    
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(15, 3*num_classes))
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, label in enumerate(unique_labels):
        # Get indices for this class
        class_indices = np.where(labels == label)[0]
        
        # Sample random images
        sample_indices = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j] if num_classes > 1 else axes[j]
            ax.imshow(images[idx], cmap='gray')
            
            # Add class label only to the first image of each row
            if j == 0:
                ax.set_ylabel(class_names[label], fontsize=12, rotation=0, labelpad=40, ha='right', va='center')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
    
    plt.suptitle("Geological Pattern Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() 
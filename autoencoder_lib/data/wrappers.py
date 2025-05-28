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
from PIL import Image
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
    
    Both dataset types now use the same file structure:
    - Individual PNG files for each sample
    - .npy metadata file with filenames, labels, and parameters
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset to generate. Options:
        - "layered_geological": Original format from AutoEncoderJupyterTest.ipynb (2 classes)
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
        Dictionary containing dataset information (unified format):
        - 'filenames': List of image file paths
        - 'labels': List of corresponding class labels
        - 'label_names': List of class names
        - 'params': Dictionary of generation parameters
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)
    
    logger.info(f"Generating {dataset_type} dataset...")
    logger.info(f"Parameters: {num_samples_per_class} samples per class, {image_size}x{image_size} images")
    
    if dataset_type == "layered_geological":
        # Use the original format from AutoEncoderJupyterTest.ipynb (2 classes)
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
        # Use the new framework format (5 classes) - now uses same file structure
        # Extract geological-specific parameters
        num_layers_range = kwargs.get('num_layers_range', (3, 7))
        noise_level = kwargs.get('noise_level', 0.1)
        
        # Create the dataset using the updated LayeredGeologicalDataset
        dataset = LayeredGeologicalDataset()
        
        # Generate the dataset
        dataset_info = dataset.generate(
            output_dir=output_dir,
            num_samples_per_class=num_samples_per_class,
            image_size=image_size,
            num_layers_range=num_layers_range,
            noise_level=noise_level,
            random_seed=random_seed,
            force_regenerate=force_regenerate
        )
        
        if visualize:
            logger.info("Displaying sample visualizations...")
            # Use the same visualization function since format is now unified
            visualize_dataset_samples(dataset_info, samples_per_class=5)
            
            # Show statistics using the unified format
            stats = get_layered_stats(dataset_info)
            print("\nDataset Statistics:")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Number of classes: {stats['num_classes']}")
            print(f"Class distribution: {stats['class_distribution']}")
            print(f"Image size: {stats['image_size']}x{stats['image_size']}")
        
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
    
    This wrapper provides unified visualization for different dataset formats.
    Both dataset types now use the same file structure (PNG files + .npy metadata).
    
    Parameters:
    -----------
    dataset_info : dict, optional
        Dataset information dictionary (if already loaded)
    dataset_path : str, optional
        Path to dataset file or directory (if loading from disk)
    dataset_type : str
        Type of dataset format. Options:
        - "auto": Automatically detect format (based on number of classes)
        - "layered_geological": Original format (2 classes)
        - "geological": New framework format (5 classes)
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
            # Try to detect format - both use .npy files now
            if dataset_path.endswith('.npy') or Path(dataset_path).is_dir():
                # Load to check number of classes for auto-detection
                temp_info = load_layered_dataset(dataset_path)
                if len(temp_info['label_names']) == 2:
                    dataset_type = "layered_geological"
                elif len(temp_info['label_names']) == 5:
                    dataset_type = "geological"
                else:
                    logger.warning("Could not auto-detect dataset type, assuming layered_geological")
                    dataset_type = "layered_geological"
                dataset_info = temp_info
            else:
                logger.warning("Could not auto-detect dataset type, assuming layered_geological")
                dataset_type = "layered_geological"
        
        if dataset_info is None:
            # Load using the unified loader
            dataset_info = load_layered_dataset(dataset_path)
    
    if dataset_info is None:
        raise ValueError("Must provide either dataset_info or dataset_path")
    
    # Detect format if auto (based on loaded data)
    if dataset_type == "auto":
        if 'label_names' in dataset_info:
            if len(dataset_info['label_names']) == 2:
                dataset_type = "layered_geological"
            elif len(dataset_info['label_names']) == 5:
                dataset_type = "geological"
            else:
                dataset_type = "layered_geological"  # Default fallback
        else:
            raise ValueError("Could not auto-detect dataset format")
    
    logger.info(f"Analyzing {dataset_type} dataset...")
    
    # Both dataset types now use the same format, so use unified analysis
    return _analyze_unified_format(
        dataset_info, dataset_type, tsne_perplexity, tsne_random_state, 
        max_samples_for_tsne, show_statistics, figure_size
    )


def _analyze_unified_format(
    dataset_info: Dict[str, Any],
    dataset_type: str,
    tsne_perplexity: int,
    tsne_random_state: int,
    max_samples_for_tsne: int,
    show_statistics: bool,
    figure_size: Tuple[int, int]
) -> Dict[str, Any]:
    """Analyze dataset in unified format (PNG files + .npy metadata)"""
    
    # Load images into memory for analysis
    images = []
    labels = []
    
    logger.info("Loading images for analysis...")
    sample_indices = list(range(len(dataset_info['filenames'])))
    if len(sample_indices) > max_samples_for_tsne:
        sample_indices = np.random.choice(sample_indices, max_samples_for_tsne, replace=False)
    
    # Determine base directory for image files
    # The filenames in the unified format are relative to the dataset directory
    first_filename = dataset_info['filenames'][0]
    
    # Try to find the base directory by looking for existing files
    base_dir = None
    possible_dirs = [Path('.'), Path('demo_original_dataset'), Path('demo_enhanced_dataset')]
    
    # Also check if we can infer from the filename structure
    if '/' in first_filename:
        possible_dirs.append(Path(first_filename).parent.parent)
    
    for test_dir in possible_dirs:
        test_path = test_dir / first_filename
        if test_path.exists():
            base_dir = test_dir
            break
    
    if base_dir is None:
        # Try to infer from the current working directory
        logger.warning("Could not determine base directory, using current directory")
        base_dir = Path('.')
    
    logger.info(f"Using base directory: {base_dir}")
    
    for idx in sample_indices:
        img_path = base_dir / dataset_info['filenames'][idx]
        
        if img_path.exists():
            try:
                # Use PIL to load the image more reliably
                img = Image.open(img_path)
                img_array = np.array(img)
                
                # Convert to grayscale if needed
                if len(img_array.shape) == 3:
                    img_array = np.mean(img_array, axis=2)
                elif len(img_array.shape) == 4:  # RGBA
                    img_array = np.mean(img_array[:, :, :3], axis=2)
                
                images.append(img_array.flatten())
                labels.append(dataset_info['labels'][idx])
            except Exception as e:
                logger.warning(f"Could not load image {img_path}: {e}")
        else:
            logger.warning(f"Could not find image file: {img_path}")
    
    if not images:
        raise FileNotFoundError("Could not load any images from the dataset")
    
    images = np.array(images)
    labels = np.array(labels)
    class_names = dataset_info['label_names']
    
    logger.info(f"Loaded {len(images)} images for analysis")
    
    # Perform dimensionality reduction
    logger.info("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=min(tsne_perplexity, len(images)-1), random_state=tsne_random_state)
    tsne_result = tsne.fit_transform(images)
    
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=tsne_random_state)
    pca_result = pca.fit_transform(images)
    
    # Create visualizations - but fix the path issue for visualize_dataset_samples
    # We need to create a modified dataset_info with absolute paths for visualization
    viz_dataset_info = dataset_info.copy()
    viz_dataset_info['filenames'] = [str(base_dir / fname) for fname in dataset_info['filenames']]
    
    visualize_dataset_samples(viz_dataset_info, samples_per_class=5)
    
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
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
        print(f"Dataset Type: {dataset_type}")
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
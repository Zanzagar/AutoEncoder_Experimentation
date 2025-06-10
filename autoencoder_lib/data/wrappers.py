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
from sklearn.metrics import silhouette_score
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
    # New parameters for train-test split
    create_train_test_split: bool = True,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1,
    split_seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a dataset with automatic visualization and statistics.
    
    This wrapper supports multiple dataset types and provides a unified interface
    for dataset generation across the autoencoder experimentation framework.
    
    Both dataset types now use the same file structure:
    - Individual PNG files for each sample
    - .npy metadata file with filenames, labels, and parameters
    - Optional train/test/validation split indices for reproducible experiments
    
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
    create_train_test_split : bool
        Whether to create and save train/test/validation split indices
    train_ratio : float
        Proportion of data for training (default: 0.7)
    test_ratio : float
        Proportion of data for testing (default: 0.2)
    validation_ratio : float
        Proportion of data for validation (default: 0.1)
    split_seed : int, optional
        Random seed for reproducible splits (uses random_seed if not specified)
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
        - 'split_info': Dictionary with train/test/validation indices (if create_train_test_split=True)
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
        
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: 'layered_geological', 'geological'")
    
    # Create train/test/validation split if requested
    if create_train_test_split:
        logger.info("Creating train/test/validation split...")
        
        # Use split_seed if provided, otherwise use random_seed
        actual_split_seed = split_seed if split_seed is not None else random_seed
        
        if actual_split_seed is not None:
            np.random.seed(actual_split_seed)
        
        total_samples = len(dataset_info['filenames'])
        
        # Validate ratios
        if abs(train_ratio + test_ratio + validation_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + test_ratio + validation_ratio}")
        
        # Create stratified split to maintain class balance
        indices_by_class = {}
        for idx, label in enumerate(dataset_info['labels']):
            if label not in indices_by_class:
                indices_by_class[label] = []
            indices_by_class[label].append(idx)
        
        train_indices = []
        test_indices = []
        val_indices = []
        
        for label, class_indices in indices_by_class.items():
            np.random.shuffle(class_indices)
            n_samples = len(class_indices)
            
            n_train = int(n_samples * train_ratio)
            n_test = int(n_samples * test_ratio)
            n_val = n_samples - n_train - n_test  # Remaining goes to validation
            
            train_indices.extend(class_indices[:n_train])
            test_indices.extend(class_indices[n_train:n_train + n_test])
            if validation_ratio > 0:
                val_indices.extend(class_indices[n_train + n_test:])
        
        # Shuffle the combined indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        if validation_ratio > 0:
            np.random.shuffle(val_indices)
        
        # Create split info
        split_info = {
            'train_indices': sorted(train_indices),
            'test_indices': sorted(test_indices),
            'metadata': {
                'split_seed': actual_split_seed,
                'train_ratio': train_ratio,
                'test_ratio': test_ratio,
                'validation_ratio': validation_ratio,
                'total_samples': total_samples,
                'train_samples': len(train_indices),
                'test_samples': len(test_indices),
                'stratified': True  # We use stratified splitting
            }
        }
        
        if validation_ratio > 0:
            split_info['validation_indices'] = sorted(val_indices)
            split_info['metadata']['validation_samples'] = len(val_indices)
        
        # Add split info to dataset
        dataset_info['split_info'] = split_info
        
        # Save updated dataset info with split
        output_path = Path(output_dir)
        np.save(output_path / 'dataset_info.npy', dataset_info)
        
        logger.info(f"Split created: {len(train_indices)} train, {len(test_indices)} test" +
                   (f", {len(val_indices)} validation" if validation_ratio > 0 else ""))
        
        # Print split statistics
        if visualize:
            print(f"\nTrain/Test Split Created:")
            print(f"Split seed: {actual_split_seed}")
            print(f"Train samples: {len(train_indices)} ({train_ratio:.1%})")
            print(f"Test samples: {len(test_indices)} ({test_ratio:.1%})")
            if validation_ratio > 0:
                print(f"Validation samples: {len(val_indices)} ({validation_ratio:.1%})")
    
    return dataset_info


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
    
    # Calculate silhouette scores
    logger.info("Computing Silhouette Scores...")
    tsne_silhouette = silhouette_score(tsne_result, labels) if len(np.unique(labels)) > 1 else 0
    pca_silhouette = silhouette_score(pca_result, labels) if len(np.unique(labels)) > 1 else 0
    
    # Create visualizations - but fix the path issue for visualize_dataset_samples
    # We need to create a modified dataset_info with absolute paths for visualization
    viz_dataset_info = dataset_info.copy()
    viz_dataset_info['filenames'] = [str(base_dir / fname) for fname in dataset_info['filenames']]
    
    visualize_dataset_samples(viz_dataset_info, samples_per_class=5)
    
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Define colors for each class
    colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
    
    # t-SNE plot with class labels
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            axes[0, 0].scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                             c=[colors[i]], label=class_name, alpha=0.7, s=20)
    
    axes[0, 0].set_title(f't-SNE Projection (n={len(images)}, Silhouette={tsne_silhouette:.3f})')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # PCA plot with class labels
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            axes[0, 1].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                             c=[colors[i]], label=class_name, alpha=0.7, s=20)
    
    axes[0, 1].set_title(f'PCA Projection (n={len(images)}, Silhouette={pca_silhouette:.3f})')
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Class distribution
    class_counts = {}
    for label in dataset_info['labels']:
        class_name = dataset_info['label_names'][label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    axes[1, 0].bar(class_counts.keys(), class_counts.values(), color=colors[:len(class_counts)])
    axes[1, 0].set_title('Class Distribution')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Per-class pixel intensity distribution
    axes[1, 1].set_title('Pixel Intensity Distribution by Class')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Density')
    
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            class_pixels = images[mask].flatten()
            axes[1, 1].hist(class_pixels, bins=30, alpha=0.6, 
                          label=class_name, color=colors[i], density=True)
    
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    stats = get_layered_stats(dataset_info)
    
    if show_statistics:
        pixel_values = images.flatten()
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
        print(f"t-SNE Silhouette Score: {tsne_silhouette:.3f}")
        print(f"PCA Silhouette Score: {pca_silhouette:.3f}")
    
    return {
        'tsne_projection': tsne_result,
        'pca_projection': pca_result,
        'class_distribution': class_counts,
        'statistics': stats,
        'pca_explained_variance': pca.explained_variance_ratio_[:2],
        'silhouette_scores': {
            'tsne': tsne_silhouette,
            'pca': pca_silhouette
        },
        'pixel_statistics': {
            'min': np.min(images.flatten()),
            'max': np.max(images.flatten()),
            'mean': np.mean(images.flatten()),
            'std': np.std(images.flatten())
        }
    }


def get_split_data(dataset_info: Dict[str, Any], split_type: str = "train") -> Dict[str, Any]:
    """
    Extract train, test, or validation data using the saved split indices.
    
    Parameters:
    -----------
    dataset_info : dict
        Dataset information dictionary containing split_info
    split_type : str
        Type of split to extract: "train", "test", or "validation"
        
    Returns:
    --------
    dict
        Dictionary containing the split data:
        - 'filenames': List of image file paths for this split
        - 'labels': List of corresponding class labels for this split
        - 'indices': List of original dataset indices for this split
        - 'label_names': List of class names (same for all splits)
        - 'metadata': Split metadata information
    """
    if 'split_info' not in dataset_info:
        raise ValueError("Dataset does not contain split information. "
                        "Generate dataset with create_train_test_split=True")
    
    split_info = dataset_info['split_info']
    valid_types = ['train', 'test', 'validation']
    
    if split_type not in valid_types:
        raise ValueError(f"split_type must be one of {valid_types}, got '{split_type}'")
    
    # Get indices for the requested split
    indices_key = f"{split_type}_indices"
    if indices_key not in split_info:
        if split_type == "validation" and split_info['metadata'].get('validation_ratio', 0) == 0:
            raise ValueError("No validation split available. Dataset was created with validation_ratio=0")
        else:
            raise ValueError(f"Split type '{split_type}' not found in dataset")
    
    indices = split_info[indices_key]
    
    # Extract data for this split
    split_filenames = [dataset_info['filenames'][i] for i in indices]
    split_labels = [dataset_info['labels'][i] for i in indices]
    
    split_data = {
        'filenames': split_filenames,
        'labels': split_labels,
        'indices': indices,
        'label_names': dataset_info['label_names'],
        'metadata': split_info['metadata']
    }
    
    return split_data


def get_all_splits(dataset_info: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Get all available splits (train, test, validation) from the dataset.
    
    Parameters:
    -----------
    dataset_info : dict
        Dataset information dictionary containing split_info
        
    Returns:
    --------
    tuple
        (train_data, test_data, validation_data)
        validation_data is None if no validation split was created
    """
    train_data = get_split_data(dataset_info, "train")
    test_data = get_split_data(dataset_info, "test")
    
    # Check if validation split exists
    validation_data = None
    if 'validation_indices' in dataset_info.get('split_info', {}):
        validation_data = get_split_data(dataset_info, "validation")
    
    return train_data, test_data, validation_data


def print_split_summary(dataset_info: Dict[str, Any]) -> None:
    """
    Print a summary of the dataset splits.
    
    Parameters:
    -----------
    dataset_info : dict
        Dataset information dictionary containing split_info
    """
    if 'split_info' not in dataset_info:
        print("❌ No split information available in dataset")
        return
    
    split_info = dataset_info['split_info']
    metadata = split_info['metadata']
    
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Split seed: {metadata['split_seed']}")
    print(f"Stratified: {metadata['stratified']}")
    print()
    
    # Training split
    print(f"🏋️  Training:   {metadata['train_samples']:4d} samples ({metadata['train_ratio']:.1%})")
    
    # Test split  
    print(f"🧪 Testing:    {metadata['test_samples']:4d} samples ({metadata['test_ratio']:.1%})")
    
    # Validation split (if exists)
    if 'validation_samples' in metadata:
        print(f"✅ Validation: {metadata['validation_samples']:4d} samples ({metadata['validation_ratio']:.1%})")
    
    # Class distribution per split
    print("\nClass distribution per split:")
    train_data = get_split_data(dataset_info, "train")
    test_data = get_split_data(dataset_info, "test")
    
    for i, class_name in enumerate(dataset_info['label_names']):
        train_count = sum(1 for label in train_data['labels'] if label == i)
        test_count = sum(1 for label in test_data['labels'] if label == i)
        
        print(f"  {class_name}:")
        print(f"    Train: {train_count:3d}, Test: {test_count:3d}", end="")
        
        if 'validation_indices' in dataset_info.get('split_info', {}):
            val_data = get_split_data(dataset_info, "validation")
            val_count = sum(1 for label in val_data['labels'] if label == i)
            print(f", Val: {val_count:3d}")
        else:
            print()

def create_train_validation_test_split(dataset_info, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2, 
                                      random_seed=42, batch_size=32, shuffle_train=True):
    """
    Create proper train/validation/test split with data loaders following ML best practices.
    
    This function implements the scientifically rigorous 3-way split where:
    - Training data: Used for model parameter updates via backpropagation
    - Validation data: Used for monitoring during training, early stopping, hyperparameter tuning
    - Test data: Used ONLY for final unbiased evaluation (completely unseen during training)
    
    Args:
        dataset_info: Dataset information dictionary from generate_dataset()
        train_ratio: Proportion of data for training (default: 0.6)
        validation_ratio: Proportion of data for validation (default: 0.2) 
        test_ratio: Proportion of data for testing (default: 0.2)
        random_seed: Random seed for reproducible splits
        batch_size: Batch size for training data loader
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Dictionary containing:
        {
            'train_loader': DataLoader for training,
            'validation_data': Tensor of validation data,
            'validation_labels': Tensor of validation labels,
            'test_data': Tensor of test data (for final evaluation only),
            'test_labels': Tensor of test labels,
            'split_info': Information about the data split,
            'class_names': List of class names,
            'data_loader': StandardDataLoader instance for advanced usage
        }
        
    Example:
        # Generate dataset
        dataset_info = generate_dataset("geological", "my_dataset", num_samples_per_class=500)
        
        # Create proper 3-way split
        data_split = create_train_validation_test_split(
            dataset_info, 
            train_ratio=0.6, 
            validation_ratio=0.2, 
            test_ratio=0.2
        )
        
        # Use with new experiment runner
        runner = ExperimentRunner()
        model, history = runner.train_autoencoder(
            model=my_model,
            train_loader=data_split['train_loader'],
            validation_data=data_split['validation_data'],
            validation_labels=data_split['validation_labels'],
            test_data=data_split['test_data'],  # Optional - for final evaluation
            test_labels=data_split['test_labels'],
            class_names=data_split['class_names']
        )
    """
    from .loaders import StandardDataLoader
    import torch
    
    # Validate split ratios
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    print(f"🔄 Creating 3-way data split: {train_ratio*100:.1f}% train, "
          f"{validation_ratio*100:.1f}% validation, {test_ratio*100:.1f}% test")
    
    # Create data loader with 3-way split capability
    data_loader = StandardDataLoader(
        dataset_info=dataset_info,
        random_seed=random_seed,
        batch_size=batch_size,
        shuffle_train=shuffle_train
    )
    
    # Create the split
    split_info = data_loader.create_split(
        dataset_info=dataset_info,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        validation_ratio=validation_ratio,
        random_seed=random_seed
    )
    
    # Create data loaders
    loaders = data_loader.create_data_loaders(dataset_info, split_info)
    train_loader = loaders[0]
    test_loader = loaders[1]
    validation_loader = loaders[2] if len(loaders) > 2 else None
    
    if validation_loader is None:
        raise ValueError("Validation split was not created properly")
    
    # Load validation and test data as tensors for monitoring/evaluation
    tensors = data_loader.load_data_tensors(dataset_info, split_info)
    train_data_tensor, train_labels_tensor = tensors[0], tensors[1]
    test_data_tensor, test_labels_tensor = tensors[2], tensors[3]
    validation_data_tensor, validation_labels_tensor = tensors[4], tensors[5]
    
    # Extract class names if available
    class_names = dataset_info.get('class_names', None)
    
    # Print split summary
    print(f"✅ Data split created successfully:")
    print(f"   • Training: {len(train_data_tensor)} samples")
    print(f"   • Validation: {len(validation_data_tensor)} samples (for monitoring)")
    print(f"   • Test: {len(test_data_tensor)} samples (for final evaluation)")
    if class_names:
        print(f"   • Classes: {len(class_names)} ({', '.join(class_names)})")
    
    return {
        'train_loader': train_loader,
        'validation_data': validation_data_tensor,
        'validation_labels': validation_labels_tensor,
        'test_data': test_data_tensor,
        'test_labels': test_labels_tensor,
        'split_info': split_info,
        'class_names': class_names,
        'data_loader': data_loader,
        'train_data_tensor': train_data_tensor,  # For compatibility
        'train_labels_tensor': train_labels_tensor
    } 
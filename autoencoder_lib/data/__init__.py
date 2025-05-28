"""
Data Module for AutoEncoder Experimentation

This module provides comprehensive data handling capabilities including:
- Abstract base classes for datasets and data loaders
- Concrete implementations for geological datasets
- Data loading and preprocessing utilities
- Visualization functions for datasets
- Data augmentation and normalization tools
- High-level wrapper functions for easy notebook usage

Key Components:
- base: Abstract base classes (BaseDataset, BaseDataLoader)
- geological: Geological dataset implementation (new framework format)
- layered_geological: Original layered geological dataset (from AutoEncoderJupyterTest.ipynb)
- loaders: Standard data loader implementation
- preprocessing: Data normalization, augmentation, and transformation utilities
- wrappers: High-level wrapper functions for dataset generation and visualization

Usage:
    from autoencoder_lib.data import generate_dataset, visualize_dataset
    
    # Generate original format dataset (compatible with AutoEncoderJupyterTest.ipynb)
    dataset_info = generate_dataset(
        dataset_type="layered_geological",
        output_dir="my_dataset",
        num_samples_per_class=500,
        image_size=64
    )
    
    # Generate new framework format dataset
    dataset_info = generate_dataset(
        dataset_type="geological",
        output_dir="my_dataset",
        num_samples_per_class=100,
        image_size=64
    )
    
    # Visualize any dataset
    analysis = visualize_dataset(dataset_info=dataset_info)
"""

# Import base classes
from .base import BaseDataset, BaseDataLoader

# Import concrete implementations
from .geological import LayeredGeologicalDataset
from .loaders import StandardDataLoader, ShapeDataset

# Import preprocessing utilities
from .preprocessing import (
    StandardNormalizer,
    MinMaxNormalizer, 
    RobustNormalizer,
    DataAugmenter,
    PreprocessingPipeline,
    calculate_data_statistics
)

# Import original layered geological dataset functions
from .layered_geological import (
    generate_layered_dataset,
    visualize_dataset_samples,
    load_layered_dataset,
    get_dataset_statistics
)

# Import high-level wrapper functions
from .wrappers import generate_dataset, visualize_dataset

# Define public API
__all__ = [
    # Base classes
    'BaseDataset',
    'BaseDataLoader',
    
    # Concrete implementations
    'LayeredGeologicalDataset',
    'StandardDataLoader',
    'ShapeDataset',
    
    # Preprocessing utilities
    'StandardNormalizer',
    'MinMaxNormalizer',
    'RobustNormalizer', 
    'DataAugmenter',
    'PreprocessingPipeline',
    'calculate_data_statistics',
    
    # Original layered geological dataset functions
    'generate_layered_dataset',
    'visualize_dataset_samples',
    'load_layered_dataset',
    'get_dataset_statistics',
    
    # High-level wrapper functions (recommended interface)
    'generate_dataset',
    'visualize_dataset'
]

# Version information
__version__ = '0.1.0' 
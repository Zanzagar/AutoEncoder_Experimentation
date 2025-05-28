"""
Data Module for AutoEncoder Experimentation

This module provides comprehensive data handling capabilities including:
- Abstract base classes for datasets and data loaders
- Concrete implementations for geological datasets
- Data loading and preprocessing utilities
- Visualization functions for datasets
- Data augmentation and normalization tools

Key Components:
- base: Abstract base classes (BaseDataset, BaseDataLoader)
- geological: Geological dataset implementation
- loaders: Standard data loader implementation
- preprocessing: Data normalization, augmentation, and transformation utilities

Usage:
    from autoencoder_lib.data import LayeredGeologicalDataset, StandardDataLoader
    from autoencoder_lib.data.preprocessing import MinMaxNormalizer, DataAugmenter
"""

from .base import BaseDataset, BaseDataLoader
from .geological import LayeredGeologicalDataset
from .loaders import StandardDataLoader, ShapeDataset

# Import preprocessing utilities
from .preprocessing import (
    BasePreprocessor,
    MinMaxNormalizer,
    StandardNormalizer,
    RobustNormalizer,
    DataAugmenter,
    PreprocessingPipeline,
    create_standard_pipeline,
    create_robust_pipeline,
    create_zscore_pipeline,
    preprocess_for_pytorch,
    batch_preprocess,
    calculate_data_statistics
)

__all__ = [
    # Base classes
    'BaseDataset',
    'BaseDataLoader',
    
    # Concrete implementations
    'LayeredGeologicalDataset',
    'StandardDataLoader',
    'ShapeDataset',
    
    # Preprocessing
    'BasePreprocessor',
    'MinMaxNormalizer',
    'StandardNormalizer',
    'RobustNormalizer',
    'DataAugmenter',
    'PreprocessingPipeline',
    'create_standard_pipeline',
    'create_robust_pipeline',
    'create_zscore_pipeline',
    'preprocess_for_pytorch',
    'batch_preprocess',
    'calculate_data_statistics'
]

# Version information
__version__ = '0.1.0' 
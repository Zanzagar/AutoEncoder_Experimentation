"""
Abstract base classes for dataset handling in autoencoder experiments.

This module provides the foundation for all dataset implementations,
ensuring consistent interfaces and behavior across different dataset types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from pathlib import Path


class BaseDataset(ABC):
    """
    Abstract base class for all dataset implementations.
    
    Provides a consistent interface for dataset generation, loading,
    and preprocessing operations.
    """
    
    def __init__(self, name: str, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the base dataset.
        
        Args:
            name: Name identifier for the dataset
            output_dir: Directory to save dataset files (optional)
        """
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else None
        self._dataset_info = None
        self._is_generated = False
    
    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate the dataset with specified parameters.
        
        Returns:
            Dictionary containing dataset information and metadata
        """
        pass
    
    @abstractmethod
    def load(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an existing dataset from disk.
        
        Args:
            dataset_path: Path to the dataset directory or file
            
        Returns:
            Dictionary containing dataset information and metadata
        """
        pass
    
    @abstractmethod
    def save(self, dataset_info: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset_info: Dataset information dictionary
            output_path: Path where to save the dataset
        """
        pass
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get dataset information.
        
        Returns:
            Dataset information dictionary if available, None otherwise
        """
        return self._dataset_info
    
    def is_generated(self) -> bool:
        """
        Check if dataset has been generated.
        
        Returns:
            True if dataset is generated, False otherwise
        """
        return self._is_generated


class BaseDataLoader(ABC):
    """
    Abstract base class for data loading and preprocessing.
    
    Handles data splitting, normalization, and batch preparation
    for training and evaluation.
    """
    
    def __init__(self, dataset_info: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            dataset_info: Dataset information dictionary
        """
        self.dataset_info = dataset_info
        self._train_data = None
        self._test_data = None
        self._split_info = None
    
    @abstractmethod
    def create_split(self, test_ratio: float = 0.2, seed: int = 42) -> Dict[str, Any]:
        """
        Create train/test split of the dataset.
        
        Args:
            test_ratio: Fraction of data to use for testing
            seed: Random seed for reproducible splits
            
        Returns:
            Dictionary containing split information
        """
        pass
    
    @abstractmethod
    def load_split(self, split_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load existing train/test split from disk.
        
        Args:
            split_path: Path to the split information file
            
        Returns:
            Dictionary containing split information
        """
        pass
    
    @abstractmethod
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (features, labels) for training
        """
        pass
    
    @abstractmethod
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get test data.
        
        Returns:
            Tuple of (features, labels) for testing
        """
        pass
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data using specified method.
        
        Args:
            data: Input data array
            method: Normalization method ('minmax', 'zscore', 'unit')
            
        Returns:
            Normalized data array
        """
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'unit':
            return data / np.linalg.norm(data, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown normalization method: {method}") 
"""
Concrete data loader implementations for autoencoder experiments.

This module provides concrete implementations of the BaseDataLoader interface,
including functionality for data splitting, normalization, and batch preparation.
"""

import os
import json
import time
import glob
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

from .base import BaseDataLoader

# Configure logging
logger = logging.getLogger(__name__)


class ShapeDataset(Dataset):
    """
    PyTorch Dataset class for loading shape/pattern images.
    
    This class handles loading images from either a directory structure,
    from pre-loaded dataset information dictionaries, or from direct arrays.
    """
    
    def __init__(self, 
                 dataset_info_or_images: Union[str, Dict[str, Any], np.ndarray], 
                 labels: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Args:
            dataset_info_or_images: Either:
                - A path to the dataset directory
                - A dictionary containing 'filenames' and 'labels'
                - A numpy array of images (if labels is also provided)
            labels: Optional labels array (only used if first arg is images array)
        """
        if labels is not None:
            # Direct images and labels provided
            self.images = dataset_info_or_images
            self.labels = labels
            self.from_arrays = True
            self.from_directory = False
        elif isinstance(dataset_info_or_images, dict):
            # It's already a dictionary with filenames and labels
            self.filenames = dataset_info_or_images['filenames']
            self.labels = dataset_info_or_images['labels']
            self.from_arrays = False
            self.from_directory = False
        else:
            # It's a directory path - load from there
            self.dataset_dir = dataset_info_or_images
            self.classes = self._get_classes()
            self.filenames = []
            self.labels = []
            self._load_from_directory()
            self.from_arrays = False
            self.from_directory = True
    
    def _get_classes(self) -> List[str]:
        """Get class folders in the dataset directory."""
        class_dirs = [d for d in os.listdir(self.dataset_dir) 
                     if os.path.isdir(os.path.join(self.dataset_dir, d))]
        logger.info(f"Found classes: {class_dirs}")
        return class_dirs
    
    def _load_from_directory(self) -> None:
        """Load all images from the directory structure."""
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_dir, class_name)
            class_files = glob.glob(os.path.join(class_dir, '*.png'))
            
            for file in class_files:
                self.filenames.append(file)
                self.labels.append(i)
    
    def __len__(self) -> int:
        if self.from_arrays:
            return len(self.images)
        else:
            return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (input_tensor, target_tensor, label)
        """
        if self.from_arrays:
            # Direct array access
            img = self.images[idx]
            label = self.labels[idx]
            
            # Ensure image is in [0, 1] range and has correct shape
            if img.max() > 1.0:
                img = img / 255.0
            
            # Convert to tensor and add channel dimension if needed
            img = torch.tensor(img).float()
            if len(img.shape) == 2:  # Add channel dimension [H, W] -> [1, H, W]
                img = img.unsqueeze(0)
            
            label = torch.tensor(label).long()
            
            return img, img, label  # Return (input, target, label)
        else:
            # File-based access
            img_path = self.filenames[idx]
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            label = self.labels[idx]
            
            # Transform to tensor and normalize
            img = np.array(img)
            img = img / 255.0  # Normalize to [0, 1]
            img = torch.tensor(img).float().unsqueeze(0)  # Add channel dimension [1, H, W]
            
            return img, img, label  # Return (input, target, label)


class StandardDataLoader(BaseDataLoader):
    """
    Standard implementation of the BaseDataLoader interface.
    
    This class provides comprehensive data loading, splitting, and preprocessing
    functionality for autoencoder experiments.
    """
    
    def __init__(self, 
                 dataset_info: Optional[Dict[str, Any]] = None,
                 random_seed: int = 42,
                 test_ratio: float = 0.2,
                 validation_ratio: float = 0.0,
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 num_workers: int = 0):
        """
        Initialize the data loader.
        
        Args:
            dataset_info: Dataset information dictionary (can be set later)
            random_seed: Random seed for reproducible splits
            test_ratio: Proportion of data to use for testing
            validation_ratio: Proportion of data to use for validation
            batch_size: Batch size for data loaders
            shuffle_train: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
        """
        # Initialize parent class with dummy dataset_info if not provided
        super().__init__(dataset_info or {})
        
        self.random_seed = random_seed
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Storage for split information
        self._split_info = None
        self._train_dataset = None
        self._test_dataset = None
        self._validation_dataset = None
    
    def set_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Set the dataset information.
        
        Args:
            dataset_info: Dataset information dictionary
        """
        self.dataset_info = dataset_info
    
    def seed_worker(self, worker_id: int) -> None:
        """
        Function to ensure DataLoader workers have deterministic behavior.
        
        Args:
            worker_id: Worker process ID
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
    
    def create_split(self, 
                    dataset_info: Dict[str, Any], 
                    train_ratio: float = 0.7,
                    test_ratio: float = 0.3,
                    validation_ratio: float = 0.0,
                    random_seed: Optional[int] = None,
                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create train/test/validation split for the dataset.
        
        Args:
            dataset_info: Dataset information dictionary
            train_ratio: Fraction of data to use for training
            test_ratio: Fraction of data to use for testing
            validation_ratio: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits
            save_path: Optional path to save split information
            
        Returns:
            Dictionary containing split indices and metadata
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Determine total samples based on dataset structure
        if 'images' in dataset_info:
            # Direct arrays case
            total_samples = len(dataset_info['images'])
        elif 'filenames' in dataset_info:
            # Filenames case
            total_samples = len(dataset_info['filenames'])
        else:
            raise ValueError("Dataset info must contain either 'images' or 'filenames'")
            
        logger.info(f"Creating data split with train_ratio={train_ratio}, "
                   f"test_ratio={test_ratio}, validation_ratio={validation_ratio}, "
                   f"seed={random_seed}")
        
        indices = np.random.permutation(total_samples)
        
        # Calculate split indices
        test_size = int(total_samples * test_ratio)
        val_size = int(total_samples * validation_ratio)
        train_size = total_samples - test_size - val_size
        
        # Create splits
        train_indices = indices[:train_size].tolist()
        test_indices = indices[train_size:train_size + test_size].tolist()
        
        split_info = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'metadata': {
                'random_seed': random_seed,
                'train_ratio': train_ratio,
                'test_ratio': test_ratio,
                'validation_ratio': validation_ratio,
                'total_samples': total_samples,
                'train_samples': len(train_indices),
                'test_samples': len(test_indices)
            }
        }
        
        # Add validation split if requested
        if validation_ratio > 0:
            val_indices = indices[train_size + test_size:].tolist()
            split_info['validation_indices'] = val_indices
            split_info['metadata']['validation_samples'] = len(val_indices)
        
        # Save split information if path provided
        if save_path:
            self.save_split(split_info, save_path)
        
        self._split_info = split_info
        logger.info(f"Created split: {len(train_indices)} train, {len(test_indices)} test"
                   + (f", {len(val_indices)} validation" if validation_ratio > 0 else ""))
        
        return split_info
    
    def load_split(self, split_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an existing data split from file.
        
        Args:
            split_path: Path to the split file
            
        Returns:
            Dictionary containing split information
        """
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, 'r') as f:
            split_info = json.load(f)
        
        self._split_info = split_info
        logger.info(f"Loaded split from {split_path}: "
                   f"{len(split_info['train_indices'])} train, "
                   f"{len(split_info['test_indices'])} test")
        
        return split_info
    
    def save_split(self, split_info: Dict[str, Any], save_path: str) -> None:
        """
        Save split information to file.
        
        Args:
            split_info: Split information dictionary
            save_path: Path to save the split file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split information saved to {save_path}")
    
    def load_or_create_split(self, 
                           dataset_info: Dict[str, Any],
                           split_path: str) -> Dict[str, Any]:
        """
        Load an existing split or create a new one if it doesn't exist.
        
        Args:
            dataset_info: Dataset information dictionary
            split_path: Path to the split file
            
        Returns:
            Dictionary containing split information
        """
        if os.path.exists(split_path):
            return self.load_split(split_path)
        else:
            return self.create_split(dataset_info, split_path)
    
    def get_split_datasets(self, 
                          dataset_info: Dict[str, Any], 
                          split_info: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], ...]:
        """
        Create dataset dictionaries for each split.
        
        Args:
            dataset_info: Original dataset information
            split_info: Split information (uses stored split if None)
            
        Returns:
            Tuple of dataset dictionaries (train, test, [validation])
        """
        if split_info is None:
            if self._split_info is None:
                raise ValueError("No split information available. Create or load a split first.")
            split_info = self._split_info
        
        # Create train dataset info
        train_info = {
            'filenames': [dataset_info['filenames'][i] for i in split_info['train_indices']],
            'labels': [dataset_info['labels'][i] for i in split_info['train_indices']]
        }
        
        # Create test dataset info
        test_info = {
            'filenames': [dataset_info['filenames'][i] for i in split_info['test_indices']],
            'labels': [dataset_info['labels'][i] for i in split_info['test_indices']]
        }
        
        # Create validation dataset info if available
        if 'validation_indices' in split_info:
            val_info = {
                'filenames': [dataset_info['filenames'][i] for i in split_info['validation_indices']],
                'labels': [dataset_info['labels'][i] for i in split_info['validation_indices']]
            }
            return train_info, test_info, val_info
        
        return train_info, test_info
    
    def create_data_loaders(self, 
                           dataset_info: Dict[str, Any],
                           split_info: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, ...]:
        """
        Create PyTorch DataLoaders for train/test/validation splits.
        
        Args:
            dataset_info: Dataset information dictionary
            split_info: Split information (uses stored split if None)
            
        Returns:
            Tuple of DataLoaders (train, test, [validation])
        """
        # Get split datasets
        split_datasets = self.get_split_datasets(dataset_info, split_info)
        train_info, test_info = split_datasets[:2]
        
        # Create PyTorch datasets
        train_dataset = ShapeDataset(train_info)
        test_dataset = ShapeDataset(test_info)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker if self.num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker if self.num_workers > 0 else None
        )
        
        # Store datasets for later use
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        
        # Create validation loader if validation split exists
        if len(split_datasets) > 2:
            val_info = split_datasets[2]
            val_dataset = ShapeDataset(val_info)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=self.seed_worker if self.num_workers > 0 else None
            )
            self._validation_dataset = val_dataset
            return train_loader, test_loader, val_loader
        
        return train_loader, test_loader
    
    def load_data_tensors(self, 
                         dataset_info: Dict[str, Any],
                         split_info: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, ...]:
        """
        Load all data into memory as tensors for visualization and analysis.
        
        Args:
            dataset_info: Dataset information dictionary
            split_info: Split information (uses stored split if None)
            
        Returns:
            Tuple of (train_data, train_labels, test_data, test_labels, [val_data, val_labels])
        """
        # Get split datasets
        split_datasets = self.get_split_datasets(dataset_info, split_info)
        train_info, test_info = split_datasets[:2]
        
        # Load training data
        train_dataset = ShapeDataset(train_info)
        train_data = []
        train_labels = []
        
        for i in range(len(train_dataset)):
            img, _, label = train_dataset[i]
            train_data.append(img)
            train_labels.append(label)
        
        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)
        
        # Load test data
        test_dataset = ShapeDataset(test_info)
        test_data = []
        test_labels = []
        
        for i in range(len(test_dataset)):
            img, _, label = test_dataset[i]
            test_data.append(img)
            test_labels.append(label)
        
        test_data = torch.stack(test_data)
        test_labels = torch.tensor(test_labels)
        
        # Load validation data if available
        if len(split_datasets) > 2:
            val_info = split_datasets[2]
            val_dataset = ShapeDataset(val_info)
            val_data = []
            val_labels = []
            
            for i in range(len(val_dataset)):
                img, _, label = val_dataset[i]
                val_data.append(img)
                val_labels.append(label)
            
            val_data = torch.stack(val_data)
            val_labels = torch.tensor(val_labels)
            
            return train_data, train_labels, test_data, test_labels, val_data, val_labels
        
        return train_data, train_labels, test_data, test_labels
    
    def normalize_data(self, 
                      data: torch.Tensor, 
                      method: str = 'minmax',
                      fit_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize data using specified method.
        
        Args:
            data: Data tensor to normalize
            method: Normalization method ('minmax', 'zscore', 'none')
            fit_data: Data to fit normalization parameters on (if None, uses data)
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'none':
            return data, {'method': 'none'}
        
        # Use fit_data for computing parameters if provided, otherwise use data
        reference_data = fit_data if fit_data is not None else data
        
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = reference_data.min()
            max_val = reference_data.max()
            
            if max_val - min_val == 0:
                logger.warning("Data has zero variance, normalization will have no effect")
                normalized_data = data
            else:
                normalized_data = (data - min_val) / (max_val - min_val)
            
            params = {
                'method': 'minmax',
                'min_val': min_val.item(),
                'max_val': max_val.item()
            }
            
        elif method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean_val = reference_data.mean()
            std_val = reference_data.std()
            
            if std_val == 0:
                logger.warning("Data has zero variance, normalization will have no effect")
                normalized_data = data - mean_val
            else:
                normalized_data = (data - mean_val) / std_val
            
            params = {
                'method': 'zscore',
                'mean_val': mean_val.item(),
                'std_val': std_val.item()
            }
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_data, params
    
    def denormalize_data(self, 
                        normalized_data: torch.Tensor, 
                        normalization_params: Dict[str, Any]) -> torch.Tensor:
        """
        Reverse normalization using stored parameters.
        
        Args:
            normalized_data: Normalized data tensor
            normalization_params: Parameters from normalization
            
        Returns:
            Denormalized data tensor
        """
        method = normalization_params['method']
        
        if method == 'none':
            return normalized_data
        elif method == 'minmax':
            min_val = normalization_params['min_val']
            max_val = normalization_params['max_val']
            return normalized_data * (max_val - min_val) + min_val
        elif method == 'zscore':
            mean_val = normalization_params['mean_val']
            std_val = normalization_params['std_val']
            return normalized_data * std_val + mean_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_class_distribution(self, 
                             dataset_info: Dict[str, Any],
                             split_info: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[int, int]]:
        """
        Get class distribution for each split.
        
        Args:
            dataset_info: Dataset information dictionary
            split_info: Split information (uses stored split if None)
            
        Returns:
            Dictionary with class distributions for each split
        """
        if split_info is None:
            if self._split_info is None:
                raise ValueError("No split information available. Create or load a split first.")
            split_info = self._split_info
        
        distributions = {}
        
        # Training distribution
        train_labels = [dataset_info['labels'][i] for i in split_info['train_indices']]
        train_dist = {}
        for label in train_labels:
            train_dist[label] = train_dist.get(label, 0) + 1
        distributions['train'] = train_dist
        
        # Test distribution
        test_labels = [dataset_info['labels'][i] for i in split_info['test_indices']]
        test_dist = {}
        for label in test_labels:
            test_dist[label] = test_dist.get(label, 0) + 1
        distributions['test'] = test_dist
        
        # Validation distribution if available
        if 'validation_indices' in split_info:
            val_labels = [dataset_info['labels'][i] for i in split_info['validation_indices']]
            val_dist = {}
            for label in val_labels:
                val_dist[label] = val_dist.get(label, 0) + 1
            distributions['validation'] = val_dist
        
        return distributions
    
    def get_split_info(self) -> Optional[Dict[str, Any]]:
        """
        Get the current split information.
        
        Returns:
            Split information dictionary or None if no split created
        """
        return self._split_info
    
    def get_datasets(self) -> Tuple[Optional[ShapeDataset], ...]:
        """
        Get the created PyTorch datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset, validation_dataset)
        """
        return self._train_dataset, self._test_dataset, self._validation_dataset
    
    def get_train_data(self, dataset_info: Dict[str, Any], split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get training data.
        
        Args:
            dataset_info: Dataset information dictionary
            split_info: Split information dictionary
            
        Returns:
            Dictionary containing training data
        """
        train_indices = split_info['train_indices']
        
        if 'images' in dataset_info:
            # Direct image data
            train_images = dataset_info['images'][train_indices]
            train_labels = np.array([dataset_info['labels'][i] for i in train_indices])
        else:
            # Load from filenames
            train_images = []
            train_labels = []
            
            for idx in train_indices:
                img_path = dataset_info['filenames'][idx]
                img = Image.open(img_path).convert('L')
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                train_images.append(img_array)
                train_labels.append(dataset_info['labels'][idx])
            
            train_images = np.array(train_images)
            train_labels = np.array(train_labels)
        
        return {
            'images': train_images,
            'labels': train_labels
        }
    
    def get_test_data(self, dataset_info: Dict[str, Any], split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get test data.
        
        Args:
            dataset_info: Dataset information dictionary
            split_info: Split information dictionary
            
        Returns:
            Dictionary containing test data
        """
        test_indices = split_info['test_indices']
        
        if 'images' in dataset_info:
            # Direct image data
            test_images = dataset_info['images'][test_indices]
            test_labels = np.array([dataset_info['labels'][i] for i in test_indices])
        else:
            # Load from filenames
            test_images = []
            test_labels = []
            
            for idx in test_indices:
                img_path = dataset_info['filenames'][idx]
                img = Image.open(img_path).convert('L')
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                test_images.append(img_array)
                test_labels.append(dataset_info['labels'][idx])
            
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
        
        return {
            'images': test_images,
            'labels': test_labels
        } 
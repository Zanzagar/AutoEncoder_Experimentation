"""
Data Preprocessing Utilities

This module provides comprehensive data preprocessing capabilities including:
- Normalization methods (min-max, z-score, robust scaling)
- Data augmentation techniques for geological and general image data
- Transformation pipelines with reversible operations
- Batch preprocessing utilities
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
import random
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors."""
    
    def __init__(self):
        self.is_fitted = False
        self.parameters = {}
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BasePreprocessor':
        """Fit the preprocessor to the data."""
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using fitted parameters."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the transformation."""
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform the data in one step."""
        return self.fit(data).transform(data)


class MinMaxNormalizer(BasePreprocessor):
    """Min-Max normalization to [0, 1] range."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def fit(self, data: np.ndarray) -> 'MinMaxNormalizer':
        """Fit the normalizer to the data."""
        original_shape = data.shape
        data_flat = data.reshape(-1, 1) if data.ndim > 1 else data.reshape(-1, 1)
        self.scaler.fit(data_flat)
        self.parameters['original_shape'] = original_shape
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to normalized range."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        normalized_flat = self.scaler.transform(data_flat)
        return normalized_flat.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        denormalized_flat = self.scaler.inverse_transform(data_flat)
        return denormalized_flat.reshape(original_shape)


class StandardNormalizer(BasePreprocessor):
    """Z-score normalization (mean=0, std=1)."""
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray) -> 'StandardNormalizer':
        """Fit the normalizer to the data."""
        original_shape = data.shape
        data_flat = data.reshape(-1, 1) if data.ndim > 1 else data.reshape(-1, 1)
        self.scaler.fit(data_flat)
        self.parameters['original_shape'] = original_shape
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to z-score normalized form."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        normalized_flat = self.scaler.transform(data_flat)
        return normalized_flat.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the z-score normalization."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        denormalized_flat = self.scaler.inverse_transform(data_flat)
        return denormalized_flat.reshape(original_shape)


class RobustNormalizer(BasePreprocessor):
    """Robust normalization using median and IQR."""
    
    def __init__(self):
        super().__init__()
        self.scaler = RobustScaler()
    
    def fit(self, data: np.ndarray) -> 'RobustNormalizer':
        """Fit the normalizer to the data."""
        original_shape = data.shape
        data_flat = data.reshape(-1, 1) if data.ndim > 1 else data.reshape(-1, 1)
        self.scaler.fit(data_flat)
        self.parameters['original_shape'] = original_shape
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using robust scaling."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        normalized_flat = self.scaler.transform(data_flat)
        return normalized_flat.reshape(original_shape)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the robust normalization."""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        denormalized_flat = self.scaler.inverse_transform(data_flat)
        return denormalized_flat.reshape(original_shape)


class DataAugmenter:
    """Data augmentation utilities for geological and general image data."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def add_noise(
        self, 
        data: np.ndarray, 
        noise_type: str = 'gaussian',
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Add noise to the data."""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, data.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, data.shape)
        elif noise_type == 'salt_pepper':
            noise = np.random.choice([-noise_level, 0, noise_level], 
                                   size=data.shape, 
                                   p=[0.1, 0.8, 0.1])
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return np.clip(data + noise, 0, 1)
    
    def rotate_image(
        self, 
        image: np.ndarray, 
        angle_range: Tuple[float, float] = (-15, 15)
    ) -> np.ndarray:
        """Rotate image by random angle within range."""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        return ndimage.rotate(image, angle, reshape=False, mode='nearest')
    
    def flip_image(
        self, 
        image: np.ndarray, 
        horizontal: bool = True, 
        vertical: bool = True
    ) -> np.ndarray:
        """Randomly flip image horizontally and/or vertically."""
        result = image.copy()
        
        if horizontal and np.random.random() > 0.5:
            result = np.fliplr(result)
        
        if vertical and np.random.random() > 0.5:
            result = np.flipud(result)
        
        return result
    
    def scale_image(
        self, 
        image: np.ndarray, 
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """Scale image by random factor within range."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        # Use order=1 for linear interpolation and preserve shape
        scaled = ndimage.zoom(image, scale_factor, order=1, mode='nearest')
        
        # Ensure output has same shape as input by cropping or padding
        if scaled.shape != image.shape:
            if scaled.shape[0] > image.shape[0] or scaled.shape[1] > image.shape[1]:
                # Crop to original size
                start_y = (scaled.shape[0] - image.shape[0]) // 2
                start_x = (scaled.shape[1] - image.shape[1]) // 2
                scaled = scaled[start_y:start_y + image.shape[0], 
                              start_x:start_x + image.shape[1]]
            else:
                # Pad to original size
                pad_y = (image.shape[0] - scaled.shape[0]) // 2
                pad_x = (image.shape[1] - scaled.shape[1]) // 2
                scaled = np.pad(scaled, 
                              ((pad_y, image.shape[0] - scaled.shape[0] - pad_y),
                               (pad_x, image.shape[1] - scaled.shape[1] - pad_x)),
                              mode='edge')
        
        return scaled
    
    def shift_image(
        self, 
        image: np.ndarray, 
        shift_range: Tuple[int, int] = (-2, 2)
    ) -> np.ndarray:
        """Shift image by random offset within range."""
        shift_x = np.random.randint(shift_range[0], shift_range[1] + 1)
        shift_y = np.random.randint(shift_range[0], shift_range[1] + 1)
        return ndimage.shift(image, (shift_y, shift_x), mode='nearest')
    
    def geological_augmentation(
        self, 
        image: np.ndarray, 
        augmentation_probability: float = 0.5
    ) -> np.ndarray:
        """Apply geological-specific augmentations."""
        result = image.copy()
        
        # Add geological noise (simulating measurement noise)
        if np.random.random() < augmentation_probability:
            result = self.add_noise(result, 'gaussian', 0.05)
        
        # Slight rotation (geological structures can be tilted)
        if np.random.random() < augmentation_probability:
            result = self.rotate_image(result, (-5, 5))
        
        # Horizontal flip (geological structures can be mirrored)
        if np.random.random() < augmentation_probability:
            result = self.flip_image(result, horizontal=True, vertical=False)
        
        # Slight scaling (different zoom levels in geological surveys)
        if np.random.random() < augmentation_probability:
            result = self.scale_image(result, (0.95, 1.05))
        
        return np.clip(result, 0, 1)
    
    def augment_batch(
        self, 
        images: np.ndarray, 
        labels: np.ndarray,
        augmentation_factor: int = 2,
        geological_mode: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment a batch of images."""
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            # Keep original
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])
            
            # Create augmented versions
            for _ in range(augmentation_factor - 1):
                if geological_mode:
                    aug_image = self.geological_augmentation(images[i])
                else:
                    # General augmentation
                    aug_image = images[i].copy()
                    aug_image = self.add_noise(aug_image, 'gaussian', 0.1)
                    aug_image = self.rotate_image(aug_image, (-10, 10))
                    aug_image = self.flip_image(aug_image)
                
                # Ensure augmented image has the same shape as original
                if aug_image.shape != images[i].shape:
                    # Resize to match original shape if needed
                    from scipy import ndimage
                    zoom_factors = [images[i].shape[j] / aug_image.shape[j] for j in range(len(images[i].shape))]
                    aug_image = ndimage.zoom(aug_image, zoom_factors, order=1, mode='nearest')
                
                augmented_images.append(aug_image)
                augmented_labels.append(labels[i])
        
        # Convert to numpy arrays, ensuring all images have the same shape
        try:
            augmented_images_array = np.array(augmented_images)
            augmented_labels_array = np.array(augmented_labels)
        except ValueError as e:
            # If there are still shape mismatches, fix them
            target_shape = images[0].shape
            fixed_images = []
            for img in augmented_images:
                if img.shape != target_shape:
                    # Resize to target shape
                    zoom_factors = [target_shape[j] / img.shape[j] for j in range(len(target_shape))]
                    img = ndimage.zoom(img, zoom_factors, order=1, mode='nearest')
                fixed_images.append(img)
            
            augmented_images_array = np.array(fixed_images)
            augmented_labels_array = np.array(augmented_labels)
        
        return augmented_images_array, augmented_labels_array


class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessing operations."""
    
    def __init__(self, steps: List[Tuple[str, BasePreprocessor]]):
        self.steps = steps
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'PreprocessingPipeline':
        """Fit all preprocessors in the pipeline."""
        current_data = data
        for name, preprocessor in self.steps:
            preprocessor.fit(current_data)
            current_data = preprocessor.transform(current_data)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data through the pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        current_data = data
        for name, preprocessor in self.steps:
            current_data = preprocessor.transform(current_data)
        
        return current_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse transform through the pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before inverse_transform")
        
        current_data = data
        # Apply inverse transforms in reverse order
        for name, preprocessor in reversed(self.steps):
            current_data = preprocessor.inverse_transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)


def create_standard_pipeline() -> PreprocessingPipeline:
    """Create a standard preprocessing pipeline."""
    return PreprocessingPipeline([
        ('minmax', MinMaxNormalizer(feature_range=(0, 1)))
    ])


def create_robust_pipeline() -> PreprocessingPipeline:
    """Create a robust preprocessing pipeline."""
    return PreprocessingPipeline([
        ('robust', RobustNormalizer())
    ])


def create_zscore_pipeline() -> PreprocessingPipeline:
    """Create a z-score normalization pipeline."""
    return PreprocessingPipeline([
        ('zscore', StandardNormalizer())
    ])


def preprocess_for_pytorch(
    data: np.ndarray, 
    labels: np.ndarray,
    normalizer: Optional[BasePreprocessor] = None,
    augment: bool = False,
    augmentation_factor: int = 2,
    geological_mode: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Preprocess data for PyTorch training."""
    
    # Normalize if requested
    if normalizer is not None:
        if not normalizer.is_fitted:
            normalizer.fit(data)
        data = normalizer.transform(data)
    
    # Augment if requested
    if augment:
        augmenter = DataAugmenter()
        data, labels = augmenter.augment_batch(
            data, labels, augmentation_factor, geological_mode
        )
    
    # Convert to PyTorch tensors
    # Add channel dimension if needed (for CNN compatibility)
    if data.ndim == 3:  # (batch, height, width)
        data = data[:, np.newaxis, :, :]  # (batch, 1, height, width)
    
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)
    
    return data_tensor, labels_tensor


def batch_preprocess(
    data_batch: np.ndarray,
    preprocessor: BasePreprocessor,
    device: str = 'cpu'
) -> torch.Tensor:
    """Preprocess a batch of data for inference."""
    
    if not preprocessor.is_fitted:
        raise ValueError("Preprocessor must be fitted before batch processing")
    
    # Preprocess
    processed_data = preprocessor.transform(data_batch)
    
    # Add channel dimension if needed
    if processed_data.ndim == 3:
        processed_data = processed_data[:, np.newaxis, :, :]
    
    # Convert to tensor and move to device
    tensor = torch.FloatTensor(processed_data).to(device)
    
    return tensor


def calculate_data_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for the data."""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q25': float(np.percentile(data, 25)),
        'q75': float(np.percentile(data, 75)),
        'shape': data.shape,
        'dtype': str(data.dtype)
    } 
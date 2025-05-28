"""
Geological dataset generation for autoencoder experiments.

This module provides functionality to generate synthetic geological patterns
including layered structures, fault systems, and other geological features.
Uses the same file structure as the original AutoEncoderJupyterTest.ipynb format.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, Union, List
from pathlib import Path
import json
from PIL import Image
import os
from .base import BaseDataset
import logging

logger = logging.getLogger(__name__)


class LayeredGeologicalDataset(BaseDataset):
    """
    Generator for synthetic layered geological patterns.
    
    Creates datasets with various geological structures including:
    - Horizontal layers
    - Folded structures  
    - Fault systems
    - Intrusions
    - Unconformities
    
    Uses the same file structure as the original AutoEncoderJupyterTest.ipynb format:
    - Individual PNG files for each sample
    - .npy metadata file with filenames, labels, and parameters
    """
    
    def __init__(self, name: str = "layered_geological", output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize LayeredGeologicalDataset.
        
        Args:
            name: Name of the dataset
            output_dir: Default output directory for saving datasets
        """
        super().__init__(name, output_dir)
        
        # Define geological pattern classes
        self.pattern_classes = [
            'horizontal_layers',
            'folded_layers', 
            'faulted_layers',
            'intrusion_patterns',
            'unconformity_patterns'
        ]
        
    def generate(self, 
                 output_dir: str = 'layered_geologic_patterns_dataset',
                 num_samples_per_class: int = 50,
                 image_size: int = 64,
                 num_layers_range: Tuple[int, int] = (3, 8),
                 noise_level: float = 0.1,
                 random_seed: Optional[int] = None,
                 force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate layered geological patterns dataset.
        
        Args:
            output_dir: Directory to save the dataset
            num_samples_per_class: Number of samples per geological pattern class
            image_size: Size of generated images (image_size x image_size)
            num_layers_range: Range for number of layers in each pattern
            noise_level: Amount of noise to add (0.0 to 1.0)
            random_seed: Random seed for reproducibility
            force_regenerate: Force regeneration even if dataset exists
            
        Returns:
            Dictionary containing generated dataset information (same format as original)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        output_path = Path(output_dir)
        
        # Check if dataset already exists
        if not force_regenerate and (output_path / 'dataset_info.npy').exists():
            logger.info(f"Dataset already exists at {output_path}, loading existing dataset")
            return self.load(output_path)
        
        # Create output directory
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Generating {len(self.pattern_classes)} classes of geological patterns...")
        
        all_filenames = []
        all_labels = []
        
        for class_idx, pattern_class in enumerate(self.pattern_classes):
            logger.info(f"  Generating {pattern_class}...")
            
            # Create class subdirectory
            class_dir = output_path / pattern_class
            class_dir.mkdir(exist_ok=True)
            
            for sample_idx in range(num_samples_per_class):
                try:
                    if pattern_class == 'horizontal_layers':
                        image = self._generate_horizontal_layers(image_size, num_layers_range, noise_level)
                    elif pattern_class == 'folded_layers':
                        image = self._generate_folded_layers(image_size, num_layers_range, noise_level)
                    elif pattern_class == 'faulted_layers':
                        image = self._generate_faulted_layers(image_size, num_layers_range, noise_level)
                    elif pattern_class == 'intrusion_patterns':
                        image = self._generate_intrusion_patterns(image_size, num_layers_range, noise_level)
                    elif pattern_class == 'unconformity_patterns':
                        image = self._generate_unconformity_patterns(image_size, num_layers_range, noise_level)
                    else:
                        raise ValueError(f"Unknown pattern class: {pattern_class}")
                    
                    # Convert to 0-255 range for PNG saving
                    image_uint8 = (image * 255).astype(np.uint8)
                    
                    # Save as PNG file with consistent naming
                    filename = class_dir / f"{pattern_class}_{sample_idx:04d}.png"
                    Image.fromarray(image_uint8, mode='L').save(filename)
                    
                    # Store relative path for metadata
                    relative_filename = str(filename.relative_to(output_path))
                    all_filenames.append(relative_filename)
                    all_labels.append(class_idx)
                    
                except Exception as e:
                    logger.error(f"Error generating {pattern_class} sample {sample_idx}: {e}")
                    raise
        
        # Create dataset info in the same format as original
        dataset_info = {
            'filenames': all_filenames,
            'labels': all_labels,
            'label_names': self.pattern_classes,
            'params': {
                'image_size': image_size,
                'num_samples_per_class': num_samples_per_class,
                'num_layers_range': num_layers_range,
                'noise_level': noise_level,
                'random_seed': random_seed,
                'num_classes': len(self.pattern_classes),
                'total_samples': len(all_filenames)
            }
        }
        
        # Save dataset info as .npy file (same as original format)
        np.save(output_path / 'dataset_info.npy', dataset_info)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Generated {len(all_filenames)} total samples across {len(self.pattern_classes)} classes")
        
        # Store dataset info internally
        self._dataset_info = dataset_info
        self._is_generated = True
        
        return dataset_info
    
    @property
    def dataset_info(self) -> Optional[Dict[str, Any]]:
        """Get the dataset information."""
        return self._dataset_info
    
    def _generate_horizontal_layers(self, size: int, num_layers_range: Tuple[int, int], noise_level: float) -> np.ndarray:
        """Generate horizontal layered patterns."""
        image = np.zeros((size, size))
        num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
        
        # Create layer boundaries
        layer_boundaries = np.sort(np.random.uniform(0, size, num_layers - 1))
        layer_boundaries = np.concatenate([[0], layer_boundaries, [size]])
        
        # Fill layers with different intensities
        for i in range(num_layers):
            start_y = int(layer_boundaries[i])
            end_y = int(layer_boundaries[i + 1])
            intensity = np.random.uniform(0.2, 1.0)
            image[start_y:end_y, :] = intensity
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_folded_layers(self, size: int, num_layers_range: Tuple[int, int], noise_level: float) -> np.ndarray:
        """Generate folded layer patterns using mathematical functions."""
        image = np.zeros((size, size))
        num_layers = np.random.randint(num_layers_range[0], num_layers_range[1] + 1)
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # Create folding parameters
        fold_amplitude = np.random.uniform(size * 0.05, size * 0.15)
        fold_frequency = np.random.uniform(0.5, 1.5)
        fold_phase = np.random.uniform(0, 2 * np.pi)
        
        # Generate folded surface
        folded_surface = y + fold_amplitude * np.sin(fold_frequency * x / size * 2 * np.pi + fold_phase)
        
        # Create layers
        layer_thickness = size / num_layers
        
        for layer_idx in range(num_layers):
            # Define layer boundaries
            layer_bottom = layer_idx * layer_thickness
            layer_top = (layer_idx + 1) * layer_thickness
            
            # Create mask for this layer
            layer_mask = (folded_surface >= layer_bottom) & (folded_surface < layer_top)
            
            # Assign intensity to this layer
            intensity = np.random.uniform(0.2, 1.0)
            image[layer_mask] = intensity
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_faulted_layers(self, size: int, num_layers_range: Tuple[int, int], noise_level: float) -> np.ndarray:
        """Generate faulted layer patterns."""
        # Start with horizontal layers
        image = self._generate_horizontal_layers(size, num_layers_range, 0)
        
        # Add fault
        fault_position = np.random.uniform(size * 0.3, size * 0.7)
        fault_offset = np.random.uniform(-size * 0.2, size * 0.2)
        fault_width = np.random.randint(1, 3)
        
        # Apply fault displacement
        fault_col = int(fault_position)
        
        # Create fault zone
        for col in range(max(0, fault_col - fault_width), min(size, fault_col + fault_width)):
            image[:, col] = np.random.uniform(0.1, 0.3)  # Fault zone material
        
        # Displace one side with proper bounds checking
        if fault_col < size:
            displaced_section = image[:, fault_col:].copy()
            image[:, fault_col:] = 0
            
            offset_pixels = int(abs(fault_offset))
            
            if fault_offset > 0:
                # Move right side down
                if offset_pixels < size and displaced_section.shape[0] > offset_pixels:
                    target_rows = min(size - offset_pixels, displaced_section.shape[0] - offset_pixels)
                    if target_rows > 0:
                        image[offset_pixels:offset_pixels + target_rows, fault_col:] = displaced_section[:target_rows, :]
            else:
                # Move right side up
                if offset_pixels < size and displaced_section.shape[0] > offset_pixels:
                    target_rows = min(size - offset_pixels, displaced_section.shape[0] - offset_pixels)
                    if target_rows > 0:
                        image[:target_rows, fault_col:] = displaced_section[offset_pixels:offset_pixels + target_rows, :]
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_intrusion_patterns(self, size: int, num_layers_range: Tuple[int, int], noise_level: float) -> np.ndarray:
        """Generate intrusion patterns."""
        # Start with horizontal layers
        image = self._generate_horizontal_layers(size, num_layers_range, 0)
        
        # Add intrusion
        intrusion_type = np.random.choice(['sill', 'dike', 'pluton'])
        intrusion_intensity = np.random.uniform(0.7, 1.0)
        
        # Calculate safe thickness range (ensure minimum thickness of 1)
        max_thickness = max(2, size // 8)  # Ensure at least 2 pixels thick
        min_thickness = 1
        
        if intrusion_type == 'sill':
            # Horizontal intrusion
            intrusion_y = np.random.randint(size // 4, 3 * size // 4)
            intrusion_thickness = np.random.randint(min_thickness, max_thickness)
            end_y = min(intrusion_y + intrusion_thickness, size)
            image[intrusion_y:end_y, :] = intrusion_intensity
            
        elif intrusion_type == 'dike':
            # Vertical intrusion
            intrusion_x = np.random.randint(size // 4, 3 * size // 4)
            intrusion_thickness = np.random.randint(min_thickness, max_thickness)
            end_x = min(intrusion_x + intrusion_thickness, size)
            image[:, intrusion_x:end_x] = intrusion_intensity
            
        elif intrusion_type == 'pluton':
            # Circular intrusion
            center_x = np.random.randint(size // 4, 3 * size // 4)
            center_y = np.random.randint(size // 4, 3 * size // 4)
            max_radius = max(1, size // 8)  # Ensure at least 1 pixel radius
            radius = np.random.randint(1, max_radius + 1)
            
            y, x = np.ogrid[:size, :size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            image[mask] = intrusion_intensity
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def _generate_unconformity_patterns(self, size: int, num_layers_range: Tuple[int, int], noise_level: float) -> np.ndarray:
        """Generate unconformity patterns."""
        image = np.zeros((size, size))
        
        # Generate lower sequence
        lower_layers = np.random.randint(2, num_layers_range[1] // 2 + 1)
        lower_image = self._generate_horizontal_layers(size, (lower_layers, lower_layers), 0)
        
        # Create unconformity surface
        unconformity_level = np.random.uniform(size * 0.4, size * 0.7)
        surface_variation = np.random.uniform(size * 0.05, size * 0.15)
        
        x = np.arange(size)
        surface_y = unconformity_level + surface_variation * np.sin(2 * np.pi * x / size * np.random.uniform(1, 3))
        
        # Apply lower sequence below unconformity
        for col in range(size):
            surface_row = int(np.clip(surface_y[col], 0, size - 1))
            image[surface_row:, col] = lower_image[surface_row:, col]
        
        # Generate upper sequence above unconformity
        upper_layers = np.random.randint(2, num_layers_range[1] // 2 + 1)
        
        for layer_idx in range(upper_layers):
            layer_thickness = np.random.uniform(size * 0.05, size * 0.15)
            intensity = np.random.uniform(0.2, 1.0)
            
            for col in range(size):
                surface_row = int(np.clip(surface_y[col], 0, size - 1))
                layer_top = max(0, surface_row - int((layer_idx + 1) * layer_thickness))
                layer_bottom = max(0, surface_row - int(layer_idx * layer_thickness))
                
                image[layer_top:layer_bottom, col] = intensity
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
    
    def save(self, dataset_info: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save dataset to disk using the original format.
        
        Args:
            dataset_info: Dataset information dictionary
            output_path: Path where to save the dataset
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save dataset info as .npy file (same as original format)
        np.save(output_path / 'dataset_info.npy', dataset_info)
        
        logger.info(f"Dataset metadata saved to {output_path / 'dataset_info.npy'}")
    
    def load(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an existing dataset from disk.
        
        Args:
            dataset_path: Path to the dataset directory or dataset_info.npy file
            
        Returns:
            Dictionary containing dataset information and metadata
        """
        dataset_path = Path(dataset_path)
        
        # Handle both directory path and direct file path
        if dataset_path.is_file() and dataset_path.name == 'dataset_info.npy':
            dataset_info_path = dataset_path
        else:
            dataset_info_path = dataset_path / 'dataset_info.npy'
        
        if not dataset_info_path.exists():
            raise FileNotFoundError(f"Dataset info file does not exist: {dataset_info_path}")
        
        # Load the dataset info
        dataset_info = np.load(dataset_info_path, allow_pickle=True).item()
        
        # Validate the loaded data structure
        required_keys = ['filenames', 'labels', 'label_names', 'params']
        for key in required_keys:
            if key not in dataset_info:
                raise ValueError(f"Invalid dataset format: missing key '{key}'")
        
        self._dataset_info = dataset_info
        self._is_generated = True
        
        logger.info(f"Loaded dataset with {len(dataset_info['filenames'])} samples from {dataset_info_path}")
        
        return dataset_info 
"""
Geological dataset generation for autoencoder experiments.

This module provides functionality to generate synthetic geological patterns
including layered structures, fault systems, and other geological features.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional, Union, List
from pathlib import Path
import json
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
                 random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate layered geological patterns dataset.
        
        Args:
            output_dir: Directory to save the dataset
            num_samples_per_class: Number of samples per geological pattern class
            image_size: Size of generated images (image_size x image_size)
            num_layers_range: Range for number of layers in each pattern
            noise_level: Amount of noise to add (0.0 to 1.0)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing generated dataset information
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(f"Generating {len(self.pattern_classes)} classes of geological patterns...")
        
        all_images = []
        all_labels = []
        
        for class_idx, pattern_class in enumerate(self.pattern_classes):
            logger.info(f"  Generating {pattern_class}...")
            
            class_images = []
            for sample_idx in range(num_samples_per_class):
                try:
                    if pattern_class == 'horizontal_layers':
                        print(f"    Generating horizontal layers sample {sample_idx + 1}")
                        image = self._generate_horizontal_layers(image_size, num_layers_range, noise_level)
                        print(f"    Successfully generated horizontal layers sample {sample_idx + 1}")
                    elif pattern_class == 'folded_layers':
                        print(f"    Generating folded layer sample {sample_idx + 1}/{num_samples_per_class}")
                        image = self._generate_folded_layers(image_size, num_layers_range, noise_level)
                        print(f"    Successfully generated folded layer sample {sample_idx + 1}")
                    elif pattern_class == 'faulted_layers':
                        print(f"    Generating faulted layers sample {sample_idx + 1}")
                        image = self._generate_faulted_layers(image_size, num_layers_range, noise_level)
                        print(f"    Successfully generated faulted layers sample {sample_idx + 1}")
                    elif pattern_class == 'intrusion_patterns':
                        print(f"    Generating intrusion patterns sample {sample_idx + 1}")
                        image = self._generate_intrusion_patterns(image_size, num_layers_range, noise_level)
                        print(f"    Successfully generated intrusion patterns sample {sample_idx + 1}")
                    elif pattern_class == 'unconformity_patterns':
                        print(f"    Generating unconformity patterns sample {sample_idx + 1}")
                        image = self._generate_unconformity_patterns(image_size, num_layers_range, noise_level)
                        print(f"    Successfully generated unconformity patterns sample {sample_idx + 1}")
                    else:
                        raise ValueError(f"Unknown pattern class: {pattern_class}")
                    
                    print(f"    Generated image shape: {image.shape}")
                    class_images.append(image)
                    
                except Exception as e:
                    print(f"    Error generating {pattern_class} sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Convert to numpy array
            print(f"  Converting {len(class_images)} images to numpy array...")
            try:
                class_images_array = np.array(class_images)
                print(f"  Class images array shape: {class_images_array.shape}")
                all_images.extend(class_images)
                all_labels.extend([class_idx] * num_samples_per_class)
            except Exception as e:
                print(f"  Error converting to numpy array: {e}")
                print(f"  Individual image shapes: {[img.shape for img in class_images[:3]]}")
                raise
        
        # Convert to numpy arrays
        images = np.array(all_images)
        labels = np.array(all_labels)
        
        # Create dataset info
        dataset_info = {
            'images': images,
            'labels': labels,
            'class_names': self.pattern_classes,
            'metadata': {
                'num_classes': len(self.pattern_classes),
                'image_size': image_size,
                'num_samples_per_class': num_samples_per_class,
                'total_samples': len(images),
                'generation_params': {
                    'num_layers_range': num_layers_range,
                    'noise_level': noise_level,
                    'random_seed': random_seed
                }
            }
        }
        
        # Save dataset
        if output_dir:
            self.save(dataset_info, output_dir)
            logger.info(f"Dataset saved to {output_dir}")
        
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
        Save dataset to disk.
        
        Args:
            dataset_info: Dataset information dictionary
            output_path: Path where to save the dataset
        """
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save dataset arrays
        np.save(output_path / 'dataset_info.npy', dataset_info)
        np.save(output_path / 'images.npy', dataset_info['images'])
        np.save(output_path / 'labels.npy', dataset_info['labels'])
        
        # Save metadata as JSON
        metadata = {k: v for k, v in dataset_info.items() 
                   if k not in ['images', 'labels']}
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy_types(obj)
        
        metadata = recursive_convert(metadata)
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
    
    def load(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an existing dataset from disk.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary containing dataset information and metadata
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Try to load the complete dataset_info.npy first
        dataset_info_path = dataset_path / 'dataset_info.npy'
        if dataset_info_path.exists():
            dataset_info = np.load(dataset_info_path, allow_pickle=True).item()
        else:
            # Load components separately
            images = np.load(dataset_path / 'images.npy')
            labels = np.load(dataset_path / 'labels.npy')
            
            # Load metadata
            with open(dataset_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            dataset_info = {
                'images': images,
                'labels': labels,
                **metadata
            }
        
        self._dataset_info = dataset_info
        self._is_generated = True
        
        return dataset_info 
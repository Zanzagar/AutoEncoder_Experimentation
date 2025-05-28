"""
Layered Geological Dataset Implementation

This module implements the exact dataset generation process from the original
AutoEncoderJupyterTest.ipynb notebook, maintaining compatibility with the
existing data format and file structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_layered_dataset(
    output_dir: str = 'layered_geologic_patterns_dataset',
    num_samples_per_class: int = 500,
    image_size: int = 64,
    force_regenerate: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Generate a dataset of synthetic images with two distinct layered pattern classes:
    1. Consistent layers with moderate orientation and thickness
    2. Highly variable layers with perpendicular orientation and extreme thickness variations
    
    This function replicates the exact behavior from the original AutoEncoderJupyterTest.ipynb
    notebook to ensure compatibility with existing workflows.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the generated dataset
    num_samples_per_class : int
        Number of samples to generate per class
    image_size : int
        Size of the square images (width and height)
    force_regenerate : bool
        If True, regenerate even if dataset already exists
        
    Returns:
    --------
    dict or None
        Dictionary containing dataset information with keys:
        - 'filenames': List of image file paths
        - 'labels': List of corresponding class labels (0 or 1)
        - 'label_names': List of class names ['consistent_layers', 'variable_layers']
        - 'params': Dictionary of generation parameters
    """
    # Check if dataset already exists
    info_path = os.path.join(output_dir, 'dataset_info.npy')
    if os.path.exists(output_dir) and os.path.exists(info_path) and not force_regenerate:
        logger.info(f"Loading existing dataset from {info_path}")
        try:
            dataset_info = np.load(info_path, allow_pickle=True).item()
            logger.info(f"Successfully loaded dataset with {len(dataset_info['filenames'])} images.")
            return dataset_info
        except Exception as e:
            logger.error(f"Error loading existing dataset: {e}")
            if not force_regenerate:
                logger.warning("Set force_regenerate=True to create a new dataset.")
                return None

    # Create output directory structure
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return None

    # Define pattern classes and create their directories
    pattern_classes = ['consistent_layers', 'variable_layers']
    for pattern_type in pattern_classes:
        try:
            os.makedirs(os.path.join(output_dir, pattern_type), exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory for {pattern_type}: {e}")
            return None

    # Dictionary to store file paths and labels
    dataset_info = {
        'filenames': [],
        'labels': [],
        'label_names': pattern_classes,
        'params': {
            'image_size': image_size,
            'num_samples_per_class': num_samples_per_class
        }
    }

    # Generate images for each pattern class
    for class_idx, pattern_type in enumerate(pattern_classes):
        class_dir = os.path.join(output_dir, pattern_type)
        logger.info(f"Generating {num_samples_per_class} '{pattern_type}' images...")

        for i in tqdm(range(num_samples_per_class), desc=f"Generating {pattern_type}"):
            # Create a blank white image
            img = Image.new('L', (image_size, image_size), color=255)
            draw = ImageDraw.Draw(img)

            if pattern_type == 'consistent_layers':
                # Consistent layers: moderate orientation and thickness
                angle_deg = random.uniform(30, 60)
                num_layers = random.randint(5, 8)
                base_thickness = random.randint(max(1, image_size // 15), max(3, image_size // 10))
            else:  # 'variable_layers'
                # Highly variable layers: perpendicular orientation
                angle_deg = random.uniform(120, 150)
                num_layers = random.randint(3, 15)
                base_thickness = random.randint(1, max(5, image_size // 5))

            angle_rad = np.radians(angle_deg)
            current_pos = 0
            
            while current_pos < image_size * 1.5:  # Ensure coverage even for diagonal
                # Slight thickness variation within the same sample
                thickness = max(1, base_thickness + random.randint(-base_thickness//3, base_thickness//3))
                
                # Calculate line coords based on angle and current position offset
                offset_x = current_pos * np.sin(angle_rad)
                offset_y = -current_pos * np.cos(angle_rad)
                
                # Line endpoints extended beyond image boundaries
                x0 = offset_x - image_size * np.cos(angle_rad) * 2
                y0 = offset_y - image_size * np.sin(angle_rad) * 2
                x1 = offset_x + image_size * np.cos(angle_rad) * 2
                y1 = offset_y + image_size * np.sin(angle_rad) * 2
                
                # Draw layer: black lines on white background
                draw.line([(x0, y0), (x1, y1)], fill=0, width=thickness)
                
                # Move to next layer position
                gap_multiplier = 1.5 if pattern_type == 'variable_layers' else 1.0
                gap = random.randint(max(1, thickness // 2), int(thickness * gap_multiplier))
                current_pos += thickness + gap

            # Save the image
            filename = os.path.join(class_dir, f"{pattern_type}_{i:05d}.png")
            try:
                img.save(filename)
                dataset_info['filenames'].append(filename)
                dataset_info['labels'].append(class_idx)
            except Exception as e:
                logger.error(f"Error saving image {filename}: {e}")
                continue

    # Save dataset info
    try:
        np.save(info_path, dataset_info)
        logger.info(f"\nLayered pattern dataset generation complete.")
        logger.info(f"Total images generated: {len(dataset_info['filenames'])}")
        logger.info(f"Images saved to {output_dir}")
        logger.info(f"Dataset info saved to {info_path}")
    except Exception as e:
        logger.error(f"Error saving dataset info: {e}")

    return dataset_info


def visualize_dataset_samples(
    dataset_info: Dict[str, Any], 
    samples_per_class: int = 5, 
    figure_size: Tuple[int, int] = (15, 8)
) -> None:
    """
    Visualize random samples from each class in the dataset.
    
    This function replicates the exact visualization from the original notebook.
    
    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset information
    samples_per_class : int
        Number of samples to display per class
    figure_size : tuple
        Size of the figure to display
    """
    if not dataset_info.get('filenames'):
        logger.error("Error: No filenames found in dataset_info.")
        return

    unique_labels = sorted(list(set(dataset_info.get('labels', []))))
    label_names = dataset_info.get('label_names', [f"Class {i}" for i in unique_labels])
    
    num_classes = len(unique_labels)
    plt.figure(figsize=figure_size)
    
    # For each class
    for i, label in enumerate(unique_labels):
        # Get indices of this class
        indices = [j for j, l in enumerate(dataset_info['labels']) if l == label]
        
        # Select random samples
        sample_indices = random.sample(indices, min(samples_per_class, len(indices)))
        
        # Plot each sample
        for j, idx in enumerate(sample_indices):
            img_path = dataset_info['filenames'][idx]
            img = Image.open(img_path)
            
            ax = plt.subplot(num_classes, samples_per_class, i * samples_per_class + j + 1)
            ax.imshow(img, cmap='gray')
            
            # Add class label only to the first image of each row
            if j == 0:
                ax.set_ylabel(label_names[label], fontsize=12, rotation=0, labelpad=40, ha='right', va='center')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

    plt.suptitle("Synthetic Layered Geologic Pattern Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def load_layered_dataset(dataset_path: str) -> Optional[Dict[str, Any]]:
    """
    Load an existing layered geological dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset directory or dataset_info.npy file
        
    Returns:
    --------
    dict or None
        Dictionary containing dataset information
    """
    # Handle both directory path and direct file path
    if dataset_path.endswith('.npy'):
        info_path = dataset_path
    else:
        info_path = os.path.join(dataset_path, 'dataset_info.npy')
    
    if not os.path.exists(info_path):
        logger.error(f"Dataset info file not found: {info_path}")
        return None
        
    try:
        dataset_info = np.load(info_path, allow_pickle=True).item()
        logger.info(f"Successfully loaded dataset with {len(dataset_info['filenames'])} images.")
        return dataset_info
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None


def get_dataset_statistics(dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for the layered geological dataset.
    
    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset information
        
    Returns:
    --------
    dict
        Dictionary containing dataset statistics
    """
    if not dataset_info or not dataset_info.get('filenames'):
        return {}
    
    stats = {
        'total_samples': len(dataset_info['filenames']),
        'num_classes': len(set(dataset_info['labels'])),
        'class_names': dataset_info.get('label_names', []),
        'class_distribution': {},
        'image_size': dataset_info.get('params', {}).get('image_size', 'Unknown'),
        'samples_per_class': dataset_info.get('params', {}).get('num_samples_per_class', 'Unknown')
    }
    
    # Calculate class distribution
    for label in dataset_info['labels']:
        class_name = dataset_info['label_names'][label] if label < len(dataset_info['label_names']) else f"Class {label}"
        stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
    
    return stats 
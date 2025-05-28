"""
Dataset Visualization Functions

This module contains functions for visualizing datasets, including:
- Sample grids showing examples from each class
- Class distribution plots
- Dataset statistics and summaries
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from typing import Dict, List, Tuple, Optional, Union


def visualize_dataset_samples(
    dataset_info: Dict,
    samples_per_class: int = 5,
    figure_size: Tuple[int, int] = (15, 8),
    random_seed: Optional[int] = None
) -> None:
    """
    Visualize random samples from each class in the dataset.
    
    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset information with keys:
        - 'filenames': List of image file paths OR 'images': numpy array of images
        - 'labels': List of corresponding labels
        - 'label_names' or 'class_names': List of class names (optional)
    samples_per_class : int, default=5
        Number of samples to display per class
    figure_size : tuple, default=(15, 8)
        Size of the figure to display (width, height)
    random_seed : int, optional
        Random seed for reproducible sample selection
        
    Returns:
    --------
    None
        Displays the visualization using matplotlib
    """
    # Check if we have images or filenames
    has_images = 'images' in dataset_info
    has_filenames = 'filenames' in dataset_info
    
    if not has_images and not has_filenames:
        print("Error: No 'images' or 'filenames' found in dataset_info.")
        return

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    unique_labels = sorted(list(set(dataset_info.get('labels', []))))
    
    # Handle both 'label_names' and 'class_names'
    label_names = dataset_info.get('label_names', 
                                  dataset_info.get('class_names', 
                                  [f"Class {i}" for i in unique_labels]))
    
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
            if has_images:
                # Use direct image data
                img = dataset_info['images'][idx]
                if len(img.shape) > 2:
                    img = img.squeeze()  # Remove extra dimensions
            else:
                # Load from file path
                img_path = dataset_info['filenames'][idx]
                
                # Handle both absolute and relative paths
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    continue
                    
                try:
                    img = Image.open(img_path)
                    img = np.array(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
                
            ax = plt.subplot(num_classes, samples_per_class, i * samples_per_class + j + 1)
            ax.imshow(img, cmap='gray')
            
            # Add class label only to the first image of each row
            if j == 0:
                ax.set_ylabel(label_names[label], fontsize=12, rotation=0, 
                            labelpad=40, ha='right', va='center')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

    plt.suptitle("Dataset Sample Visualization", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


def plot_class_distribution(
    dataset_info: Dict,
    figure_size: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the distribution of samples across classes.
    
    Parameters:
    -----------
    dataset_info : dict
        Dictionary containing dataset information
    figure_size : tuple, default=(10, 6)
        Size of the figure to display
    """
    if not dataset_info.get('labels'):
        print("Error: No labels found in dataset_info.")
        return
        
    labels = dataset_info['labels']
    
    # Handle both 'label_names' and 'class_names'
    label_names = dataset_info.get('label_names', 
                                  dataset_info.get('class_names', 
                                  [f"Class {i}" for i in sorted(set(labels))]))
    
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figure_size)
    bars = plt.bar(range(len(unique_labels)), counts, alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.xticks(range(len(unique_labels)), [label_names[i] for i in unique_labels], rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Dataset Statistics:")
    print(f"Total samples: {len(labels)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Samples per class: {dict(zip([label_names[i] for i in unique_labels], counts))}")


def show_sample_grid(
    images: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[int]] = None,
    label_names: Optional[List[str]] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    figure_size: Tuple[int, int] = (12, 8),
    title: str = "Sample Grid"
) -> None:
    """
    Display a grid of images with optional labels.
    
    Parameters:
    -----------
    images : array-like or list
        Images to display. Can be numpy array of shape (N, H, W) or list of arrays
    labels : list, optional
        Labels for each image
    label_names : list, optional
        Names corresponding to label indices
    grid_size : tuple, optional
        (rows, cols) for the grid. If None, automatically determined
    figure_size : tuple, default=(12, 8)
        Size of the figure
    title : str, default="Sample Grid"
        Title for the plot
    """
    if isinstance(images, list):
        images = np.array(images)
    
    n_images = len(images)
    
    # Determine grid size
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    plt.figure(figsize=figure_size)
    
    for i in range(min(n_images, rows * cols)):
        plt.subplot(rows, cols, i + 1)
        
        # Handle different image formats
        if len(images[i].shape) == 3 and images[i].shape[2] == 1:
            # Remove single channel dimension
            img = images[i].squeeze()
        else:
            img = images[i]
            
        plt.imshow(img, cmap='gray')
        
        # Add label if provided
        if labels is not None and i < len(labels):
            label = labels[i]
            if label_names is not None and label < len(label_names):
                label_text = label_names[label]
            else:
                label_text = str(label)
            plt.title(label_text, fontsize=10)
        
        plt.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_datasets(
    dataset_info_list: List[Dict],
    dataset_names: List[str],
    samples_per_dataset: int = 3
) -> None:
    """
    Compare multiple datasets side by side.
    
    Parameters:
    -----------
    dataset_info_list : list of dict
        List of dataset info dictionaries
    dataset_names : list of str
        Names for each dataset
    samples_per_dataset : int, default=3
        Number of samples to show per dataset
    """
    n_datasets = len(dataset_info_list)
    
    plt.figure(figsize=(15, 4 * n_datasets))
    
    for i, (dataset_info, name) in enumerate(zip(dataset_info_list, dataset_names)):
        if not dataset_info.get('filenames'):
            continue
            
        # Get random samples
        n_samples = min(samples_per_dataset, len(dataset_info['filenames']))
        sample_indices = random.sample(range(len(dataset_info['filenames'])), n_samples)
        
        for j, idx in enumerate(sample_indices):
            plt.subplot(n_datasets, samples_per_dataset, i * samples_per_dataset + j + 1)
            
            img_path = dataset_info['filenames'][idx]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                plt.imshow(img, cmap='gray')
            
            if j == 0:
                plt.ylabel(name, fontsize=12, rotation=0, labelpad=40, ha='right', va='center')
            
            plt.axis('off')
    
    plt.suptitle("Dataset Comparison", fontsize=16)
    plt.tight_layout()
    plt.show() 
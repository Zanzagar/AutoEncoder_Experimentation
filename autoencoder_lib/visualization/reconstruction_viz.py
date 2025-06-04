"""
Reconstruction Visualization Functions

Functions for visualizing autoencoder reconstructions and comparing them with original images.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional, Union
from PIL import Image


def visualize_reconstructions(
    originals: Union[torch.Tensor, np.ndarray],
    reconstructions: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 8,
    figure_size: Tuple[int, int] = (16, 8),
    title: str = "Original vs Reconstructed Images"
) -> None:
    """
    Visualize original images alongside their reconstructions.
    
    Args:
        originals: Original images (N, C, H, W) or (N, H, W)
        reconstructions: Reconstructed images (same shape as originals)
        labels: Labels for the images (optional)
        class_names: Names for each class (optional)
        num_samples: Number of samples to display
        figure_size: Size of the figure
        title: Title for the plot
    """
    # Convert to numpy if tensor
    if isinstance(originals, torch.Tensor):
        originals = originals.detach().cpu().numpy()
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Handle different image dimensions
    if len(originals.shape) == 4 and originals.shape[1] == 1:
        # (N, 1, H, W) -> (N, H, W)
        originals = originals.squeeze(1)
        reconstructions = reconstructions.squeeze(1)
    elif len(originals.shape) == 4:
        # (N, C, H, W) -> (N, H, W, C) for RGB
        originals = np.transpose(originals, (0, 2, 3, 1))
        reconstructions = np.transpose(reconstructions, (0, 2, 3, 1))
    
    # Select samples to display
    num_samples = min(num_samples, len(originals))
    indices = np.linspace(0, len(originals) - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(2, num_samples, figsize=figure_size)
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(indices):
        # Original image
        ax_orig = axes[0, i]
        if len(originals[idx].shape) == 3:
            ax_orig.imshow(originals[idx])
        else:
            ax_orig.imshow(originals[idx], cmap='gray')
        ax_orig.set_title('Original', fontsize=10)
        ax_orig.axis('off')
        
        # Add label if available
        if labels is not None:
            label = labels[idx]
            if class_names is not None and label < len(class_names):
                label_text = class_names[label]
            else:
                label_text = f"Class {label}"
            ax_orig.text(0.5, -0.1, label_text, transform=ax_orig.transAxes,
                        ha='center', va='top', fontsize=8)
        
        # Reconstructed image
        ax_recon = axes[1, i]
        if len(reconstructions[idx].shape) == 3:
            ax_recon.imshow(reconstructions[idx])
        else:
            ax_recon.imshow(reconstructions[idx], cmap='gray')
        ax_recon.set_title('Reconstructed', fontsize=10)
        ax_recon.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_loss_grid(
    originals: Union[torch.Tensor, np.ndarray],
    reconstructions: Union[torch.Tensor, np.ndarray],
    losses: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 12,
    figure_size: Tuple[int, int] = (16, 10)
) -> None:
    """
    Visualize reconstructions with their individual loss values.
    
    Args:
        originals: Original images
        reconstructions: Reconstructed images
        losses: Reconstruction losses for each image
        labels: Labels for the images (optional)
        class_names: Names for each class (optional)
        num_samples: Number of samples to display
        figure_size: Size of the figure
    """
    # Convert to numpy if tensor
    if isinstance(originals, torch.Tensor):
        originals = originals.detach().cpu().numpy()
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Handle different image dimensions
    if len(originals.shape) == 4 and originals.shape[1] == 1:
        originals = originals.squeeze(1)
        reconstructions = reconstructions.squeeze(1)
    elif len(originals.shape) == 4:
        originals = np.transpose(originals, (0, 2, 3, 1))
        reconstructions = np.transpose(reconstructions, (0, 2, 3, 1))
    
    # Sort by loss (highest to lowest) and select samples
    loss_indices = np.argsort(losses)[::-1]
    num_samples = min(num_samples, len(originals))
    selected_indices = loss_indices[:num_samples]
    
    cols = min(6, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=figure_size)
    if rows == 1 and cols == 1:
        axes = axes.reshape(2, 1)
    elif rows == 1:
        axes = axes.reshape(2, cols)
    elif cols == 1:
        axes = axes.reshape(rows * 2, 1)
    
    for i, idx in enumerate(selected_indices):
        row = (i // cols) * 2
        col = i % cols
        
        # Original image
        ax_orig = axes[row, col]
        if len(originals[idx].shape) == 3:
            ax_orig.imshow(originals[idx])
        else:
            ax_orig.imshow(originals[idx], cmap='gray')
        
        # Add label and loss
        title = f"Loss: {losses[idx]:.4f}"
        if labels is not None:
            label = labels[idx]
            if class_names is not None and label < len(class_names):
                title = f"{class_names[label]}\n{title}"
            else:
                title = f"Class {label}\n{title}"
        
        ax_orig.set_title(title, fontsize=8)
        ax_orig.axis('off')
        
        # Reconstructed image
        ax_recon = axes[row + 1, col]
        if len(reconstructions[idx].shape) == 3:
            ax_recon.imshow(reconstructions[idx])
        else:
            ax_recon.imshow(reconstructions[idx], cmap='gray')
        ax_recon.set_title('Reconstructed', fontsize=8)
        ax_recon.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = (i // cols) * 2
        col = i % cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.suptitle('Reconstruction Quality (Sorted by Loss)', fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_reconstruction_quality(
    originals: Union[torch.Tensor, np.ndarray],
    reconstructions_list: List[Union[torch.Tensor, np.ndarray]],
    model_names: List[str],
    sample_idx: int = 0,
    figure_size: Tuple[int, int] = (15, 5)
) -> None:
    """
    Compare reconstruction quality across different models for the same input.
    
    Args:
        originals: Original images
        reconstructions_list: List of reconstructions from different models
        model_names: Names of the models
        sample_idx: Index of the sample to compare
        figure_size: Size of the figure
    """
    # Convert to numpy if tensor
    if isinstance(originals, torch.Tensor):
        originals = originals.detach().cpu().numpy()
    
    reconstructions_np = []
    for recons in reconstructions_list:
        if isinstance(recons, torch.Tensor):
            recons = recons.detach().cpu().numpy()
        reconstructions_np.append(recons)
    
    # Handle different image dimensions
    if len(originals.shape) == 4 and originals.shape[1] == 1:
        originals = originals.squeeze(1)
        for i in range(len(reconstructions_np)):
            reconstructions_np[i] = reconstructions_np[i].squeeze(1)
    elif len(originals.shape) == 4:
        originals = np.transpose(originals, (0, 2, 3, 1))
        for i in range(len(reconstructions_np)):
            reconstructions_np[i] = np.transpose(reconstructions_np[i], (0, 2, 3, 1))
    
    num_models = len(reconstructions_list)
    fig, axes = plt.subplots(1, num_models + 1, figsize=figure_size)
    
    # Show original
    if len(originals[sample_idx].shape) == 3:
        axes[0].imshow(originals[sample_idx])
    else:
        axes[0].imshow(originals[sample_idx], cmap='gray')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # Show reconstructions from each model
    for i, (recons, name) in enumerate(zip(reconstructions_np, model_names)):
        if len(recons[sample_idx].shape) == 3:
            axes[i + 1].imshow(recons[sample_idx])
        else:
            axes[i + 1].imshow(recons[sample_idx], cmap='gray')
        axes[i + 1].set_title(name, fontsize=12)
        axes[i + 1].axis('off')
    
    plt.suptitle(f'Reconstruction Comparison - Sample {sample_idx}', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_error_heatmap(
    originals: Union[torch.Tensor, np.ndarray],
    reconstructions: Union[torch.Tensor, np.ndarray],
    figure_size: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot a heatmap showing pixel-wise reconstruction errors.
    
    Args:
        originals: Original images (showing first image only)
        reconstructions: Reconstructed images (showing first image only)
        figure_size: Size of the figure
    """
    # Convert to numpy if tensor
    if isinstance(originals, torch.Tensor):
        originals = originals.detach().cpu().numpy()
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()
    
    # Handle different image dimensions and take first image
    if len(originals.shape) == 4 and originals.shape[1] == 1:
        original = originals[0].squeeze(0)
        reconstruction = reconstructions[0].squeeze(0)
    elif len(originals.shape) == 4:
        original = np.transpose(originals[0], (1, 2, 0))
        reconstruction = np.transpose(reconstructions[0], (1, 2, 0))
    else:
        original = originals[0]
        reconstruction = reconstructions[0]
    
    # Calculate absolute error
    error = np.abs(original - reconstruction)
    
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    
    # Original
    if len(original.shape) == 3:
        axes[0].imshow(original)
    else:
        im0 = axes[0].imshow(original, cmap='gray')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstruction
    if len(reconstruction.shape) == 3:
        axes[1].imshow(reconstruction)
    else:
        im1 = axes[1].imshow(reconstruction, cmap='gray')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Error heatmap
    if len(error.shape) == 3:
        error_gray = np.mean(error, axis=2)
    else:
        error_gray = error
    
    im2 = axes[2].imshow(error_gray, cmap='hot')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title('Absolute Error')
    axes[2].axis('off')
    
    plt.suptitle('Reconstruction Error Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()


def animate_training_reconstructions(
    reconstruction_history: List[Union[torch.Tensor, np.ndarray]],
    original: Union[torch.Tensor, np.ndarray],
    epochs: List[int],
    figure_size: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create an animation showing how reconstructions improve during training.
    
    Args:
        reconstruction_history: List of reconstructions at different epochs
        original: Original image
        epochs: Epoch numbers corresponding to each reconstruction
        figure_size: Size of the figure
        save_path: Path to save the animation (optional)
    """
    # Convert to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    
    reconstructions = []
    for recons in reconstruction_history:
        if isinstance(recons, torch.Tensor):
            recons = recons.detach().cpu().numpy()
        reconstructions.append(recons)
    
    # Handle different image dimensions
    if len(original.shape) == 3 and original.shape[0] == 1:
        original = original.squeeze(0)
        for i in range(len(reconstructions)):
            reconstructions[i] = reconstructions[i].squeeze(0)
    elif len(original.shape) == 3:
        original = np.transpose(original, (1, 2, 0))
        for i in range(len(reconstructions)):
            reconstructions[i] = np.transpose(reconstructions[i], (1, 2, 0))
    
    fig, axes = plt.subplots(1, len(reconstructions) + 1, figsize=figure_size)
    
    # Show original
    if len(original.shape) == 3:
        axes[0].imshow(original)
    else:
        axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show reconstructions at different epochs
    for i, (recons, epoch) in enumerate(zip(reconstructions, epochs)):
        if len(recons.shape) == 3:
            axes[i + 1].imshow(recons)
        else:
            axes[i + 1].imshow(recons, cmap='gray')
        axes[i + 1].set_title(f'Epoch {epoch}')
        axes[i + 1].axis('off')
    
    plt.suptitle('Reconstruction Progress During Training', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_dual_reconstructions(
    model,
    train_data: Union[torch.Tensor, np.ndarray],
    train_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    test_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
    test_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    n_samples: int = 5,
    device: str = 'cpu',
    view_indices: Optional[List[int]] = None,
    figure_size: Optional[Tuple[int, int]] = None
) -> None:
    """
    Visualize reconstructions from both training and test data side by side.
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels (optional)
        test_data: Test data tensor (optional, uses train_data if None)
        test_labels: Test labels (optional)
        class_names: List of class names for labels
        n_samples: Number of samples to show from each dataset
        device: Device to run model on
        view_indices: Specific indices to view (if None, uses first n_samples)
        figure_size: Size of figure (auto-calculated if None)
    """
    import torch
    
    model.eval()
    
    # Convert to tensors if needed
    if isinstance(train_data, np.ndarray):
        train_data = torch.FloatTensor(train_data)
    if test_data is None:
        test_data = train_data
    elif isinstance(test_data, np.ndarray):
        test_data = torch.FloatTensor(test_data)
    
    # Process view indices or use defaults
    if view_indices is not None:
        train_view_indices = view_indices[:len(view_indices)//2]
        test_view_indices = view_indices[len(view_indices)//2:]
        
        train_view_data = train_data[train_view_indices].to(device)
        train_view_labels = train_labels[train_view_indices] if train_labels is not None else None
        
        test_view_data = test_data[test_view_indices].to(device)
        test_view_labels = test_labels[test_view_indices] if test_labels is not None else None
    else:
        # Use first n_samples from each dataset
        n_train = min(n_samples, len(train_data))
        n_test = min(n_samples, len(test_data))
        
        train_view_data = train_data[:n_train].to(device)
        train_view_labels = train_labels[:n_train] if train_labels is not None else None
        
        test_view_data = test_data[:n_test].to(device)
        test_view_labels = test_labels[:n_test] if test_labels is not None else None
    
    # Generate reconstructions
    with torch.no_grad():
        # Handle different model architectures
        try:
            # Try standard autoencoder interface
            _, train_decoded = model(train_view_data)
            _, test_decoded = model(test_view_data)
        except:
            # Try direct decode method
            train_decoded = model.decode(model.encode(train_view_data))
            test_decoded = model.decode(model.encode(test_view_data))
    
    # Create figure with 4 rows: train original, train reconstructed, test original, test reconstructed
    n_train_images = len(train_view_data)
    n_test_images = len(test_view_data)
    max_images = max(n_train_images, n_test_images)
    
    # Create appropriate figure size
    if figure_size is None:
        fig_width = max_images * 2
        figure_size = (fig_width, 8)
    
    fig, axes = plt.subplots(4, max_images, figsize=figure_size)
    if max_images == 1:
        axes = axes.reshape(4, 1)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    # Add a title to the figure
    plt.suptitle('Train (top) vs. Test (bottom) Reconstructions', fontsize=14)
    
    # Train original images (row 0)
    for i in range(max_images):
        if i < n_train_images:
            # Handle different image formats
            img = train_view_data[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            
            axes[0][i].imshow(img.cpu().numpy(), cmap='gray')
            
            # If available, add class labels as text below images
            if class_names is not None and train_view_labels is not None:
                class_idx = train_view_labels[i].item()
                class_text = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                axes[0][i].text(img.shape[1]//2, img.shape[0] + 5, class_text, fontsize=8, ha="center")
        else:
            axes[0][i].axis('off')
        
        axes[0][i].set_xticks([])
        axes[0][i].set_yticks([])
        
        if i == 0:
            axes[0][i].set_ylabel("Train Original", fontsize=10)
    
    # Train reconstructed images (row 1)
    for i in range(max_images):
        if i < n_train_images:
            img = train_decoded[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            
            axes[1][i].imshow(img.cpu().numpy(), cmap='gray')
        else:
            axes[1][i].axis('off')
        
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])
        
        if i == 0:
            axes[1][i].set_ylabel("Train Recon", fontsize=10)
    
    # Test original images (row 2)
    for i in range(max_images):
        if i < n_test_images:
            img = test_view_data[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            
            axes[2][i].imshow(img.cpu().numpy(), cmap='gray')
            
            # If available, add class labels as text below images
            if class_names is not None and test_view_labels is not None:
                class_idx = test_view_labels[i].item()
                class_text = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
                axes[2][i].text(img.shape[1]//2, img.shape[0] + 5, class_text, fontsize=8, ha="center")
        else:
            axes[2][i].axis('off')
        
        axes[2][i].set_xticks([])
        axes[2][i].set_yticks([])
        
        if i == 0:
            axes[2][i].set_ylabel("Test Original", fontsize=10)
    
    # Test reconstructed images (row 3)
    for i in range(max_images):
        if i < n_test_images:
            img = test_decoded[i]
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            elif len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            
            axes[3][i].imshow(img.cpu().numpy(), cmap='gray')
        else:
            axes[3][i].axis('off')
        
        axes[3][i].set_xticks([])
        axes[3][i].set_yticks([])
        
        if i == 0:
            axes[3][i].set_ylabel("Test Recon", fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make space for the suptitle
    plt.show() 
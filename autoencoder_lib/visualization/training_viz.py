"""
Training Visualization Functions

Functions for visualizing training progress, loss curves, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd


def plot_training_curves(
    history: Dict[str, List],
    figure_size: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary containing 'train_loss' and optionally 'val_loss'
        figure_size: Size of the figure
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=figure_size)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate if available
    if 'learning_rate' in history:
        axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_title('Learning Rate Schedule', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        # Plot silhouette scores if available
        if 'train_silhouette' in history:
            axes[1].plot(epochs, history['train_silhouette'], 'b-', 
                        label='Training Silhouette', linewidth=2)
        if 'test_silhouette' in history:
            axes[1].plot(epochs, history['test_silhouette'], 'r-', 
                        label='Test Silhouette', linewidth=2)
        
        if 'train_silhouette' in history or 'test_silhouette' in history:
            axes[1].set_title('Silhouette Scores', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_landscape(
    train_losses: List[float],
    test_losses: Optional[List[float]] = None,
    epochs: Optional[List[int]] = None,
    figure_size: Tuple[int, int] = (12, 8),
    smoothing_window: int = 5
) -> None:
    """
    Plot detailed loss landscape with smoothing and statistics.
    
    Args:
        train_losses: Training loss values
        test_losses: Test loss values (optional)
        epochs: Epoch numbers (optional, defaults to sequential)
        figure_size: Size of the figure
        smoothing_window: Window size for smoothing
    """
    if epochs is None:
        epochs = list(range(1, len(train_losses) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Raw loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', alpha=0.7, label='Training Loss')
    if test_losses:
        axes[0, 0].plot(epochs, test_losses, 'r-', alpha=0.7, label='Test Loss')
    axes[0, 0].set_title('Raw Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Smoothed loss curves
    def smooth_curve(values, window):
        return pd.Series(values).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    train_smooth = smooth_curve(train_losses, smoothing_window)
    axes[0, 1].plot(epochs, train_smooth, 'b-', linewidth=2, label=f'Training (smoothed)')
    if test_losses:
        test_smooth = smooth_curve(test_losses, smoothing_window)
        axes[0, 1].plot(epochs, test_smooth, 'r-', linewidth=2, label=f'Test (smoothed)')
    
    axes[0, 1].set_title(f'Smoothed Loss Curves (window={smoothing_window})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss distribution
    axes[1, 0].hist(train_losses, bins=30, alpha=0.7, label='Training Loss', density=True)
    if test_losses:
        axes[1, 0].hist(test_losses, bins=30, alpha=0.7, label='Test Loss', density=True)
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss improvement rate
    train_improvements = np.diff(train_losses)
    axes[1, 1].plot(epochs[1:], train_improvements, 'b-', alpha=0.7, label='Training')
    if test_losses:
        test_improvements = np.diff(test_losses)
        axes[1, 1].plot(epochs[1:], test_improvements, 'r-', alpha=0.7, label='Test')
    
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Loss Change Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Change')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_performance_grid(
    results: Dict[str, Dict],
    metrics: List[str] = ['final_train_loss', 'final_test_loss', 'train_silhouette', 'test_silhouette'],
    figure_size: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a comprehensive performance grid showing multiple metrics.
    
    Args:
        results: Dictionary of results with structure {model_name: {metric: value}}
        metrics: List of metrics to display
        figure_size: Size of the figure
        save_path: Path to save the plot (optional)
    """
    if not results:
        print("No results to display")
        return
    
    # Extract model names and organize data
    model_names = list(results.keys())
    num_models = len(model_names)
    num_metrics = len(metrics)
    
    # Create grid layout
    cols = min(4, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract metric values for all models
        values = []
        labels = []
        
        for model_name in model_names:
            if metric in results[model_name]:
                values.append(results[model_name][metric])
                labels.append(model_name)
        
        if not values:
            ax.text(0.5, 0.5, f'No data for\n{metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric.replace('_', ' ').title())
            continue
        
        # Create bar plot
        bars = ax.bar(range(len(values)), values, alpha=0.7)
        
        # Color bars based on performance (lower is better for loss, higher for silhouette)
        if 'loss' in metric.lower():
            # Lower is better - use red to green gradient
            norm_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
            colors = plt.cm.RdYlGn_r(norm_values)
        else:
            # Higher is better - use green to red gradient
            norm_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
            colors = plt.cm.RdYlGn(norm_values)
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_latent_dimension_analysis(
    latent_dims: List[int],
    metrics_dict: Dict[str, List[float]],
    figure_size: Tuple[int, int] = (15, 10)
) -> None:
    """
    Analyze performance across different latent dimensions.
    
    Args:
        latent_dims: List of latent dimensions tested
        metrics_dict: Dictionary with metric names as keys and lists of values
        figure_size: Size of the figure
    """
    metrics = list(metrics_dict.keys())
    num_metrics = len(metrics)
    
    # Create subplots
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figure_size)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = metrics_dict[metric]
        
        # Plot with markers and lines
        ax.plot(latent_dims, values, 'o-', linewidth=2, markersize=8)
        
        # Highlight best performance
        if 'loss' in metric.lower():
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        ax.plot(latent_dims[best_idx], values[best_idx], 'r*', 
               markersize=15, label=f'Best: {latent_dims[best_idx]}D')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value annotations
        for x, y in zip(latent_dims, values):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Latent Dimension Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(
    histories: Dict[str, Dict[str, List]],
    convergence_threshold: float = 0.001,
    patience: int = 5,
    figure_size: Tuple[int, int] = (15, 8)
) -> None:
    """
    Analyze convergence behavior across different models.
    
    Args:
        histories: Dictionary of training histories by model name
        convergence_threshold: Threshold for considering loss converged
        patience: Number of epochs to wait for improvement
        figure_size: Size of the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    model_names = list(histories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Plot training curves for all models
    for i, (name, history) in enumerate(histories.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0, 0].plot(epochs, history['train_loss'], 
                       color=colors[i], label=name, linewidth=2)
    
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convergence epochs analysis
    convergence_epochs = []
    model_labels = []
    
    for name, history in histories.items():
        train_loss = history['train_loss']
        
        # Find convergence epoch
        converged_epoch = len(train_loss)  # Default to last epoch
        
        for epoch in range(patience, len(train_loss)):
            recent_losses = train_loss[epoch-patience:epoch]
            if all(abs(train_loss[epoch] - loss) < convergence_threshold for loss in recent_losses):
                converged_epoch = epoch + 1  # Convert to 1-indexed
                break
        
        convergence_epochs.append(converged_epoch)
        model_labels.append(name)
    
    bars = axes[0, 1].bar(range(len(convergence_epochs)), convergence_epochs, 
                         color=colors[:len(convergence_epochs)], alpha=0.7)
    
    for bar, epoch in zip(bars, convergence_epochs):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{epoch}', ha='center', va='bottom')
    
    axes[0, 1].set_title('Convergence Speed')
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Epochs to Convergence')
    axes[0, 1].set_xticks(range(len(model_labels)))
    axes[0, 1].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Final performance comparison
    final_losses = [histories[name]['train_loss'][-1] for name in model_names]
    bars = axes[1, 0].bar(range(len(final_losses)), final_losses, 
                         color=colors[:len(final_losses)], alpha=0.7)
    
    for bar, loss in zip(bars, final_losses):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{loss:.4f}', ha='center', va='bottom')
    
    axes[1, 0].set_title('Final Training Loss')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Final Loss')
    axes[1, 0].set_xticks(range(len(model_labels)))
    axes[1, 0].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Loss improvement analysis
    improvement_rates = []
    for name in model_names:
        train_loss = histories[name]['train_loss']
        initial_loss = train_loss[0]
        final_loss = train_loss[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        improvement_rates.append(improvement)
    
    bars = axes[1, 1].bar(range(len(improvement_rates)), improvement_rates, 
                         color=colors[:len(improvement_rates)], alpha=0.7)
    
    for bar, rate in zip(bars, improvement_rates):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{rate:.1f}%', ha='center', va='bottom')
    
    axes[1, 1].set_title('Loss Improvement Rate')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_xticks(range(len(model_labels)))
    axes[1, 1].set_xticklabels(model_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Convergence Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_training_efficiency(
    training_times: Dict[str, float],
    final_performances: Dict[str, float],
    model_parameters: Dict[str, int],
    figure_size: Tuple[int, int] = (15, 5)
) -> None:
    """
    Analyze training efficiency in terms of time vs performance.
    
    Args:
        training_times: Dictionary of training times by model name
        final_performances: Dictionary of final performance metrics
        model_parameters: Dictionary of parameter counts by model name
        figure_size: Size of the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    
    model_names = list(training_times.keys())
    
    # Training time vs performance
    times = [training_times[name] for name in model_names]
    perfs = [final_performances[name] for name in model_names]
    
    scatter = axes[0].scatter(times, perfs, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    
    for i, name in enumerate(model_names):
        axes[0].annotate(name, (times[i], perfs[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    axes[0].set_xlabel('Training Time (seconds)')
    axes[0].set_ylabel('Final Performance')
    axes[0].set_title('Training Efficiency')
    axes[0].grid(True, alpha=0.3)
    
    # Parameters vs performance
    params = [model_parameters[name] for name in model_names]
    
    scatter = axes[1].scatter(params, perfs, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    
    for i, name in enumerate(model_names):
        axes[1].annotate(name, (params[i], perfs[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    axes[1].set_xlabel('Number of Parameters')
    axes[1].set_ylabel('Final Performance')
    axes[1].set_title('Model Complexity vs Performance')
    axes[1].grid(True, alpha=0.3)
    
    # Training time vs parameters
    scatter = axes[2].scatter(params, times, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    
    for i, name in enumerate(model_names):
        axes[2].annotate(name, (params[i], times[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    axes[2].set_xlabel('Number of Parameters')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Model Complexity vs Training Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 
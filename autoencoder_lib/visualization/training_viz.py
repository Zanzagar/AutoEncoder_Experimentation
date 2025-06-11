"""
Training Visualization Functions

Functions for visualizing training progress, loss curves, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import os


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
        # Handle case where learning_rate might be a single value or a list
        lr_data = history['learning_rate']
        if isinstance(lr_data, (int, float)):
            # If single value, create a flat line across all epochs
            lr_data = [lr_data] * len(epochs)
        elif isinstance(lr_data, list) and len(lr_data) == 1:
            # If list with single value, expand to match epochs
            lr_data = lr_data * len(epochs)
        
        # Only plot if we have valid data
        if isinstance(lr_data, list) and len(lr_data) == len(epochs):
            axes[1].plot(epochs, lr_data, 'g-', linewidth=2)
            axes[1].set_title('Learning Rate Schedule', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        else:
            # Fall back to silhouette scores if learning rate data is invalid
            if 'train_silhouette_scores' in history and history['train_silhouette_scores']:
                axes[1].plot(range(1, len(history['train_silhouette_scores'])+1), 
                           history['train_silhouette_scores'], 'b-', 
                           label='Training Silhouette', linewidth=2)
            if 'test_silhouette_scores' in history and history['test_silhouette_scores']:
                axes[1].plot(range(1, len(history['test_silhouette_scores'])+1), 
                           history['test_silhouette_scores'], 'r-', 
                           label='Test Silhouette', linewidth=2)
            
            if ('train_silhouette_scores' in history and history['train_silhouette_scores']) or \
               ('test_silhouette_scores' in history and history['test_silhouette_scores']):
                axes[1].set_title('Silhouette Scores', fontsize=14)
                axes[1].set_xlabel('Training Steps')
                axes[1].set_ylabel('Silhouette Score')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].axis('off')
    else:
        # Plot silhouette scores if available
        if 'train_silhouette_scores' in history and history['train_silhouette_scores']:
            axes[1].plot(range(1, len(history['train_silhouette_scores'])+1), 
                       history['train_silhouette_scores'], 'b-', 
                       label='Training Silhouette', linewidth=2)
        if 'test_silhouette_scores' in history and history['test_silhouette_scores']:
            axes[1].plot(range(1, len(history['test_silhouette_scores'])+1), 
                       history['test_silhouette_scores'], 'r-', 
                       label='Test Silhouette', linewidth=2)
        
        if ('train_silhouette_scores' in history and history['train_silhouette_scores']) or \
           ('test_silhouette_scores' in history and history['test_silhouette_scores']):
            axes[1].set_title('Silhouette Scores', fontsize=14)
            axes[1].set_xlabel('Training Steps')
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


def plot_performance_heatmap(heatmap_data: np.ndarray,
                            row_labels: List[str],
                            col_labels: List[str],
                            metric_name: str,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None,
                            show_plot: bool = True,
                            figure_size: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a performance heatmap from data matrix.
    
    Args:
        heatmap_data: 2D numpy array with performance values
        row_labels: Labels for rows (e.g., architecture names)
        col_labels: Labels for columns (e.g., latent dimensions)
        metric_name: Name of the metric being visualized
        title: Optional custom title
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        figure_size: Size of the figure
    """
    import seaborn as sns
    
    # Create DataFrame for seaborn
    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=col_labels)
    
    # Create the heatmap
    plt.figure(figsize=figure_size)
    
    # Choose colormap based on metric type
    if 'loss' in metric_name.lower() or 'time' in metric_name.lower():
        # Lower is better - use reverse colormap
        cmap = 'RdYlGn_r'
        cbar_label = f'{metric_name.replace("_", " ").title()} (Lower is Better)'
    else:
        # Higher is better
        cmap = 'RdYlGn'
        cbar_label = f'{metric_name.replace("_", " ").title()} (Higher is Better)'
    
    # Create heatmap with annotations
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.4f',
        cmap=cmap,
        center=None,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': cbar_label}
    )
    
    # Set title
    if title is None:
        title = f'Performance Heatmap: {metric_name.replace("_", " ").title()}'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Architecture', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Highlight best performance
    if not np.isnan(heatmap_data).all():
        if 'loss' in metric_name.lower() or 'time' in metric_name.lower():
            best_coords = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
        else:
            best_coords = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
        
        # Add border around best cell
        plt.gca().add_patch(plt.Rectangle(
            (best_coords[1], best_coords[0]), 1, 1,
            fill=False, edgecolor='black', linewidth=3
        ))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    plt.close()


def plot_multiple_performance_heatmaps(heatmap_data_dict: Dict[str, np.ndarray],
                                      row_labels: List[str],
                                      col_labels: List[str],
                                      save_path: Optional[str] = None,
                                      show_plot: bool = True,
                                      figure_size: Tuple[int, int] = (20, 16)) -> None:
    """
    Create multiple performance heatmaps in a single figure.
    
    Args:
        heatmap_data_dict: Dictionary mapping metric names to 2D numpy arrays
        row_labels: Labels for rows (e.g., architecture names)
        col_labels: Labels for columns (e.g., latent dimensions)
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        figure_size: Size of the figure
    """
    import seaborn as sns
    
    metrics = list(heatmap_data_dict.keys())
    if not metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics[:4]):  # Show up to 4 metrics
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        heatmap_data = heatmap_data_dict[metric]
        
        # Create DataFrame for seaborn
        heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=col_labels)
        
        # Choose colormap
        if 'loss' in metric.lower() or 'time' in metric.lower():
            cmap = 'RdYlGn_r'
        else:
            cmap = 'RdYlGn'
        
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            ax=ax,
            square=False,
            linewidths=0.5,
            cbar_kws={'label': metric.replace("_", " ").title()}
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Configuration' if idx >= len(metrics) - 2 else '', fontsize=10)
        ax.set_ylabel('Architecture' if idx % 2 == 0 else '', fontsize=10)
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Performance Analysis Overview', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    plt.close()


def plot_3d_performance_surface(data_points: List[Dict],
                               metric_name: str,
                               x_label: str = 'Architecture',
                               y_label: str = 'Parameter',
                               architecture_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               show_plot: bool = True,
                               figure_size: Tuple[int, int] = (15, 10)) -> bool:
    """
    Generate 3D performance surface visualization from data points.
    
    Args:
        data_points: List of dictionaries with 'x', 'y', 'z' keys for coordinates and values
        metric_name: Name of the metric being visualized
        x_label: Label for x-axis
        y_label: Label for y-axis  
        architecture_names: Optional list of architecture names for x-axis labels
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
        figure_size: Size of the figure
        
    Returns:
        Boolean indicating success
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import griddata
        
        if len(data_points) < 4:
            print("⚠️ Insufficient data points for surface generation")
            return False
        
        # Extract coordinates and values
        x = [dp['x'] for dp in data_points]
        y = [dp['y'] for dp in data_points]
        z = [dp['z'] for dp in data_points]
        
        # Create interpolation grid
        x_range = np.linspace(min(x), max(x), max(len(set(x)), 3))
        y_range = np.linspace(min(y), max(y), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate surface
        Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=np.nan)
        
        # Create 3D surface plot
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Plot original data points
        ax.scatter(x, y, z, c='red', s=50, alpha=1, label='Data Points')
        
        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'Performance Surface: {metric_name.replace("_", " ").title()}', pad=20)
        
        # Set architecture labels if provided
        if architecture_names and x_label.lower() == 'architecture':
            ax.set_xticks(range(len(architecture_names)))
            ax.set_xticklabels(architecture_names, rotation=45, ha='right')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5, 
                    label=metric_name.replace('_', ' ').title())
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        plt.close()
        
        return True
        
    except ImportError as e:
        print(f"⚠️ 3D plotting requirements not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error generating 3D surface: {e}")
        return False


def plot_performance_contour(data_points: List[Dict],
                            metric_name: str,
                            x_label: str = 'Architecture',
                            y_label: str = 'Parameter',
                            architecture_names: Optional[List[str]] = None,
                            save_path: Optional[str] = None,
                            show_plot: bool = True,
                            figure_size: Tuple[int, int] = (12, 8)) -> bool:
    """
    Generate performance contour map from data points.
    
    Args:
        data_points: List of dictionaries with 'x', 'y', 'z' keys for coordinates and values
        metric_name: Name of the metric being visualized
        x_label: Label for x-axis
        y_label: Label for y-axis
        architecture_names: Optional list of architecture names for x-axis labels
        save_path: Path to save the visualization
        show_plot: Whether to display the plot
        figure_size: Size of the figure
        
    Returns:
        Boolean indicating success
    """
    try:
        from scipy.interpolate import griddata
        
        if len(data_points) < 4:
            print("⚠️ Insufficient data points for contour generation")
            return False
        
        # Extract coordinates and values
        x = [dp['x'] for dp in data_points]
        y = [dp['y'] for dp in data_points]
        z = [dp['z'] for dp in data_points]
        
        # Create interpolation grid
        x_range = np.linspace(min(x), max(x), max(len(set(x)), 3))
        y_range = np.linspace(min(y), max(y), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate surface
        Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=np.nan)
        
        # Create contour plot
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.4, linewidths=0.5)
        
        # Plot original data points
        ax.scatter(x, y, c=z, s=100, edgecolors='white', linewidth=2, 
                  cmap='viridis', label='Data Points')
        
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label(metric_name.replace('_', ' ').title())
        
        # Customize plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Performance Contour Map: {metric_name.replace("_", " ").title()}')
        
        # Set architecture labels if provided
        if architecture_names and x_label.lower() == 'architecture':
            ax.set_xticks(range(len(architecture_names)))
            ax.set_xticklabels(architecture_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        plt.close()
        
        return True
        
    except ImportError as e:
        print(f"⚠️ Contour plotting requirements not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error generating contour plot: {e}")
        return False


def plot_metrics_vs_latent_dim(
    all_results: Dict[str, List[Tuple]],
    save_path: Optional[str] = None,
    session_timestamp: Optional[str] = None,
    random_seed: Optional[int] = None,
    figure_size: Tuple[int, int] = (16, 12)
) -> None:
    """
    Plot metrics (test/train loss and silhouette scores) versus latent dimension for each architecture.
    This implementation matches the AutoEncoderJupyterTest.ipynb reference patterns.
    
    Args:
        all_results: Dictionary mapping architecture -> list of (model, history) tuples
        save_path: Path to save the figure (optional)
        session_timestamp: Session timestamp for title/filename
        random_seed: Random seed for title
        figure_size: Size of the figure
    """
    if not all_results:
        print("No results to plot")
        return
    
    print("📊 Plotting metrics vs latent dimension...")
    
    # Extract architectures and create colors using tab10 colormap like reference notebook
    architectures = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    fig.suptitle(f'Performance Metrics vs Latent Dimension\n{f"Session: {session_timestamp}" if session_timestamp else ""}{f" | Seed: {random_seed}" if random_seed else ""}', 
                 fontsize=16, fontweight='bold')
    
    # Prepare data structures for metrics
    metrics_data = {
        'train_loss': {},
        'test_loss': {},
        'train_silhouette': {},
        'test_silhouette': {}
    }
    
    # Extract data from results
    for arch_idx, (architecture, results) in enumerate(all_results.items()):
        latent_dims = []
        train_losses = []
        test_losses = []
        train_silhouettes = []
        test_silhouettes = []
        
        for model, history in results:
            # Extract latent dimension from model or history
            if hasattr(model, 'latent_dim'):
                latent_dim = model.latent_dim
            elif isinstance(history, dict) and 'latent_dim' in history:
                latent_dim = history['latent_dim']
            else:
                # Try to extract from model architecture
                continue
            
            # Extract metrics from history
            if isinstance(history, dict):
                train_loss = history.get('final_train_loss', history.get('train_loss', [None])[-1] if history.get('train_loss') else None)
                test_loss = history.get('final_test_loss', history.get('test_loss', [None])[-1] if history.get('test_loss') else None)
                train_sil = history.get('final_train_silhouette', history.get('train_silhouette'))
                test_sil = history.get('final_test_silhouette', history.get('test_silhouette'))
                
                if train_loss is not None and test_loss is not None:
                    latent_dims.append(latent_dim)
                    train_losses.append(train_loss)
                    test_losses.append(test_loss)
                    train_silhouettes.append(train_sil if train_sil is not None else 0)
                    test_silhouettes.append(test_sil if test_sil is not None else 0)
        
        # Sort by latent dimension
        if latent_dims:
            sorted_data = sorted(zip(latent_dims, train_losses, test_losses, train_silhouettes, test_silhouettes))
            latent_dims, train_losses, test_losses, train_silhouettes, test_silhouettes = zip(*sorted_data)
            
            metrics_data['train_loss'][architecture] = (latent_dims, train_losses)
            metrics_data['test_loss'][architecture] = (latent_dims, test_losses)
            metrics_data['train_silhouette'][architecture] = (latent_dims, train_silhouettes)
            metrics_data['test_silhouette'][architecture] = (latent_dims, test_silhouettes)
    
    # Plot Train Loss vs Latent Dimension
    axes[0, 0].set_title('Train Loss vs Latent Dimension', fontweight='bold')
    for arch_idx, (architecture, (latent_dims, train_losses)) in enumerate(metrics_data['train_loss'].items()):
        axes[0, 0].plot(latent_dims, train_losses, 'o-', color=colors[arch_idx], 
                       label=architecture, linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Train Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Test Loss vs Latent Dimension  
    axes[0, 1].set_title('Test Loss vs Latent Dimension', fontweight='bold')
    for arch_idx, (architecture, (latent_dims, test_losses)) in enumerate(metrics_data['test_loss'].items()):
        axes[0, 1].plot(latent_dims, test_losses, 'o--', color=colors[arch_idx], 
                       label=architecture, linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Train Silhouette vs Latent Dimension
    axes[1, 0].set_title('Train Silhouette vs Latent Dimension', fontweight='bold')
    for arch_idx, (architecture, (latent_dims, train_silhouettes)) in enumerate(metrics_data['train_silhouette'].items()):
        axes[1, 0].plot(latent_dims, train_silhouettes, 'o-', color=colors[arch_idx], 
                       label=architecture, linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Train Silhouette Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Test Silhouette vs Latent Dimension
    axes[1, 1].set_title('Test Silhouette vs Latent Dimension', fontweight='bold')
    for arch_idx, (architecture, (latent_dims, test_silhouettes)) in enumerate(metrics_data['test_silhouette'].items()):
        axes[1, 1].plot(latent_dims, test_silhouettes, 'o--', color=colors[arch_idx], 
                       label=architecture, linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Latent Dimension') 
    axes[1, 1].set_ylabel('Test Silhouette Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path and session_timestamp:
        filename = f"metrics_vs_latent_dim_{session_timestamp}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_metrics(
    data: Union[Dict[str, List], List[Dict]],
    metrics: List[str] = ['train_loss', 'test_loss'],
    architecture_filter: Optional[str] = None,
    save_path: Optional[str] = None,
    session_timestamp: Optional[str] = None,
    figure_size: Tuple[int, int] = (12, 7)
) -> None:
    """
    Plot training metrics for architectures and latent dimensions.
    
    Supports two input formats:
    1. Dict[str, List[Tuple]] - architecture -> list of (model, history) tuples
    2. List[Dict] - list of history dictionaries
    
    Args:
        data: Results data in either format
        metrics: List of metrics to plot
        architecture_filter: If provided, only plot this architecture
        save_path: Directory to save plots (optional)
        session_timestamp: Session timestamp for filenames
        figure_size: Size of each plot
    """
    
    # Normalize input to architecture -> histories format
    if isinstance(data, dict):
        # Format 1: Dict[str, List[Tuple]] from experiment results
        architectures = {}
        for arch, results in data.items():
            if architecture_filter and arch != architecture_filter:
                continue
            architectures[arch] = [history for model, history in results]
    else:
        # Format 2: List[Dict] from loaded histories
        architectures = {}
        for history in data:
            arch = history.get('architecture', 'unknown')
            if architecture_filter and arch != architecture_filter:
                continue
            if arch not in architectures:
                architectures[arch] = []
            architectures[arch].append(history)
    
    if not architectures:
        print(f"No data to plot for architecture filter: {architecture_filter}")
        return
    
    # Process each architecture
    for architecture, arch_histories in architectures.items():
        if not arch_histories:
            print(f"No results to plot for {architecture}")
            continue
            
        print(f"Plotting training metrics for {architecture} architecture...")
        
        # Sort by latent dimension within each architecture
        arch_histories.sort(key=lambda x: x.get('latent_dim', 0))
        
        # Create color map for latent dimensions
        latent_dims = [h.get('latent_dim', 0) for h in arch_histories]
        unique_dims = sorted(set(latent_dims))
        
        if len(unique_dims) > 1:
            color_indices = [np.log2(dim)/np.log2(max(unique_dims)) for dim in unique_dims]
            color_map = plt.cm.viridis
            dim_colors = {dim: color_map(idx) for dim, idx in zip(unique_dims, color_indices)}
        else:
            dim_colors = {unique_dims[0]: 'blue'}
        
        # Plot each metric
        for metric in metrics:
            plt.figure(figsize=figure_size)
            
            plotted_any = False
            for history in arch_histories:
                latent_dim = history.get('latent_dim', 'Unknown')
                
                if metric in history and len(history[metric]) > 0:
                    steps = range(1, len(history[metric]) + 1)
                    plt.plot(
                        steps, 
                        history[metric], 
                        label=f"dim={latent_dim}",
                        linewidth=2,
                        color=dim_colors.get(latent_dim, 'blue')
                    )
                    plotted_any = True
            
            if not plotted_any:
                plt.close()
                continue
            
            plt.title(f"{architecture}: {metric.replace('_', ' ').title()} vs. Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Sort legend by latent dimension
            handles, labels = plt.gca().get_legend_handles_labels()
            if labels:
                try:
                    dim_values = [int(label.split('=')[1]) for label in labels]
                    sorted_indices = np.argsort(dim_values)
                    plt.legend([handles[i] for i in sorted_indices], [labels[i] for i in sorted_indices])
                except (ValueError, IndexError):
                    plt.legend()
            
            plt.grid(alpha=0.3)
            
            # Visual enhancements
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Save if requested
            if save_path and session_timestamp:
                filename = f"{architecture}_{metric}_{session_timestamp}.png"
                plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
            
            plt.show()
        
        # Plot silhouette scores if available
        if any('silhouette_scores' in h for h in arch_histories):
            plt.figure(figsize=figure_size)
            
            plotted_any = False
            for history in arch_histories:
                latent_dim = history.get('latent_dim', 'Unknown')
                
                if 'silhouette_scores' in history and len(history['silhouette_scores']) > 0:
                    steps = range(1, len(history['silhouette_scores']) + 1)
                    plt.plot(
                        steps, 
                        history['silhouette_scores'], 
                        label=f"dim={latent_dim}",
                        linewidth=2,
                        color=dim_colors.get(latent_dim, 'blue')
                    )
                    plotted_any = True
            
            if plotted_any:
                plt.title(f"{architecture}: Silhouette Score vs. Training Steps")
                plt.xlabel("Training Steps")
                plt.ylabel("Silhouette Score")
                
                # Sort legend by latent dimension
                handles, labels = plt.gca().get_legend_handles_labels()
                if labels:
                    try:
                        dim_values = [int(label.split('=')[1]) for label in labels]
                        sorted_indices = np.argsort(dim_values)
                        plt.legend([handles[i] for i in sorted_indices], [labels[i] for i in sorted_indices])
                    except (ValueError, IndexError):
                        plt.legend()
                
                plt.grid(alpha=0.3)
                
                # Visual enhancements
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                
                plt.tight_layout()
                
                # Save if requested
                if save_path and session_timestamp:
                    filename = f"{architecture}_silhouette_scores_{session_timestamp}.png"
                    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
                
                plt.show()
            else:
                plt.close()


def plot_systematic_training_curves(
    systematic_results: Dict[str, List[Dict]], 
    figure_size: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training curves organized by architecture with latent dimensions as legend.
    Features dual x-axis (training steps + epochs) and fixed validation loss plotting.
    
    Args:
        systematic_results: Dictionary with structure {architecture: [experiment_results]}
        figure_size: Size of the figure
        save_path: Path to save the plot (optional)
    """
    architectures = list(systematic_results.keys())
    n_archs = len(architectures)
    
    if n_archs == 0:
        print("No results to plot")
        return
    
    # Create subplots - 2 columns (train/validation) × n_architectures rows
    fig, axes = plt.subplots(n_archs, 2, figsize=figure_size)
    
    # Handle single architecture case
    if n_archs == 1:
        axes = axes.reshape(1, -1)
    
    # Color map for different latent dimensions using rainbow colormap to match AutoEncoderJupyterTest.ipynb
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    
    def _calculate_epoch_labels_systematic(num_steps, epochs):
        """Calculate epoch labels for systematic experiments with fewer training steps"""
        if epochs <= 0 or num_steps <= 0:
            return []
        
        # For systematic experiments, assume simpler recording (mostly end-of-epoch)
        steps_per_epoch = max(1, num_steps / epochs)
        
        epoch_labels = []
        for step in range(num_steps):
            epoch_pos = (step + 1) / steps_per_epoch
            if epoch_pos == int(epoch_pos):
                epoch_labels.append(f"{int(epoch_pos)}")
            else:
                epoch_labels.append(f"{epoch_pos:.1f}")
        
        return epoch_labels
    
    for arch_idx, (architecture, results) in enumerate(systematic_results.items()):
        train_ax = axes[arch_idx, 0]
        val_ax = axes[arch_idx, 1]
        
        # Group results by latent dimension
        latent_groups = {}
        for result in results:
            latent_dim = result['latent_dim']
            if latent_dim not in latent_groups:
                latent_groups[latent_dim] = []
            latent_groups[latent_dim].append(result)
        
        # Track if we need secondary x-axis for this architecture
        max_steps = 0
        has_epochs = False
        
        # Plot for each latent dimension
        color_idx = 0
        for latent_dim in sorted(latent_groups.keys()):
            color = colors[color_idx % len(colors)]
            
            for result in latent_groups[latent_dim]:
                history = result['history']
                
                # Extract training and validation losses (not test_loss)
                train_losses = history.get('train_loss', [])
                val_losses = history.get('validation_loss', [])
                epochs = history.get('epochs', 0)
                
                if train_losses:
                    steps = range(1, len(train_losses) + 1)
                    train_ax.plot(steps, train_losses, color=color, alpha=0.7, 
                                linewidth=2, label=f'dim={latent_dim}' if result == latent_groups[latent_dim][0] else "")
                    max_steps = max(max_steps, len(train_losses))
                    if epochs > 0:
                        has_epochs = True
                
                if val_losses:
                    steps = range(1, len(val_losses) + 1)
                    val_ax.plot(steps, val_losses, color=color, alpha=0.7, 
                               linewidth=2, label=f'dim={latent_dim}' if result == latent_groups[latent_dim][0] else "")
                    max_steps = max(max_steps, len(val_losses))
                    if epochs > 0:
                        has_epochs = True
            
            color_idx += 1
        
        # Set titles and labels for primary axes
        train_ax.set_title(f'{architecture}: Training Loss vs. Steps', fontsize=12, fontweight='bold')
        train_ax.set_xlabel('Training Steps', fontsize=10)
        train_ax.set_ylabel('Training Loss', fontsize=10)
        train_ax.grid(True, alpha=0.3)
        train_ax.legend(fontsize=9)
        
        val_ax.set_title(f'{architecture}: Validation Loss vs. Steps', fontsize=12, fontweight='bold')
        val_ax.set_xlabel('Training Steps', fontsize=10)
        val_ax.set_ylabel('Validation Loss', fontsize=10)
        val_ax.grid(True, alpha=0.3)
        val_ax.legend(fontsize=9)
        
        # Add secondary x-axis showing epochs if we have epoch information
        if has_epochs and max_steps > 0:
            # Get epoch info from first result that has it
            epochs_info = 0
            for latent_dim in sorted(latent_groups.keys()):
                for result in latent_groups[latent_dim]:
                    epochs_info = result['history'].get('epochs', 0)
                    if epochs_info > 0:
                        break
                if epochs_info > 0:
                    break
            
            if epochs_info > 0:
                # Create secondary x-axis for training plot
                train_ax2 = train_ax.twiny()
                train_ax2.set_xlim(train_ax.get_xlim())
                
                # Calculate epoch tick positions
                epoch_ticks = []
                epoch_labels = []
                if max_steps > 0 and epochs_info > 0:
                    steps_per_epoch = max_steps / epochs_info
                    for epoch in range(1, epochs_info + 1):
                        tick_pos = epoch * steps_per_epoch
                        if tick_pos <= max_steps:
                            epoch_ticks.append(tick_pos)
                            epoch_labels.append(str(epoch))
                
                if epoch_ticks:
                    train_ax2.set_xticks(epoch_ticks)
                    train_ax2.set_xticklabels(epoch_labels, fontsize=9)
                    train_ax2.set_xlabel('Epochs', fontsize=10)
                
                # Create secondary x-axis for validation plot
                val_ax2 = val_ax.twiny()
                val_ax2.set_xlim(val_ax.get_xlim())
                
                if epoch_ticks:
                    val_ax2.set_xticks(epoch_ticks)
                    val_ax2.set_xticklabels(epoch_labels, fontsize=9)
                    val_ax2.set_xlabel('Epochs', fontsize=10)
    
    plt.suptitle('Training Progress by Architecture and Latent Dimension', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved systematic training curves to: {save_path}")
    
    plt.show()


def plot_architecture_latent_heatmaps(
    all_results: Dict[str, List[Tuple]],
    save_path: Optional[str] = None,
    session_timestamp: Optional[str] = None,
    random_seed: Optional[int] = None,
    figure_size: Tuple[int, int] = (15, 10)
) -> None:
    """
    Create heatmaps showing performance metrics across architectures and latent dimensions.
    This implementation matches the AutoEncoderJupyterTest.ipynb reference patterns.
    
    Args:
        all_results: Dictionary mapping architecture -> list of (model, history) tuples
        save_path: Path to save the figure (optional)
        session_timestamp: Session timestamp for title/filename  
        random_seed: Random seed for title
        figure_size: Size of the figure
    """
    if not all_results:
        print("No results to analyze")
        return
        
    print("🔥 Creating architecture × latent dimension heatmaps...")
    
    # Extract all architectures and latent dimensions
    architectures = list(all_results.keys())
    all_latent_dims = set()
    
    for arch, results in all_results.items():
        for model, history in results:
            latent_dim = getattr(model, 'latent_dim', history.get('latent_dim', 'Unknown'))
            if isinstance(latent_dim, (int, float)):
                all_latent_dims.add(latent_dim)
    
    all_latent_dims = sorted(list(all_latent_dims))
    
    if not all_latent_dims:
        print("No valid latent dimensions found")
        return
    
    # Initialize data matrices
    train_loss_matrix = np.zeros((len(architectures), len(all_latent_dims)))
    test_loss_matrix = np.zeros((len(architectures), len(all_latent_dims)))
    train_loss_matrix.fill(np.nan)
    test_loss_matrix.fill(np.nan)
    
    # Fill matrices with data
    for arch_idx, architecture in enumerate(architectures):
        results = all_results[architecture]
        
        for model, history in results:
            # Extract latent dimension
            if hasattr(model, 'latent_dim'):
                latent_dim = model.latent_dim
            elif isinstance(history, dict) and 'latent_dim' in history:
                latent_dim = history['latent_dim']
            else:
                continue
            
            if latent_dim in all_latent_dims:
                dim_idx = all_latent_dims.index(latent_dim)
                
                # Extract final train loss - try multiple possible keys
                train_loss = None
                if isinstance(history, dict):
                    # Try different possible keys for final train loss
                    if 'final_train_loss' in history and history['final_train_loss'] is not None:
                        train_loss = history['final_train_loss']
                    elif 'train_loss' in history and history['train_loss'] and len(history['train_loss']) > 0:
                        train_loss = history['train_loss'][-1]
                    elif 'training_loss' in history and history['training_loss'] is not None:
                        train_loss = history['training_loss']
                
                if train_loss is not None:
                    train_loss_matrix[arch_idx, dim_idx] = train_loss
                
                # Extract final test loss - try multiple possible keys
                test_loss = None
                if isinstance(history, dict):
                    # Try different possible keys for final test loss
                    if 'final_test_loss' in history and history['final_test_loss'] is not None:
                        test_loss = history['final_test_loss']
                    elif 'test_loss' in history and history['test_loss'] and len(history['test_loss']) > 0:
                        test_loss = history['test_loss'][-1]
                    elif 'validation_loss' in history and history['validation_loss'] is not None:
                        test_loss = history['validation_loss']
                    elif 'val_loss' in history and history['val_loss'] is not None:
                        test_loss = history['val_loss']
                
                if test_loss is not None:
                    test_loss_matrix[arch_idx, dim_idx] = test_loss
    
    # Create heatmap subplot (2x1 layout to simplify like reference notebook)
    fig, axes = plt.subplots(2, 1, figsize=figure_size)
    
    # Train Loss Heatmap
    im1 = axes[0].imshow(train_loss_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_title('Train Loss Heatmap: Architecture × Latent Dimension')
    axes[0].set_ylabel('Architecture')
    axes[0].set_xticks(range(len(all_latent_dims)))
    axes[0].set_xticklabels(all_latent_dims)
    axes[0].set_yticks(range(len(architectures)))
    axes[0].set_yticklabels(architectures)
    
    # Add text annotations for train loss
    for i in range(len(architectures)):
        for j in range(len(all_latent_dims)):
            if not np.isnan(train_loss_matrix[i, j]):
                axes[0].text(j, i, f'{train_loss_matrix[i, j]:.4f}', 
                           ha='center', va='center', fontsize=10, color='black')
    
    # Add colorbar for train loss
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Train Loss')
    
    # Test Loss Heatmap
    im2 = axes[1].imshow(test_loss_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_title('Test Loss Heatmap: Architecture × Latent Dimension')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Architecture')
    axes[1].set_xticks(range(len(all_latent_dims)))
    axes[1].set_xticklabels(all_latent_dims)
    axes[1].set_yticks(range(len(architectures)))
    axes[1].set_yticklabels(architectures)
    
    # Add text annotations for test loss
    for i in range(len(architectures)):
        for j in range(len(all_latent_dims)):
            if not np.isnan(test_loss_matrix[i, j]):
                axes[1].text(j, i, f'{test_loss_matrix[i, j]:.4f}', 
                           ha='center', va='center', fontsize=10, color='black')
    
    # Add colorbar for test loss
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Test Loss')
    
    # Add title with session info
    title_parts = ['Performance Heatmaps: Architecture × Latent Dimension']
    if session_timestamp:
        title_parts.append(f'Session: {session_timestamp}')
    if random_seed is not None:
        title_parts.append(f'Seed: {random_seed}')
    
    fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_path and session_timestamp:
        filename = f"architecture_latent_heatmaps_{session_timestamp}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    
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
    axes[0, 0].set_legend()
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
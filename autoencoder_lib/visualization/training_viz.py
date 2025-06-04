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
    
    Args:
        all_results: Dictionary mapping architecture -> list of (model, history) tuples
        save_path: Path to save the figure (optional)
        session_timestamp: Session timestamp for title/filename
        random_seed: Random seed for title
        figure_size: Size of the figure
    """
    # Get list of architectures and create color array
    architectures = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    
    # Create a 2x2 subplot grid for different metrics
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    
    # Define the metrics to plot
    metrics = [
        ('final_train_loss', 'Train Loss', axes[0, 0], '-'),  # Solid line for train metrics
        ('final_test_loss', 'Test Loss', axes[0, 1], '--'),   # Dashed line for test metrics
        ('final_train_silhouette', 'Train Silhouette Score', axes[1, 0], '-'),  # Solid line for train metrics
        ('final_silhouette', 'Test Silhouette Score', axes[1, 1], '--')   # Dashed line for test metrics
    ]
    
    for i, (architecture, results) in enumerate(all_results.items()):
        # Sort results by latent dimension
        sorted_results = sorted(results, key=lambda x: x[1].get('latent_dim', 0))
        
        for metric_key, metric_label, ax, linestyle in metrics:
            latent_dims = []
            metric_values = []
            
            # Use a set to track unique latent dimensions
            seen_dims = set()
            
            for model, history in sorted_results:
                latent_dim = history.get('latent_dim', 0)
                if latent_dim not in seen_dims and metric_key in history:
                    seen_dims.add(latent_dim)
                    latent_dims.append(latent_dim)
                    metric_values.append(history.get(metric_key, 0))
            
            # Plot with consistent colors and specified linestyle
            if latent_dims and metric_values:
                ax.plot(latent_dims, metric_values, 'o-', 
                        label=architecture, 
                        color=colors[i],  # Index-based color assignment
                        linestyle=linestyle)  # Use different linestyles for train vs test
                
                ax.set_title(f'{metric_label} vs. Latent Dimension')
                ax.set_xlabel('Latent Dimension')
                ax.set_ylabel(metric_label)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2)  # Log scale for latent dimensions
                ax.legend()
    
    # Add title with session info if provided
    title = "Metrics Comparison"
    if session_timestamp and random_seed:
        title += f" - Run: {session_timestamp} (Seed: {random_seed})"
    elif session_timestamp:
        title += f" - Run: {session_timestamp}"
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save the figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 
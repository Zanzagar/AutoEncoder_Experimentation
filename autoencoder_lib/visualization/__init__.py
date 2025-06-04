"""
Visualization Module for AutoEncoder Experimentation

This module provides comprehensive visualization capabilities for:
- Dataset samples and class distributions
- t-SNE projections of raw data and latent representations
- Training progress and loss curves
- Reconstruction quality comparisons
- Latent space analysis and exploration

Key Components:
- dataset_viz: Dataset visualization functions
- tsne_viz: t-SNE projection and embedding visualization
- training_viz: Training progress and metrics visualization
- reconstruction_viz: Reconstruction quality and comparison visualization
- latent_viz: Latent space analysis and exploration
"""

# Dataset visualization
from .dataset_viz import (
    visualize_dataset_samples,
    plot_class_distribution,
    show_sample_grid,
    compare_datasets
)

# t-SNE visualization
from .tsne_viz import (
    visualize_raw_data_tsne,
    visualize_latent_tsne,
    compare_tsne_embeddings,
    interactive_tsne_exploration,
    plot_with_labels,
    visualize_side_by_side_latent_spaces
)

# Training visualization
from .training_viz import (
    plot_training_curves,
    plot_loss_landscape,
    plot_performance_grid,
    plot_latent_dimension_analysis,
    plot_convergence_analysis,
    plot_training_efficiency,
    plot_performance_heatmap,
    plot_multiple_performance_heatmaps,
    plot_3d_performance_surface,
    plot_performance_contour,
    plot_metrics_vs_latent_dim,
    plot_training_metrics,
    plot_latent_dim_comparison
)

# Reconstruction visualization
from .reconstruction_viz import (
    visualize_reconstructions,
    plot_reconstruction_loss_grid,
    compare_reconstruction_quality,
    plot_reconstruction_error_heatmap,
    animate_training_reconstructions,
    visualize_dual_reconstructions
)

# Latent space visualization
from .latent_viz import (
    visualize_latent_space_2d,
    compare_latent_spaces,
    plot_latent_distribution,
    plot_latent_interpolation,
    analyze_latent_clustering,
    plot_latent_variance_analysis
)

__all__ = [
    # Dataset visualization
    'visualize_dataset_samples',
    'plot_class_distribution', 
    'show_sample_grid',
    'compare_datasets',
    
    # t-SNE visualization
    'visualize_raw_data_tsne',
    'visualize_latent_tsne',
    'compare_tsne_embeddings',
    'interactive_tsne_exploration',
    'plot_with_labels',
    'visualize_side_by_side_latent_spaces',
    
    # Training visualization
    'plot_training_curves',
    'plot_loss_landscape',
    'plot_performance_grid',
    'plot_latent_dimension_analysis',
    'plot_convergence_analysis',
    'plot_training_efficiency',
    'plot_performance_heatmap',
    'plot_multiple_performance_heatmaps',
    'plot_3d_performance_surface',
    'plot_performance_contour',
    'plot_metrics_vs_latent_dim',
    'plot_training_metrics',
    'plot_latent_dim_comparison',
    
    # Reconstruction visualization
    'visualize_reconstructions',
    'plot_reconstruction_loss_grid',
    'compare_reconstruction_quality',
    'plot_reconstruction_error_heatmap',
    'animate_training_reconstructions',
    'visualize_dual_reconstructions',
    
    # Latent space visualization
    'visualize_latent_space_2d',
    'compare_latent_spaces',
    'plot_latent_distribution',
    'plot_latent_interpolation',
    'analyze_latent_clustering',
    'plot_latent_variance_analysis'
] 
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

from .dataset_viz import (
    visualize_dataset_samples,
    plot_class_distribution,
    show_sample_grid
)

from .tsne_viz import (
    visualize_raw_data_tsne,
    visualize_latent_tsne,
    compare_tsne_embeddings
)

__all__ = [
    # Dataset visualization
    'visualize_dataset_samples',
    'plot_class_distribution', 
    'show_sample_grid',
    
    # t-SNE visualization
    'visualize_raw_data_tsne',
    'visualize_latent_tsne',
    'compare_tsne_embeddings'
] 
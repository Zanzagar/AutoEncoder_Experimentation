"""
Experiment Module

Handles training, evaluation, and experiment management for autoencoder models.
Includes utilities for systematic exploration, parameter sweeps, result tracking,
and comprehensive visualization capabilities.
"""

from .runner import ExperimentRunner
from .wrappers import (
    run_single_experiment,
    run_systematic_experiments,
    load_experiment_results,
    analyze_experiment_results
)
from .visualization import (
    plot_loss_curves,
    plot_metrics_vs_latent_dim,
    create_performance_heatmaps,
    create_comparison_tables,
    save_experiment_summary,
    plot_architecture_comparison,
    generate_comprehensive_report
)

__all__ = [
    # Core experiment functionality
    'ExperimentRunner',
    'run_single_experiment',
    'run_systematic_experiments', 
    'load_experiment_results',
    'analyze_experiment_results',
    
    # Visualization functions
    'plot_loss_curves',
    'plot_metrics_vs_latent_dim',
    'create_performance_heatmaps',
    'create_comparison_tables',
    'save_experiment_summary',
    'plot_architecture_comparison',
    'generate_comprehensive_report'
] 
"""
Experiment Module

Handles training, evaluation, and experiment management for autoencoder models.
Includes utilities for systematic exploration, parameter sweeps, result tracking,
and experiment reporting capabilities that orchestrate core visualization functions.
"""

from .runner import ExperimentRunner
from .wrappers import (
    run_single_experiment,
    run_systematic_experiments,
    load_experiment_results,
    analyze_experiment_results,
    run_latent_analysis_experiment,
    run_systematic_latent_analysis
)
from .experiment_reporting import (
    create_comparison_tables,
    save_experiment_summary,
    generate_comprehensive_report,
    analyze_reconstruction_quality,
    generate_reconstruction_comparison_report,
    create_reconstruction_visualization_batch,
    create_performance_heatmaps,
    analyze_hyperparameter_sensitivity,
    identify_optimal_configurations,
    generate_performance_surfaces
)
from .latent_analysis import (
    analyze_latent_space,
    create_latent_tsne_analysis,
    perform_latent_clustering,
    generate_latent_interpolations,
    analyze_latent_traversals,
    calculate_latent_metrics,
    run_complete_latent_analysis
)

__all__ = [
    # Core experiment functionality
    'ExperimentRunner',
    'run_single_experiment',
    'run_systematic_experiments', 
    'load_experiment_results',
    'analyze_experiment_results',
    'run_latent_analysis_experiment',
    'run_systematic_latent_analysis',
    
    # Experiment reporting functions (orchestrate core visualization)
    'create_comparison_tables',
    'save_experiment_summary',
    'generate_comprehensive_report',
    
    # Reconstruction analysis functions
    'analyze_reconstruction_quality',
    'generate_reconstruction_comparison_report',
    'create_reconstruction_visualization_batch',
    
    # Performance grid analysis functions
    'create_performance_heatmaps',
    'analyze_hyperparameter_sensitivity',
    'identify_optimal_configurations',
    'generate_performance_surfaces',
    
    # Latent space analysis functions
    'analyze_latent_space',
    'create_latent_tsne_analysis',
    'perform_latent_clustering',
    'generate_latent_interpolations',
    'analyze_latent_traversals',
    'calculate_latent_metrics',
    'run_complete_latent_analysis'
] 
"""
Experiment Module

Handles training, evaluation, and experiment management for autoencoder models.
Includes utilities for systematic exploration, parameter sweeps, result tracking,
experiment reporting capabilities that orchestrate core visualization functions,
latent space analysis, and hyperparameter optimization with Optuna.
"""

from .runner import ExperimentRunner
from .wrappers import (
    run_single_experiment,
    run_systematic_experiments,
    load_experiment_results,
    analyze_experiment_results,
    run_latent_analysis_experiment,
    run_systematic_latent_analysis,
    run_optuna_experiment_optimization,
    run_multi_metric_optuna_optimization,
    create_optuna_configuration_from_experiment
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
from .optuna_optimization import (
    OptunaObjective,
    define_hyperparameter_search_space,
    create_optuna_study,
    run_optuna_optimization,
    analyze_optuna_results,
    run_optuna_systematic_optimization
)

__all__ = [
    # Core experiment runner
    'ExperimentRunner',
    
    # Experiment wrappers
    'run_single_experiment',
    'run_systematic_experiments',
    'load_experiment_results',
    'analyze_experiment_results',
    
    # Latent analysis wrappers
    'run_latent_analysis_experiment',
    'run_systematic_latent_analysis',
    
    # Optuna optimization wrappers
    'run_optuna_experiment_optimization',
    'run_multi_metric_optuna_optimization',
    'create_optuna_configuration_from_experiment',
    
    # Experiment reporting
    'create_comparison_tables',
    'save_experiment_summary',
    'generate_comprehensive_report',
    'analyze_reconstruction_quality',
    'generate_reconstruction_comparison_report',
    'create_reconstruction_visualization_batch',
    'create_performance_heatmaps',
    'analyze_hyperparameter_sensitivity',
    'identify_optimal_configurations',
    'generate_performance_surfaces',
    
    # Latent analysis functions
    'analyze_latent_space',
    'create_latent_tsne_analysis',
    'perform_latent_clustering',
    'generate_latent_interpolations',
    'analyze_latent_traversals',
    'calculate_latent_metrics',
    'run_complete_latent_analysis',
    
    # Optuna optimization core
    'OptunaObjective',
    'define_hyperparameter_search_space',
    'create_optuna_study',
    'run_optuna_optimization',
    'analyze_optuna_results',
    'run_optuna_systematic_optimization'
] 
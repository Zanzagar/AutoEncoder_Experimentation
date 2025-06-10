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
    analyze_hyperparameter_sensitivity,
    identify_optimal_configurations
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
    'analyze_hyperparameter_sensitivity',
    'identify_optimal_configurations',
    
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

def create_experiment_with_proper_splits(dataset_info, model_class, model_kwargs, 
                                       training_kwargs=None, experiment_name=None,
                                       train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
                                       batch_size=32, random_seed=42):
    """
    Create and run an experiment with proper train/validation/test split following ML best practices.
    
    This function provides a high-level interface that:
    1. Automatically creates proper 3-way data split
    2. Uses validation data for training monitoring (not test data)
    3. Reserves test data for final unbiased evaluation only
    4. Implements early stopping based on validation loss
    
    Args:
        dataset_info: Dataset information from generate_dataset()
        model_class: Autoencoder model class to instantiate
        model_kwargs: Dictionary of arguments for model initialization
        training_kwargs: Dictionary of training parameters (epochs, learning_rate, etc.)
        experiment_name: Name for the experiment
        train_ratio: Proportion of data for training (default: 0.6)
        validation_ratio: Proportion for validation monitoring (default: 0.2)
        test_ratio: Proportion for final test evaluation (default: 0.2)
        batch_size: Batch size for training
        random_seed: Random seed for reproducible results
        
    Returns:
        Dictionary containing:
        {
            'model': Trained model,
            'history': Training history with validation monitoring,
            'data_split': Information about the data split,
            'final_metrics': Final train/validation/test metrics
        }
        
    Example:
        from autoencoder_lib.data import generate_dataset
        from autoencoder_lib.experiment import create_experiment_with_proper_splits
        from autoencoder_lib.models import SimpleAutoencoder
        
        # Generate dataset
        dataset_info = generate_dataset("geological", "my_dataset", num_samples_per_class=500)
        
        # Run experiment with proper splits
        results = create_experiment_with_proper_splits(
            dataset_info=dataset_info,
            model_class=SimpleAutoencoder,
            model_kwargs={'input_dim': 64*64, 'latent_dim': 32},
            training_kwargs={'epochs': 50, 'learning_rate': 0.001},
            experiment_name="proper_split_experiment"
        )
        
        model = results['model']
        history = results['history']
    """
    from .runner import ExperimentRunner
    from ..data import create_train_validation_test_split
    
    # Set default training parameters
    if training_kwargs is None:
        training_kwargs = {}
    
    print("🚀 Starting experiment with proper train/validation/test methodology")
    print("="*70)
    
    # Create proper 3-way data split
    data_split = create_train_validation_test_split(
        dataset_info=dataset_info,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio, 
        test_ratio=test_ratio,
        batch_size=batch_size,
        random_seed=random_seed
    )
    
    # Create experiment runner
    runner = ExperimentRunner(random_seed=random_seed)
    
    # Run experiment with validation-based monitoring
    print(f"\n🔬 Training {model_class.__name__} with validation monitoring...")
    model, history = runner.train_autoencoder(
        model=model_class(**model_kwargs),
        train_loader=data_split['train_loader'],
        validation_data=data_split['validation_data'],
        validation_labels=data_split['validation_labels'],
        test_data=data_split['test_data'],
        test_labels=data_split['test_labels'],
        class_names=data_split['class_names'],
        experiment_name=experiment_name,
        **training_kwargs
    )
    
    # Extract final metrics
    final_metrics = {
        'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
        'final_validation_loss': history.get('validation_loss', [])[-1] if history.get('validation_loss') else None,
        'early_stopped': history.get('early_stopped', False),
        'best_validation_loss': history.get('best_validation_loss', None),
        'train_samples': len(data_split['train_data_tensor']),
        'validation_samples': len(data_split['validation_data']),
        'test_samples': len(data_split['test_data'])
    }
    
    print("\n🎯 Experiment completed with proper ML methodology!")
    print("="*70)
    
    return {
        'model': model,
        'history': history,
        'data_split': data_split,
        'final_metrics': final_metrics
    } 
"""
Optuna Integration for Autoencoder Hyperparameter Optimization

This module provides comprehensive integration with Optuna for automated hyperparameter
optimization of autoencoder models. It includes parameter search space definition,
optimization strategies, pruning mechanisms, and analysis tools.
"""

import optuna
import torch
import numpy as np
import pandas as pd
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
import time
import warnings

# Import existing framework components
from .wrappers import run_single_experiment
from ..models import MODEL_ARCHITECTURES
from ..utils.reproducibility import set_seed, SeedContext


class OptunaObjective:
    """
    Optuna objective function for autoencoder optimization.
    
    This class encapsulates the objective function logic and maintains
    state needed for hyperparameter optimization trials.
    """
    
    def __init__(
        self,
        dataset_config: Dict[str, Any],
        architecture_names: List[str],
        fixed_params: Dict[str, Any],
        optimization_metric: str = 'final_test_loss',
        minimize_metric: bool = True,
        device: Optional[str] = None,
        output_dir: str = "optuna_trials",
        random_seed: int = 42,
        verbose: bool = False
    ):
        """
        Initialize the Optuna objective function.
        
        Args:
            dataset_config: Configuration for dataset generation
            architecture_names: List of architectures to optimize over
            fixed_params: Fixed parameters that won't be optimized
            optimization_metric: Metric to optimize ('final_test_loss', 'final_silhouette', etc.)
            minimize_metric: Whether to minimize (True) or maximize (False) the metric
            device: PyTorch device for training
            output_dir: Directory to save trial results
            random_seed: Base random seed for reproducibility
            verbose: Whether to print trial progress
        """
        self.dataset_config = dataset_config
        self.architecture_names = architecture_names
        self.fixed_params = fixed_params
        self.optimization_metric = optimization_metric
        self.minimize_metric = minimize_metric
        self.device = device
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Track trial results
        self.trial_results = []
        
        if self.verbose:
            print(f"🎯 Optuna Objective initialized:")
            print(f"   Optimizing: {optimization_metric} ({'minimize' if minimize_metric else 'maximize'})")
            print(f"   Architectures: {architecture_names}")
            print(f"   Output: {output_dir}")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - runs a single trial.
        
        Args:
            trial: Optuna trial object for suggesting parameters
            
        Returns:
            Objective value to optimize
        """
        try:
            # Generate trial-specific random seed
            trial_seed = self.random_seed + trial.number
            
            if self.verbose:
                print(f"\n🧪 Trial {trial.number}: Starting optimization trial...")
            
            # Suggest hyperparameters
            params = self._suggest_hyperparameters(trial)
            
            # Combine with fixed parameters
            full_params = {**self.fixed_params, **params}
            
            if self.verbose:
                print(f"   Parameters: {params}")
            
            # Run experiment with suggested parameters
            with SeedContext(trial_seed):
                experiment_result = run_single_experiment(
                    dataset_config=self.dataset_config,
                    architecture_name=params['architecture'],
                    latent_dim=params['latent_dim'],
                    learning_rate=params['learning_rate'],
                    epochs=full_params.get('epochs', 50),
                    batch_size=params['batch_size'],
                    output_dir=self.output_dir,
                    device=self.device,
                    random_seed=trial_seed,
                    save_model=False,  # Don't save models for every trial
                    verbose=False
                )
            
            # Extract objective value
            if experiment_result.get('success', False):
                metrics = experiment_result.get('metrics', {})
                objective_value = metrics.get(self.optimization_metric)
                
                if objective_value is None:
                    if self.verbose:
                        print(f"   ❌ Metric '{self.optimization_metric}' not found in results")
                    raise optuna.TrialPruned()
                
                # Store trial result
                trial_result = {
                    'trial_number': trial.number,
                    'parameters': params,
                    'metrics': metrics,
                    'objective_value': objective_value,
                    'success': True,
                    'experiment_name': experiment_result.get('experiment_name')
                }
                self.trial_results.append(trial_result)
                
                if self.verbose:
                    direction = "↓" if self.minimize_metric else "↑"
                    print(f"   ✅ {self.optimization_metric}: {objective_value:.6f} {direction}")
                
                return objective_value
            
            else:
                # Failed experiment
                error_msg = experiment_result.get('error', 'Unknown error')
                if self.verbose:
                    print(f"   ❌ Experiment failed: {error_msg}")
                
                trial_result = {
                    'trial_number': trial.number,
                    'parameters': params,
                    'error': error_msg,
                    'success': False
                }
                self.trial_results.append(trial_result)
                
                raise optuna.TrialPruned()
        
        except Exception as e:
            if self.verbose:
                print(f"   💥 Trial {trial.number} failed with exception: {str(e)}")
            raise optuna.TrialPruned()
    
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        # Architecture selection
        architecture = trial.suggest_categorical('architecture', self.architecture_names)
        
        # Core hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        latent_dim = trial.suggest_categorical('latent_dim', [2, 4, 8, 16, 32, 64])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        return {
            'architecture': architecture,
            'learning_rate': learning_rate,
            'latent_dim': latent_dim,
            'batch_size': batch_size
        }


def define_hyperparameter_search_space(
    architectures: List[str] = None,
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2),
    latent_dims: List[int] = None,
    batch_sizes: List[int] = None,
    custom_space: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Define hyperparameter search space for Optuna optimization.
    
    Args:
        architectures: List of architecture names to search over
        learning_rate_range: (min, max) range for learning rate (log scale)
        latent_dims: List of latent dimensions to consider
        batch_sizes: List of batch sizes to consider
        custom_space: Additional custom parameter definitions
        
    Returns:
        Dictionary defining the search space configuration
    """
    if architectures is None:
        architectures = ['simple_linear', 'deeper_linear', 'convolutional', 'deeper_convolutional']
    
    if latent_dims is None:
        latent_dims = [2, 4, 8, 16, 32, 64]
    
    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128]
    
    search_space = {
        'architecture': {'type': 'categorical', 'choices': architectures},
        'learning_rate': {'type': 'float', 'low': learning_rate_range[0], 'high': learning_rate_range[1], 'log': True},
        'latent_dim': {'type': 'categorical', 'choices': latent_dims},
        'batch_size': {'type': 'categorical', 'choices': batch_sizes}
    }
    
    # Add custom space parameters
    if custom_space:
        search_space.update(custom_space)
    
    return search_space


def create_optuna_study(
    study_name: str,
    direction: str = 'minimize',
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    storage: Optional[str] = None
) -> optuna.Study:
    """
    Create and configure an Optuna study.
    
    Args:
        study_name: Name for the optimization study
        direction: 'minimize' or 'maximize' the objective
        sampler: Optuna sampler (defaults to TPESampler)
        pruner: Optuna pruner (defaults to MedianPruner)
        storage: Storage backend for study persistence
        
    Returns:
        Configured Optuna study object
    """
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)
    
    if pruner is None:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage
    )
    
    return study


def run_optuna_optimization(
    dataset_config: Dict[str, Any],
    optimization_config: Dict[str, Any],
    n_trials: int = 100,
    timeout: Optional[int] = None,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    output_dir: str = "optuna_optimization_results",
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for autoencoder models.
    
    Args:
        dataset_config: Configuration for dataset generation
        optimization_config: Configuration for optimization parameters
        n_trials: Number of optimization trials to run
        timeout: Maximum time in seconds for optimization
        n_jobs: Number of parallel jobs (-1 for all cores)
        study_name: Name for the optimization study
        output_dir: Directory to save optimization results
        random_seed: Random seed for reproducibility
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing optimization results and analysis
    """
    if verbose:
        print("🚀 Starting Optuna Hyperparameter Optimization")
        print("=" * 60)
        print(f"📊 Trials: {n_trials}")
        print(f"⏰ Timeout: {timeout}s" if timeout else "⏰ No timeout")
        print(f"💻 Parallel jobs: {n_jobs}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 60)
    
    # Set up study name
    if study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f"autoencoder_optimization_{timestamp}"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract optimization configuration
    architectures = optimization_config.get('architectures', ['simple_linear', 'deeper_linear'])
    optimization_metric = optimization_config.get('metric', 'final_test_loss')
    minimize_metric = optimization_config.get('minimize', True)
    fixed_params = optimization_config.get('fixed_params', {})
    
    # Create Optuna study
    direction = 'minimize' if minimize_metric else 'maximize'
    study = create_optuna_study(
        study_name=study_name,
        direction=direction,
        storage=f"sqlite:///{output_dir}/optuna_study.db"
    )
    
    # Create objective function
    objective = OptunaObjective(
        dataset_config=dataset_config,
        architecture_names=architectures,
        fixed_params=fixed_params,
        optimization_metric=optimization_metric,
        minimize_metric=minimize_metric,
        output_dir=os.path.join(output_dir, "trials"),
        random_seed=random_seed,
        verbose=verbose
    )
    
    # Set up progress callback
    def progress_callback(study, trial):
        if verbose and trial.number % 10 == 0:
            best_value = study.best_value if study.best_trial else "N/A"
            print(f"   Trial {trial.number}: Best so far = {best_value}")
    
    # Run optimization
    try:
        if verbose:
            print(f"\n🎯 Starting optimization with {n_trials} trials...")
        
        start_time = time.time()
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=[progress_callback] if verbose else None,
            show_progress_bar=verbose
        )
        
        optimization_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Optimization completed in {optimization_time:.2f}s")
    
    except KeyboardInterrupt:
        if verbose:
            print(f"\n⚠️ Optimization interrupted by user")
        optimization_time = time.time() - start_time
    
    # Analyze results
    if verbose:
        print("\n📊 Analyzing optimization results...")
    
    analysis_results = analyze_optuna_results(
        study=study,
        objective_results=objective.trial_results,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Compile comprehensive results
    optimization_results = {
        'study_name': study_name,
        'optimization_config': optimization_config,
        'dataset_config': dataset_config,
        'study_summary': {
            'n_trials': len(study.trials),
            'best_value': study.best_value if study.best_trial else None,
            'best_params': study.best_params if study.best_trial else None,
            'optimization_time': optimization_time,
            'optimization_metric': optimization_metric,
            'direction': direction
        },
        'best_trial': study.best_trial.params if study.best_trial else None,
        'analysis': analysis_results,
        'trial_results': objective.trial_results,
        'output_directory': output_dir,
        'study_db_path': f"{output_dir}/optuna_study.db"
    }
    
    # Save comprehensive results
    results_path = os.path.join(output_dir, 'optimization_results.json')
    with open(results_path, 'w') as f:
        # Create JSON-serializable version
        json_results = _make_optuna_results_serializable(optimization_results)
        json.dump(json_results, f, indent=2)
    
    # Save study object
    study_path = os.path.join(output_dir, 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    if verbose:
        print("=" * 60)
        print("🎉 Optuna optimization complete!")
        print(f"   🏆 Best value: {study.best_value:.6f}")
        print(f"   🔧 Best params: {study.best_params}")
        print(f"   📂 Results saved to: {output_dir}")
        print(f"   🗄️ Study database: {study_db_path}")
    
    return optimization_results


def analyze_optuna_results(
    study: optuna.Study,
    objective_results: List[Dict[str, Any]],
    output_dir: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze Optuna optimization results and generate comprehensive analysis.
    
    Args:
        study: Completed Optuna study
        objective_results: Results from objective function trials
        output_dir: Directory to save analysis results
        verbose: Whether to show analysis details
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    if verbose:
        print("🔍 Performing comprehensive Optuna results analysis...")
    
    # Basic study statistics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    basic_stats = {
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'pruned_trials': len(pruned_trials),
        'failed_trials': len(failed_trials),
        'success_rate': len(completed_trials) / len(study.trials) if study.trials else 0
    }
    
    # Parameter importance analysis
    if len(completed_trials) > 1:
        try:
            param_importance = optuna.importance.get_param_importances(study)
            if verbose:
                print(f"   📈 Parameter importance calculated")
        except Exception as e:
            param_importance = {}
            if verbose:
                print(f"   ⚠️ Could not calculate parameter importance: {e}")
    else:
        param_importance = {}
    
    # Trial history analysis
    trial_history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'duration': trial.duration.total_seconds() if trial.duration else None
            })
    
    # Best trials analysis (top 5)
    best_trials = sorted(trial_history, key=lambda x: x['value'])[:5] if study.direction == optuna.study.StudyDirection.MINIMIZE else sorted(trial_history, key=lambda x: x['value'], reverse=True)[:5]
    
    # Architecture performance analysis
    arch_performance = {}
    for result in objective_results:
        if result.get('success', False):
            arch = result['parameters']['architecture']
            value = result['objective_value']
            
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(value)
    
    # Calculate architecture statistics
    arch_stats = {}
    for arch, values in arch_performance.items():
        arch_stats[arch] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
    # Hyperparameter distribution analysis
    param_distributions = {}
    for param_name in ['learning_rate', 'latent_dim', 'batch_size']:
        values = [trial.params.get(param_name) for trial in completed_trials if param_name in trial.params]
        if values:
            if isinstance(values[0], (int, float)):
                param_distributions[param_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                # Categorical parameters
                param_distributions[param_name] = {
                    'unique_values': list(set(values)),
                    'value_counts': {val: values.count(val) for val in set(values)}
                }
    
    # Compile analysis results
    analysis_results = {
        'basic_statistics': basic_stats,
        'parameter_importance': param_importance,
        'best_trials': best_trials,
        'architecture_performance': arch_stats,
        'hyperparameter_distributions': param_distributions,
        'trial_history': trial_history
    }
    
    # Generate visualizations
    if verbose:
        print("   🎨 Generating optimization visualizations...")
    
    try:
        _generate_optuna_visualizations(study, output_dir)
        analysis_results['visualizations_generated'] = True
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not generate visualizations: {e}")
        analysis_results['visualizations_generated'] = False
    
    # Save detailed analysis
    analysis_path = os.path.join(output_dir, 'detailed_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    if verbose:
        print(f"   ✅ Analysis complete!")
        print(f"     • Total trials: {basic_stats['total_trials']}")
        print(f"     • Success rate: {basic_stats['success_rate']:.2%}")
        print(f"     • Best value: {study.best_value:.6f}")
        if param_importance:
            top_param = max(param_importance.items(), key=lambda x: x[1])
            print(f"     • Most important parameter: {top_param[0]} ({top_param[1]:.3f})")
    
    return analysis_results


def _generate_optuna_visualizations(study: optuna.Study, output_dir: str) -> None:
    """Generate standard Optuna visualizations."""
    import matplotlib.pyplot as plt
    
    try:
        # Optimization history
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig1.savefig(os.path.join(output_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # Parameter importances
        if len(study.trials) > 1:
            fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
            fig2.savefig(os.path.join(output_dir, 'parameter_importances.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # Parallel coordinate plot
        fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig3.savefig(os.path.join(output_dir, 'parallel_coordinate.png'), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        # Hyperparameter slice plot
        fig4 = optuna.visualization.matplotlib.plot_slice(study)
        fig4.savefig(os.path.join(output_dir, 'hyperparameter_slice.png'), dpi=150, bbox_inches='tight')
        plt.close(fig4)
        
    except Exception as e:
        # Fallback to basic visualization if advanced plots fail
        plt.figure(figsize=(10, 6))
        values = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        plt.plot(values)
        plt.title('Optimization Progress')
        plt.xlabel('Trial')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'basic_optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close()


def _make_optuna_results_serializable(obj: Any) -> Any:
    """Convert Optuna results to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: _make_optuna_results_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_optuna_results_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


# Wrapper functions for integration with existing experiment framework

def run_optuna_systematic_optimization(
    dataset_config: Dict[str, Any],
    optimization_configs: List[Dict[str, Any]],
    n_trials_per_config: int = 50,
    output_dir: str = "systematic_optuna_optimization",
    n_jobs: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run systematic Optuna optimization with multiple configurations.
    
    Args:
        dataset_config: Configuration for dataset generation
        optimization_configs: List of optimization configurations to test
        n_trials_per_config: Number of trials per configuration
        output_dir: Base directory for saving results
        n_jobs: Number of parallel jobs
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing results from all optimization runs
    """
    if verbose:
        print("🎯 Starting Systematic Optuna Optimization")
        print("=" * 80)
        print(f"   Configurations: {len(optimization_configs)}")
        print(f"   Trials per config: {n_trials_per_config}")
        print(f"   Total trials: {len(optimization_configs) * n_trials_per_config}")
    
    all_results = {}
    
    for i, config in enumerate(optimization_configs):
        config_name = config.get('name', f'config_{i}')
        config_output_dir = os.path.join(output_dir, config_name)
        
        if verbose:
            print(f"\n--- Configuration {i+1}/{len(optimization_configs)}: {config_name} ---")
        
        try:
            result = run_optuna_optimization(
                dataset_config=dataset_config,
                optimization_config=config,
                n_trials=n_trials_per_config,
                output_dir=config_output_dir,
                n_jobs=n_jobs,
                verbose=verbose
            )
            
            all_results[config_name] = result
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Configuration {config_name} failed: {str(e)}")
            all_results[config_name] = {'error': str(e)}
    
    # Generate comparative analysis
    if verbose:
        print(f"\n📊 Generating comparative analysis...")
    
    comparative_analysis = _analyze_systematic_optuna_results(all_results)
    
    # Save comprehensive results
    comprehensive_results = {
        'individual_optimizations': all_results,
        'comparative_analysis': comparative_analysis,
        'summary': {
            'total_configurations': len(optimization_configs),
            'successful_optimizations': len([r for r in all_results.values() if 'error' not in r]),
            'failed_optimizations': len([r for r in all_results.values() if 'error' in r]),
            'total_trials': len(optimization_configs) * n_trials_per_config
        }
    }
    
    results_path = os.path.join(output_dir, 'systematic_optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(_make_optuna_results_serializable(comprehensive_results), f, indent=2)
    
    if verbose:
        print("=" * 80)
        print("🏆 Systematic Optuna optimization complete!")
        summary = comprehensive_results['summary']
        print(f"   ✅ Successful: {summary['successful_optimizations']}/{summary['total_configurations']}")
        if summary['failed_optimizations'] > 0:
            print(f"   ❌ Failed: {summary['failed_optimizations']}")
        print(f"   📂 Results saved to: {output_dir}")
    
    return comprehensive_results


def _analyze_systematic_optuna_results(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze results from systematic Optuna optimization."""
    successful_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if not successful_results:
        return {'error': 'No successful optimizations to analyze'}
    
    # Find best configuration overall
    best_config = None
    best_value = None
    
    for config_name, result in successful_results.items():
        study_summary = result.get('study_summary', {})
        value = study_summary.get('best_value')
        
        if value is not None:
            if best_value is None or value < best_value:  # Assuming minimization
                best_value = value
                best_config = config_name
    
    # Compare architectures across all configurations
    architecture_comparison = {}
    
    for config_name, result in successful_results.items():
        analysis = result.get('analysis', {})
        arch_performance = analysis.get('architecture_performance', {})
        
        for arch, stats in arch_performance.items():
            if arch not in architecture_comparison:
                architecture_comparison[arch] = []
            architecture_comparison[arch].append(stats['mean'])
    
    # Calculate overall architecture rankings
    arch_rankings = {}
    for arch, values in architecture_comparison.items():
        arch_rankings[arch] = {
            'mean_across_configs': float(np.mean(values)),
            'std_across_configs': float(np.std(values)),
            'config_count': len(values)
        }
    
    return {
        'best_configuration': {
            'name': best_config,
            'best_value': best_value
        },
        'architecture_comparison': arch_rankings,
        'configuration_summaries': {
            name: result.get('study_summary', {}) 
            for name, result in successful_results.items()
        }
    } 
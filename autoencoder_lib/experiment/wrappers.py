"""
High-level experiment wrapper functions for autoencoder experimentation.

These functions provide a clean interface for running systematic experiments,
loading results, and performing analysis. They integrate all autoencoder_lib modules
to provide complete experimental workflows.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset

from .runner import ExperimentRunner
from ..models import create_autoencoder, MODEL_ARCHITECTURES
from ..data import generate_dataset
from ..utils.reproducibility import set_seed, SeedContext


def run_single_experiment(
    architecture: str,
    latent_dim: int,
    dataset_config: Dict[str, Any],
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = "experiment_results",
    device: Optional[torch.device] = None,
    random_seed: int = 42,
    save_model: bool = True,
    plot_results: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single autoencoder experiment with specified parameters.
    
    Args:
        architecture: Model architecture name (must be in MODEL_ARCHITECTURES)
        latent_dim: Latent space dimensionality
        dataset_config: Configuration dictionary for dataset generation
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save experiment results
        device: PyTorch device ('cpu' or 'cuda')
        random_seed: Random seed for reproducibility
        save_model: Whether to save the trained model
        plot_results: Whether to generate and save plots
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing experiment results and metrics
    """
    if verbose:
        print(f"\n=== Running Single Experiment ===")
        print(f"Architecture: {architecture}, Latent Dim: {latent_dim}")
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{architecture}_latent{latent_dim}_{timestamp}"
    exp_dir = output_path / experiment_name
    exp_dir.mkdir(exist_ok=True)
    
    try:
        with SeedContext(random_seed):
            # Generate dataset
            if verbose:
                print("Generating dataset...")
            dataset_info = generate_dataset(**dataset_config)
            
            # Prepare data loaders
            train_loader, test_data, test_labels, class_names = _prepare_data_from_dataset(
                dataset_info, dataset_config, batch_size, device
            )
            
            # Create model
            if verbose:
                print(f"Creating {architecture} model...")
            
            # Get input shape from first batch
            sample_batch = next(iter(train_loader))
            input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
            
            model = create_autoencoder(
                architecture=architecture,
                input_shape=input_shape,
                latent_dim=latent_dim
            ).to(device)
            
            if verbose:
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Model created with {total_params:,} parameters")
            
            # Initialize experiment runner
            runner = ExperimentRunner(
                device=device,
                output_dir=str(exp_dir),
                random_seed=random_seed
            )
            
            # Train the model
            if verbose:
                print("Starting training...")
            
            trained_model, history = runner.train_autoencoder(
                model=model,
                train_loader=train_loader,
                test_data=test_data,
                test_labels=test_labels,
                epochs=epochs,
                learning_rate=learning_rate,
                class_names=class_names,
                save_model=save_model,
                experiment_name=experiment_name
            )
            
            # Compile results
            results = {
                'experiment_name': experiment_name,
                'architecture': architecture,
                'latent_dim': latent_dim,
                'config': {
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'random_seed': random_seed,
                    'dataset_config': dataset_config
                },
                'history': history,
                'dataset_info': {
                    'class_names': class_names,
                    'input_shape': list(input_shape),
                    'num_samples': len(train_loader.dataset)
                },
                'output_dir': str(exp_dir),
                'success': True,
                'timestamp': timestamp
            }
            
            # Save results
            results_file = exp_dir / 'experiment_results.json'
            _save_results_to_json(results, results_file)
            
            if verbose:
                final_loss = history.get('final_test_loss', 'N/A')
                print(f"Experiment completed successfully! Final test loss: {final_loss}")
                print(f"Results saved to: {exp_dir}")
            
            return results
            
    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        
        return {
            'experiment_name': experiment_name,
            'architecture': architecture,
            'latent_dim': latent_dim,
            'error': error_msg,
            'success': False,
            'timestamp': timestamp
        }


def run_systematic_experiments(
    architectures: List[str],
    latent_dims: List[int],
    dataset_config: Dict[str, Any],
    learning_rates: List[float] = [0.001],
    epochs_list: List[int] = [100],
    batch_size: int = 32,
    output_dir: str = "systematic_experiments",
    device: Optional[torch.device] = None,
    random_seed: int = 42,
    save_models: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run systematic experiments across multiple architectures and hyperparameters.
    
    Args:
        architectures: List of model architecture names to test
        latent_dims: List of latent dimensions to test
        dataset_config: Configuration for dataset generation
        learning_rates: List of learning rates to test
        epochs_list: List of epoch counts to test
        batch_size: Batch size for training
        output_dir: Directory to save all experiment results
        device: PyTorch device ('cpu' or 'cuda')
        random_seed: Random seed for reproducibility
        save_models: Whether to save trained models
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing all experiment results and summary analysis
    """
    if verbose:
        print("\n" + "="*60)
        print("STARTING SYSTEMATIC AUTOENCODER EXPERIMENTS")
        print("="*60)
    
    # Calculate total experiments
    total_experiments = len(architectures) * len(latent_dims) * len(learning_rates) * len(epochs_list)
    if verbose:
        print(f"Total experiments planned: {total_experiments}")
        print(f"Architectures: {architectures}")
        print(f"Latent dimensions: {latent_dims}")
        print(f"Learning rates: {learning_rates}")
        print(f"Epochs: {epochs_list}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize results storage
    all_results = []
    successful_experiments = 0
    failed_experiments = 0
    
    # Generate dataset once for all experiments
    if verbose:
        print("\nGenerating shared dataset...")
    dataset_info = generate_dataset(**dataset_config)
    
    # Run experiments
    experiment_count = 0
    start_time = time.time()
    
    for architecture in architectures:
        for latent_dim in latent_dims:
            for learning_rate in learning_rates:
                for epochs in epochs_list:
                    experiment_count += 1
                    
                    if verbose:
                        print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                        print(f"Config: {architecture}, latent_dim={latent_dim}, lr={learning_rate}, epochs={epochs}")
                    
                    # Run single experiment
                    result = run_single_experiment(
                        architecture=architecture,
                        latent_dim=latent_dim,
                        dataset_config=dataset_config,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        output_dir=str(output_path / f"experiment_{experiment_count:03d}"),
                        device=device,
                        random_seed=random_seed,
                        save_model=save_models,
                        plot_results=True,
                        verbose=False  # Reduce verbosity for systematic runs
                    )
                    
                    all_results.append(result)
                    
                    if result.get('success', False):
                        successful_experiments += 1
                        if verbose:
                            final_loss = result.get('history', {}).get('final_test_loss', 'N/A')
                            print(f"  ✅ SUCCESS - Final loss: {final_loss}")
                    else:
                        failed_experiments += 1
                        if verbose:
                            error = result.get('error', 'Unknown error')
                            print(f"  ❌ FAILED - {error}")
    
    total_time = time.time() - start_time
    
    # Generate summary analysis
    summary = {
        'systematic_experiment_summary': {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'total_time_seconds': total_time,
            'average_time_per_experiment': total_time / total_experiments if total_experiments > 0 else 0
        },
        'parameters': {
            'architectures': architectures,
            'latent_dims': latent_dims,
            'learning_rates': learning_rates,
            'epochs_list': epochs_list,
            'dataset_config': dataset_config,
            'batch_size': batch_size,
            'random_seed': random_seed
        },
        'all_results': all_results,
        'analysis': _analyze_systematic_results(all_results) if successful_experiments > 0 else None,
        'timestamp': datetime.now().isoformat(),
        'output_dir': str(output_path)
    }
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f'systematic_results_{timestamp}.json'
    _save_results_to_json(summary, results_file)
    
    if verbose:
        print("\n" + "="*60)
        print("SYSTEMATIC EXPERIMENTS COMPLETED")
        print("="*60)
        print(f"Total: {total_experiments}, Successful: {successful_experiments}, Failed: {failed_experiments}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results saved to: {results_file}")
    
    return summary


def load_experiment_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        Dictionary containing experiment results
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded experiment results from: {results_path}")
    
    # Print summary if available
    if 'systematic_experiment_summary' in results:
        summary = results['systematic_experiment_summary']
        print(f"  Total experiments: {summary['total_experiments']}")
        print(f"  Successful: {summary['successful_experiments']}")
        print(f"  Failed: {summary['failed_experiments']}")
    elif 'success' in results:
        status = "SUCCESS" if results['success'] else "FAILED"
        print(f"  Single experiment: {status}")
        if results['success']:
            final_loss = results.get('history', {}).get('final_test_loss', 'N/A')
            print(f"  Final test loss: {final_loss}")
    
    return results


def analyze_experiment_results(results: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze experiment results and generate summary statistics.
    
    Args:
        results: Experiment results dictionary (from load_experiment_results or run_systematic_experiments)
        verbose: Whether to print analysis summary
        
    Returns:
        Dictionary containing analysis results
    """
    if 'all_results' in results:
        # Systematic experiments
        all_results = results['all_results']
        successful_results = [r for r in all_results if r.get('success', False)]
        
        if not successful_results:
            print("No successful experiments to analyze")
            return {'error': 'No successful experiments found'}
        
        analysis = _analyze_systematic_results(successful_results)
        
        if verbose:
            print(f"\n=== Experiment Analysis ===")
            print(f"Successful experiments: {len(successful_results)}")
            print(f"Best experiment: {analysis['best_experiment']['experiment_name']}")
            print(f"Best final loss: {analysis['best_experiment']['final_test_loss']:.6f}")
            print(f"Architecture: {analysis['best_experiment']['architecture']}")
            print(f"Latent dim: {analysis['best_experiment']['latent_dim']}")
            
            if 'architecture_performance' in analysis:
                print(f"\nArchitecture Rankings:")
                for arch, perf in analysis['architecture_performance'].items():
                    print(f"  {arch}: avg loss = {perf['avg_final_loss']:.6f}")
        
        return analysis
    
    elif 'success' in results and results['success']:
        # Single experiment
        history = results.get('history', {})
        analysis = {
            'experiment_type': 'single',
            'experiment_name': results.get('experiment_name', 'unknown'),
            'architecture': results.get('architecture', 'unknown'),
            'latent_dim': results.get('latent_dim', 'unknown'),
            'final_test_loss': history.get('final_test_loss', None),
            'final_train_loss': history.get('final_train_loss', None),
            'training_time': history.get('training_time', None),
            'epochs_completed': len(history.get('train_loss', [])),
            'config': results.get('config', {})
        }
        
        if verbose:
            print(f"\n=== Single Experiment Analysis ===")
            print(f"Experiment: {analysis['experiment_name']}")
            print(f"Architecture: {analysis['architecture']}")
            print(f"Latent dim: {analysis['latent_dim']}")
            print(f"Final test loss: {analysis['final_test_loss']}")
            print(f"Training time: {analysis['training_time']:.2f}s")
        
        return analysis
    
    else:
        return {'error': 'Invalid or failed experiment results'}


# Helper functions

def _prepare_data_from_dataset(
    dataset_info: Dict[str, Any],
    dataset_config: Dict[str, Any],
    batch_size: int,
    device: torch.device
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, List[str]]:
    """Prepare data loaders from generated dataset."""
    from PIL import Image
    import os
    
    output_dir = dataset_config['output_dir']
    class_names = dataset_info['label_names']
    
    # Load training and test data
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(output_dir) / class_name
        
        if class_dir.exists():
            image_files = list(class_dir.glob("*.png"))
            
            # Simple 80/20 train/test split
            split_point = int(len(image_files) * 0.8)
            train_files = image_files[:split_point]
            test_files = image_files[split_point:]
            
            # Load training images
            for img_file in train_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                train_data.append(img_array)
                train_labels.append(class_idx)
            
            # Load test images
            for img_file in test_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                test_data.append(img_array)
                test_labels.append(class_idx)
    
    # Convert to tensors
    train_data = torch.tensor(np.array(train_data), dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(np.array(test_data), dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = TensorDataset(train_data, train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_data.to(device), test_labels.to(device), class_names


def _analyze_systematic_results(successful_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze systematic experiment results and generate summary statistics."""
    if not successful_results:
        return {}
    
    # Find best experiment
    best_experiment = min(
        successful_results, 
        key=lambda x: x.get('history', {}).get('final_test_loss', float('inf'))
    )
    
    # Analyze by architecture
    arch_performance = {}
    for result in successful_results:
        arch = result.get('architecture', 'unknown')
        final_loss = result.get('history', {}).get('final_test_loss')
        
        if final_loss is not None:
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(final_loss)
    
    # Calculate architecture averages
    arch_avg_performance = {}
    for arch, losses in arch_performance.items():
        arch_avg_performance[arch] = {
            'avg_final_loss': np.mean(losses),
            'std_final_loss': np.std(losses),
            'min_final_loss': np.min(losses),
            'max_final_loss': np.max(losses),
            'count': len(losses)
        }
    
    # Analyze by latent dimension
    latent_performance = {}
    for result in successful_results:
        latent_dim = result.get('latent_dim', 'unknown')
        final_loss = result.get('history', {}).get('final_test_loss')
        
        if final_loss is not None:
            if latent_dim not in latent_performance:
                latent_performance[latent_dim] = []
            latent_performance[latent_dim].append(final_loss)
    
    latent_avg_performance = {}
    for latent_dim, losses in latent_performance.items():
        latent_avg_performance[latent_dim] = {
            'avg_final_loss': np.mean(losses),
            'std_final_loss': np.std(losses),
            'min_final_loss': np.min(losses),
            'max_final_loss': np.max(losses),
            'count': len(losses)
        }
    
    return {
        'best_experiment': {
            'experiment_name': best_experiment.get('experiment_name', 'unknown'),
            'architecture': best_experiment.get('architecture', 'unknown'),
            'latent_dim': best_experiment.get('latent_dim', 'unknown'),
            'final_test_loss': best_experiment.get('history', {}).get('final_test_loss'),
            'config': best_experiment.get('config', {})
        },
        'architecture_performance': arch_avg_performance,
        'latent_dim_performance': latent_avg_performance,
        'total_successful_experiments': len(successful_results),
        'overall_stats': {
            'avg_final_loss': np.mean([r.get('history', {}).get('final_test_loss', 0) for r in successful_results]),
            'std_final_loss': np.std([r.get('history', {}).get('final_test_loss', 0) for r in successful_results]),
            'min_final_loss': min([r.get('history', {}).get('final_test_loss', float('inf')) for r in successful_results]),
            'max_final_loss': max([r.get('history', {}).get('final_test_loss', 0) for r in successful_results])
        }
    }


def _save_results_to_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save results data to JSON file, handling non-serializable objects."""
    # Create a copy and handle non-serializable objects
    serializable_data = _make_json_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj 
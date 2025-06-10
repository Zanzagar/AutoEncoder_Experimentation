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
import os
import pandas as pd

from .runner import ExperimentRunner
from ..visualization.training_viz import plot_systematic_training_curves
from ..visualization import (
    plot_performance_grid
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
from ..models import create_autoencoder, MODEL_ARCHITECTURES
from ..data import generate_dataset
from ..utils.reproducibility import set_seed, SeedContext
from .latent_analysis import run_complete_latent_analysis
from ..visualization.latent_viz import visualize_latent_space_2d
from ..visualization.reconstruction_viz import visualize_reconstructions  


def _load_or_generate_dataset(
    dataset_config: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load existing dataset or generate new one based on provided parameters.
    
    Args:
        dataset_config: Configuration for generating new dataset
        dataset_info: Pre-loaded dataset info dictionary
        dataset_path: Path to directory containing dataset_info.npy
        verbose: Whether to print progress information
        
    Returns:
        Dataset info dictionary with standardized keys
        
    Raises:
        ValueError: If no valid dataset source is provided
    """
    # Priority order: dataset_info > dataset_path > dataset_config
    if dataset_info is not None:
        if verbose:
            print("Using provided dataset_info")
        # Ensure standardized keys
        return _standardize_dataset_info(dataset_info, None)
    
    if dataset_path is not None:
        if verbose:
            print(f"Loading dataset from: {dataset_path}")
        
        dataset_info_file = Path(dataset_path) / 'dataset_info.npy'
        if dataset_info_file.exists():
            loaded_info = np.load(str(dataset_info_file), allow_pickle=True).item()
            
            # Standardize and add dataset path information
            standardized_info = _standardize_dataset_info(loaded_info, dataset_path)
            
            if verbose:
                class_names = standardized_info.get('class_names', [])
                print(f"✅ Loaded existing dataset with {len(class_names)} classes")
                print(f"  Classes: {class_names}")
            return standardized_info
        else:
            raise FileNotFoundError(f"dataset_info.npy not found in {dataset_path}")
    
    if dataset_config is not None:
        if verbose:
            print("Generating new dataset...")
        generated_info = generate_dataset(**dataset_config)
        return _standardize_dataset_info(generated_info, dataset_config.get('output_dir'))
    
    raise ValueError("Must provide either dataset_config, dataset_info, or dataset_path")


def _standardize_dataset_info(dataset_info: Dict[str, Any], dataset_dir: Optional[str]) -> Dict[str, Any]:
    """
    Standardize dataset info dictionary to ensure consistent key names.
    
    Args:
        dataset_info: Raw dataset info dictionary
        dataset_dir: Directory path where dataset is stored
        
    Returns:
        Standardized dataset info dictionary
    """
    standardized = dataset_info.copy()
    
    # Handle label_names vs class_names inconsistency
    if 'label_names' in dataset_info and 'class_names' not in dataset_info:
        standardized['class_names'] = dataset_info['label_names']
    elif 'class_names' in dataset_info and 'label_names' not in dataset_info:
        standardized['label_names'] = dataset_info['class_names']
    
    # Ensure we have the dataset directory path for data loading
    if dataset_dir is not None:
        standardized['dataset_directory'] = str(dataset_dir)
    
    return standardized


def run_single_experiment(
    architecture_name: str,
    latent_dim: int,
    learning_rate: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = "experiment_results",
    device: Optional[str] = None,
    random_seed: int = 42,
    save_model: bool = True,
    verbose: bool = True,
    # Dataset sources (provide one of these)
    dataset_config: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a single autoencoder experiment with specified parameters.
    
    Args:
        architecture_name: Model architecture name (must be in MODEL_ARCHITECTURES)
        latent_dim: Latent space dimensionality
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save experiment results
        device: PyTorch device ('cpu' or 'cuda')
        random_seed: Random seed for reproducibility
        save_model: Whether to save the trained model
        verbose: Whether to print progress information
        
        # Dataset sources (provide one of these):
        dataset_config: Configuration dictionary for dataset generation
        dataset_info: Pre-loaded dataset info dictionary  
        dataset_path: Path to directory containing dataset_info.npy
        
    Returns:
        Dictionary containing experiment results and metrics
    """
    if verbose:
        print(f"\n=== Running Single Experiment ===")
        print(f"Architecture: {architecture_name}, Latent Dim: {latent_dim}")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{architecture_name}_latent{latent_dim}_{timestamp}"
    exp_dir = output_path / experiment_name
    exp_dir.mkdir(exist_ok=True)
    
    try:
        with SeedContext(random_seed):
            # Load or generate dataset
            dataset_info = _load_or_generate_dataset(
                dataset_config=dataset_config,
                dataset_info=dataset_info,
                dataset_path=dataset_path,
                verbose=verbose
            )
            
            # Prepare data loaders
            train_loader, validation_data, validation_labels, test_data, test_labels, class_names = _prepare_data_from_dataset(
                dataset_info, dataset_config or {}, batch_size, device
            )
            
            # Create model
            if verbose:
                print(f"Creating {architecture_name} model...")
            
            # Get input shape from first batch
            sample_batch = next(iter(train_loader))
            input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
            
            model = create_autoencoder(
                architecture_name=architecture_name,
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
                print("Starting training with proper validation monitoring...")
            
            trained_model, history = runner.train_autoencoder(
                model=model,
                train_loader=train_loader,
                validation_data=validation_data,  # ✅ Use validation data for monitoring
                validation_labels=validation_labels,  # ✅ Use validation labels for monitoring
                epochs=epochs,
                learning_rate=learning_rate,
                class_names=class_names,
                save_model=save_model,
                experiment_name=experiment_name,
                test_data=test_data,  # ✅ Keep test data separate for final evaluation only
                test_labels=test_labels  # ✅ Keep test labels separate for final evaluation only
            )
            
            # Extract final metrics from history
            metrics = {
                'final_train_loss': history.get('final_train_loss'),
                'final_test_loss': history.get('final_test_loss'),
                'final_train_silhouette': history.get('final_train_silhouette'),
                'final_silhouette': history.get('final_test_silhouette'),
                'training_time': history.get('training_time')
            }
            
            # Save model separately as .pth file if requested
            model_path = None
            if save_model and trained_model is not None:
                model_path = exp_dir / f'{experiment_name}_model.pth'
                torch.save({
                    'model_state_dict': trained_model.state_dict(),
                    'architecture': architecture_name,
                    'latent_dim': latent_dim,
                    'input_shape': list(input_shape),
                    'model_config': {
                        'architecture_name': architecture_name,
                        'input_shape': input_shape,
                        'latent_dim': latent_dim
                    }
                }, model_path)
                if verbose:
                    print(f"Model saved to: {model_path}")
            
            # Compile results (exclude model object for JSON serialization)
            results = {
                'experiment_name': experiment_name,
                'architecture': architecture_name,
                'latent_dim': latent_dim,
                'config': {
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'random_seed': random_seed,
                    'dataset_config': dataset_config
                },
                'history': history,
                'metrics': metrics,
                'model_path': str(model_path) if model_path else None,  # Store path instead of model object
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
                final_loss = metrics.get('final_test_loss', 'N/A')
                print(f"Experiment completed successfully! Final test loss: {final_loss}")
                print(f"Results saved to: {exp_dir}")
            
            return results
            
    except Exception as e:
        error_msg = f"Experiment failed: {str(e)}"
        if verbose:
            print(f"ERROR: {error_msg}")
        
        return {
            'experiment_name': experiment_name,
            'architecture': architecture_name,
            'latent_dim': latent_dim,
            'error': error_msg,
            'success': False,
            'timestamp': timestamp,
            'metrics': {},
            'history': {},
            'model_path': None
        }


def run_systematic_experiments(
    architectures: List[str] = ['simple_linear', 'deeper_linear', 'convolutional', 'deeper_convolutional'],
    latent_dims: List[int] = [4, 8, 16, 32],
    learning_rates: List[float] = [0.001],
    epochs: int = 10,
    batch_size: int = 32,
    output_dir: str = "systematic_experiments",
    random_seed: int = 42,
    device: Optional[str] = None,
    generate_visualizations: bool = True,
    show_plots: bool = True,
    verbose: bool = True,
    # Dataset sources (provide one of these)
    dataset_config: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run systematic experiments across multiple architectures and hyperparameters.
    
    Args:
        architectures: List of architecture names to test
        latent_dims: List of latent dimensions to test
        learning_rates: List of learning rates to test
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save results
        random_seed: Random seed for reproducibility
        device: PyTorch device ('cpu' or 'cuda')
        generate_visualizations: Whether to generate comprehensive visualizations
        show_plots: Whether to display plots during generation
        verbose: Whether to print progress information
        
        # Dataset sources (provide one of these):
        dataset_config: Configuration for dataset generation
        dataset_info: Pre-loaded dataset info dictionary
        dataset_path: Path to directory containing dataset_info.npy
        
    Returns:
        Dictionary mapping architecture names to lists of experiment results
    """
    if verbose:
        print("🚀 Starting Systematic Autoencoder Experiments")
        print("=" * 60)
        print(f"📊 Architectures: {architectures}")
        print(f"🔢 Latent dimensions: {latent_dims}")
        print(f"📈 Learning rates: {learning_rates}")
        print(f"⏰ Epochs: {epochs}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 60)
    
    # Set global random seed
    set_seed(random_seed)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset once for all experiments
    if verbose:
        print("📊 Preparing dataset...")
    
    dataset_info = _load_or_generate_dataset(
        dataset_config=dataset_config,
        dataset_info=dataset_info,
        dataset_path=dataset_path,
        verbose=verbose
    )
    
    # Initialize results dictionary
    all_results = {arch: [] for arch in architectures}
    total_experiments = len(architectures) * len(latent_dims) * len(learning_rates)
    current_experiment = 0
    
    # Run experiments for each combination
    for architecture in architectures:
        if verbose:
            print(f"\n🏗️ Testing architecture: {architecture}")
            print("-" * 40)
        
        for latent_dim in latent_dims:
            for learning_rate in learning_rates:
                current_experiment += 1
                
                if verbose:
                    print(f"🧪 Experiment {current_experiment}/{total_experiments}: {architecture} (dim={latent_dim}, lr={learning_rate})")
                
                # Run single experiment
                with SeedContext(random_seed):
                    experiment_result = run_single_experiment(
                        architecture_name=architecture,
                        latent_dim=latent_dim,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        batch_size=batch_size,
                        output_dir=output_dir,
                        device=device,
                        verbose=False,  # Suppress individual experiment output
                        dataset_info=dataset_info  # Pass the shared dataset
                    )
                
                # Store result with additional metadata
                result_entry = {
                    'experiment_name': experiment_result['experiment_name'],
                    'architecture': architecture,
                    'latent_dim': latent_dim,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'metrics': experiment_result['metrics'],
                    'history': experiment_result['history'],
                    'model_path': experiment_result['model_path']
                }
                
                all_results[architecture].append(result_entry)
                
                # Individual training curves are no longer plotted here
                # All visualizations will be generated at the end
    
    if verbose:
        print(f"\n✅ All {total_experiments} experiments completed!")
        print("=" * 60)
    
    # Generate comprehensive visualization report (matching original notebook output)
    if generate_visualizations:
        if verbose:
            print("\n🎨 Generating comprehensive visualization report...")
        
        # Extract test data from dataset_info for reconstruction analysis
        # We need to prepare data once to get test_data, test_labels, class_names
        device_obj = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        _, _, _, test_data, test_labels, class_names = _prepare_data_from_dataset(
            dataset_info=dataset_info,
            dataset_config={},  # Empty config since we're just extracting existing data
            batch_size=batch_size,
            device=device_obj
        )
        
        # NEW: Generate systematic training curves by architecture
        if verbose:
            print("\n📈 Generating systematic training curves by architecture...")
        
        training_curves_path = f"{output_dir}/systematic_training_curves.png" if output_dir else None
        plot_systematic_training_curves(
            systematic_results=all_results,
            save_path=training_curves_path
        )
        
        # Use the refactored visualization functions 
        visualization_files = generate_comprehensive_report(
            systematic_results=all_results,
            output_dir=output_dir,
            show_plots=show_plots
        )
        
        # # NEW: Add comprehensive reconstruction analysis (TEMPORARILY DISABLED - function signature mismatch)
        # if verbose:
        #     print("\n🔍 Generating reconstruction analysis report...")
        # 
        # # Prepare dataset samples for reconstruction analysis
        # dataset_samples = {
        #     'test_data': test_data,
        #     'test_labels': test_labels,
        #     'class_names': class_names
        # }
        # 
        # # Generate reconstruction comparison report
        # reconstruction_files = generate_reconstruction_comparison_report(
        #     experiment_results=all_results,
        #     dataset_samples=dataset_samples,
        #     output_dir=output_dir,
        #     show_visualizations=show_plots
        # )
        # 
        # # Analyze reconstruction quality across all experiments
        # reconstruction_analysis = analyze_reconstruction_quality(
        #     experiment_results=all_results,
        #     dataset_samples=dataset_samples,
        #     show_best_worst=True
        # )
        # 
        # # 🎯 Generate comprehensive grid-based performance analysis
        # if generate_visualizations:
        #     print("\n🔥 Generating Performance Grid Analysis...")
        #     print("=" * 55)
        #     
        #     # 1. Create performance heatmaps (Architecture × Latent Dimension matrices)
        #     heatmap_files = create_performance_heatmaps(
        #         experiment_results=all_results,
        #         metrics=['final_test_loss', 'final_silhouette', 'training_time'],
        #         output_dir=output_dir,
        #         show_plots=not output_dir  # Show plots only if not saving to file
        #     )
        #     visualization_files.update(heatmap_files)
        #     
        #     # 2. Analyze hyperparameter sensitivity and importance
        #     sensitivity_analysis = analyze_hyperparameter_sensitivity(
        #         experiment_results=all_results,
        #         metrics=['final_test_loss', 'final_silhouette', 'training_time'],
        #         verbose=verbose
        #     )
        #     
        #     # 3. Identify optimal configurations with statistical confidence
        #     optimal_configs = identify_optimal_configurations(
        #         experiment_results=all_results,
        #         primary_metric='final_test_loss',
        #         minimize_metric=True,
        #         top_n=5,
        #         verbose=verbose
        #     )
        #     
        #     # 4. Generate 3D performance surfaces for key metrics
        #     if verbose:
        #         print("\n🌄 Generating 3D Performance Surfaces...")
        #     
        #     for metric in ['final_test_loss', 'final_silhouette']:
        #         surface_file = generate_performance_surfaces(
        #             experiment_results=all_results,
        #             metric=metric,
        #             output_dir=output_dir,
        #             show_plots=not output_dir
        #         )
        #         if surface_file:
        #             visualization_files[f'surface_{metric}'] = surface_file
        #     
        #     # Store analysis results for potential export
        #     analysis_results = {
        #         'sensitivity_analysis': sensitivity_analysis,
        #         'optimal_configurations': optimal_configs,
        #         'heatmap_files': heatmap_files
        #     }
        #     
        #     # Export analysis summary if output directory provided
        #     if output_dir:
        #         analysis_summary_path = Path(output_dir) / 'grid_analysis_summary.json'
        #         with open(analysis_summary_path, 'w') as f:
        #             # Convert numpy types to native Python types for JSON serialization
        #             json_friendly_analysis = convert_to_json_serializable(analysis_results)
        #             json.dump(json_friendly_analysis, f, indent=2)
        #         visualization_files['grid_analysis_summary'] = str(analysis_summary_path)
        #         print(f"💾 Saved grid analysis summary: {analysis_summary_path}")
        #     
        #     all_visualization_files = {**visualization_files, **reconstruction_files}
        
        if verbose:
            print(f"\n✅ All visualizations complete!")
            if visualization_files:
                print("📁 Generated files:")
                for file_type, file_path in visualization_files.items():
                    print(f"  - {file_type}: {file_path}")
    
    # Additional analysis summary (matching original notebook)
    if verbose:
        print("\n📊 Additional Analysis:")
        print("-" * 30)
        
        # Print architecture-specific summaries (matching original notebook)
        for architecture, results in all_results.items():
            print(f"\n🏗️ {architecture} Results:")
            print("Latent Dim | Test Loss | Test Silhouette | Train Time")
            print("-" * 50)
            
            sorted_results = sorted(results, key=lambda x: x['latent_dim'])
            for result in sorted_results:
                metrics = result['metrics']
                test_loss = metrics.get('final_test_loss', 0)
                test_silh = metrics.get('final_silhouette', 0) 
                train_time = metrics.get('training_time', 0)
                
                print(f"{result['latent_dim']:^10} | {test_loss:^9.4f} | {test_silh:^15.4f} | {train_time:^10.2f}s")
    
    # Save complete results to JSON
    results_path = Path(output_dir) / f"systematic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare serializable results (exclude model objects and use _make_json_serializable)
    serializable_results = {}
    for arch, results in all_results.items():
        serializable_results[arch] = []
        for result in results:
            # Use the helper function to properly handle serialization
            serializable_result = _make_json_serializable(result)
            serializable_results[arch].append(serializable_result)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    if verbose:
        print(f"\n💾 Complete results saved to: {results_path}")
        print("🎉 Systematic experiment workflow complete!")
    
    return all_results


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
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Prepare data loaders from generated dataset with proper 3-way split.
    
    Returns:
        Tuple of (train_loader, validation_data, validation_labels, test_data, test_labels, class_names)
    """
    from PIL import Image
    import os
    
    # Get dataset directory from dataset_info (more reliable than dataset_config)
    dataset_dir = dataset_info.get('dataset_directory')
    if dataset_dir is None:
        # Fallback to dataset_config if available
        dataset_dir = dataset_config.get('output_dir')
    
    if dataset_dir is None:
        raise ValueError("No dataset directory found in dataset_info or dataset_config")
    
    class_names = dataset_info.get('class_names', dataset_info.get('label_names', []))
    
    if not class_names:
        raise ValueError("No class names found in dataset_info")
    
    # Load training, validation, and test data
    train_data, train_labels = [], []
    validation_data, validation_labels = [], []
    test_data, test_labels = [], []
    
    print("🔄 Creating proper 3-way data split for training...")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = Path(dataset_dir) / class_name
        
        if class_dir.exists():
            image_files = list(class_dir.glob("*.png"))
            
            # Proper 3-way split: 70% train, 15% validation, 15% test
            total_files = len(image_files)
            train_split = int(total_files * 0.70)
            val_split = int(total_files * 0.85)  # 70% + 15% = 85%
            
            train_files = image_files[:train_split]
            val_files = image_files[train_split:val_split]
            test_files = image_files[val_split:]
            
            print(f"  {class_name}: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
            
            # Load training images
            for img_file in train_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                train_data.append(img_array)
                train_labels.append(class_idx)
            
            # Load validation images
            for img_file in val_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                validation_data.append(img_array)
                validation_labels.append(class_idx)
            
            # Load test images
            for img_file in test_files:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32) / 255.0
                test_data.append(img_array)
                test_labels.append(class_idx)
        else:
            print(f"Warning: Class directory not found: {class_dir}")
    
    if not train_data:
        raise ValueError(f"No training data found in {dataset_dir}")
    
    if not validation_data:
        raise ValueError(f"No validation data found in {dataset_dir}")
    
    # Convert to tensors
    train_data = torch.tensor(np.array(train_data), dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_data = torch.tensor(np.array(validation_data), dtype=torch.float32).unsqueeze(1)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    test_data = torch.tensor(np.array(test_data), dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create DataLoader for training only
    train_dataset = TensorDataset(train_data, train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✅ Data split complete:")
    print(f"   • Training: {len(train_data)} samples")
    print(f"   • Validation: {len(validation_data)} samples (for monitoring)")
    print(f"   • Test: {len(test_data)} samples (for final evaluation)")
    
    return (
        train_loader, 
        validation_data.to(device), 
        validation_labels.to(device),
        test_data.to(device), 
        test_labels.to(device), 
        class_names
    )


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
    serializable_data = _make_json_serializable(data.copy())
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def _make_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        serializable_dict = {}
        for key, value in obj.items():
            # Skip PyTorch model objects entirely
            if isinstance(value, torch.nn.Module):
                continue
            serializable_dict[key] = _make_json_serializable(value)
        return serializable_dict
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
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
    elif isinstance(obj, torch.nn.Module):
        # Skip PyTorch models entirely
        return None
    else:
        return obj 

def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj 

def run_latent_analysis_experiment(
    experiment_results: Dict[str, Any],
    include_interpolations: bool = True,
    include_traversals: bool = True,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive latent space analysis on experiment results.
    
    Args:
        experiment_results: Results from run_single_experiment or loaded results
        include_interpolations: Whether to generate interpolation analysis
        include_traversals: Whether to generate traversal analysis
        output_dir: Directory to save latent analysis results
        device: Device to run analysis on ('cpu', 'cuda', 'mps')
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing complete latent analysis results
    """
    if verbose:
        print("🔬 Starting latent space analysis experiment...")
        print("=" * 60)
    
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    if verbose:
        print(f"   Using device: {device}")
    
    # Load model from experiment results
    model = experiment_results.get('model')
    if model is None:
        model_path = experiment_results.get('model_path')
        if model_path and os.path.exists(model_path):
            # Reconstruct model from config and load weights
            config = experiment_results.get('config', {})
            from autoencoder_lib.models import get_autoencoder_model
            
            model = get_autoencoder_model(
                architecture=config.get('architecture', 'simple_linear'),
                input_shape=config.get('input_shape', (1, 64, 64)),
                latent_dim=config.get('latent_dim', 16)
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            if verbose:
                print(f"   ✅ Loaded model from {model_path}")
        else:
            raise ValueError("No model found in experiment results or model_path invalid")
    
    # Get data from experiment results
    train_data = experiment_results.get('train_data')
    train_labels = experiment_results.get('train_labels')
    test_data = experiment_results.get('test_data')
    test_labels = experiment_results.get('test_labels')
    class_names = experiment_results.get('class_names')
    
    if any(x is None for x in [train_data, train_labels, test_data, test_labels]):
        raise ValueError("Missing required data in experiment results")
    
    # Convert to tensors if needed
    if not isinstance(train_data, torch.Tensor):
        train_data = torch.tensor(train_data, dtype=torch.float32)
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.tensor(train_labels, dtype=torch.long)
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    if not isinstance(test_labels, torch.Tensor):
        test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Set up output directory
    if output_dir is None:
        experiment_name = experiment_results.get('experiment_name', 'experiment')
        output_dir = f"latent_analysis_{experiment_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"   Analyzing {len(train_data)} train + {len(test_data)} test samples")
        print(f"   Classes: {class_names if class_names else 'Unnamed'}")
        print(f"   Output directory: {output_dir}")
    
    # Run complete latent analysis
    try:
        latent_results = run_complete_latent_analysis(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            class_names=class_names,
            device=device,
            output_dir=output_dir,
            include_interpolations=include_interpolations,
            include_traversals=include_traversals
        )
        
        # Add experiment metadata
        latent_results['experiment_metadata'] = {
            'original_experiment': experiment_results.get('experiment_name', 'unknown'),
            'architecture': experiment_results.get('architecture', 'unknown'),
            'latent_dim': experiment_results.get('latent_dim', 'unknown'),
            'final_test_loss': experiment_results.get('history', {}).get('final_test_loss'),
            'analysis_output_dir': output_dir,
            'device_used': str(device),
            'include_interpolations': include_interpolations,
            'include_traversals': include_traversals
        }
        
        if verbose:
            print("=" * 60)
            print("✨ Latent space analysis experiment completed successfully!")
            print(f"   📂 All results saved to: {output_dir}")
            summary = latent_results['analysis_summary']
            print(f"   📊 Summary:")
            print(f"     • Latent Dimension: {summary['latent_dimension']}")
            print(f"     • Mean Silhouette Score: {summary['mean_silhouette_score']:.4f}")
            print(f"     • Optimal Clusters: {summary['train_optimal_clusters']}/{summary['test_optimal_clusters']}")
            if include_interpolations:
                print(f"     • Interpolations: {summary['num_interpolations']}")
            if include_traversals:
                print(f"     • Traversals: {summary['num_traversals']}")
        
        return latent_results
        
    except Exception as e:
        error_msg = f"Failed to run latent analysis: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return {'error': error_msg, 'experiment_metadata': experiment_results.get('experiment_name', 'unknown')}


def run_systematic_latent_analysis(
    systematic_results: Dict[str, List[Dict[str, Any]]],
    max_experiments: int = 5,
    include_interpolations: bool = False,
    include_traversals: bool = False,
    output_base_dir: str = "systematic_latent_analysis",
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run latent space analysis on multiple experiments from systematic results.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        max_experiments: Maximum number of experiments to analyze
        include_interpolations: Whether to generate interpolation analysis
        include_traversals: Whether to generate traversal analysis
        output_base_dir: Base directory for saving all analysis results
        device: Device to run analysis on
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing analysis results for all experiments
    """
    if verbose:
        print("🎯 Starting systematic latent space analysis...")
        print("=" * 80)
    
    # Get successful experiments
    successful_experiments = systematic_results.get('successful_results', [])
    
    if not successful_experiments:
        return {'error': 'No successful experiments found in systematic results'}
    
    # Limit number of experiments to analyze
    experiments_to_analyze = successful_experiments[:max_experiments]
    
    if verbose:
        print(f"   Analyzing {len(experiments_to_analyze)} experiments out of {len(successful_experiments)} successful")
        print(f"   Base output directory: {output_base_dir}")
    
    analysis_results = {}
    comparison_data = []
    
    for i, experiment in enumerate(experiments_to_analyze):
        exp_name = experiment.get('experiment_name', f'experiment_{i}')
        
        if verbose:
            print(f"\n--- Analyzing Experiment {i+1}/{len(experiments_to_analyze)}: {exp_name} ---")
        
        # Set up experiment-specific output directory
        exp_output_dir = os.path.join(output_base_dir, exp_name)
        
        try:
            # Run latent analysis for this experiment
            latent_results = run_latent_analysis_experiment(
                experiment_results=experiment,
                include_interpolations=include_interpolations,
                include_traversals=include_traversals,
                output_dir=exp_output_dir,
                device=device,
                verbose=verbose
            )
            
            analysis_results[exp_name] = latent_results
            
            # Collect data for comparison
            if 'analysis_summary' in latent_results:
                summary = latent_results['analysis_summary']
                comparison_data.append({
                    'experiment_name': exp_name,
                    'architecture': experiment.get('architecture', 'unknown'),
                    'latent_dim': summary.get('latent_dimension', 0),
                    'mean_silhouette_score': summary.get('mean_silhouette_score', 0),
                    'train_optimal_clusters': summary.get('train_optimal_clusters', 0),
                    'test_optimal_clusters': summary.get('test_optimal_clusters', 0),
                    'latent_variance': summary.get('latent_variance', 0),
                    'final_test_loss': experiment.get('history', {}).get('final_test_loss', 0)
                })
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed to analyze {exp_name}: {str(e)}")
            analysis_results[exp_name] = {'error': str(e)}
    
    # Generate comparison analysis
    comparison_analysis = {}
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        comparison_analysis = {
            'best_silhouette_experiment': comparison_df.loc[comparison_df['mean_silhouette_score'].idxmax()].to_dict(),
            'best_clustering_experiment': comparison_df.loc[comparison_df['train_optimal_clusters'].idxmax()].to_dict(),
            'lowest_variance_experiment': comparison_df.loc[comparison_df['latent_variance'].idxmin()].to_dict(),
            'overall_statistics': {
                'mean_silhouette_score': float(comparison_df['mean_silhouette_score'].mean()),
                'std_silhouette_score': float(comparison_df['mean_silhouette_score'].std()),
                'mean_latent_variance': float(comparison_df['latent_variance'].mean()),
                'correlation_loss_silhouette': float(comparison_df['final_test_loss'].corr(comparison_df['mean_silhouette_score']))
            }
        }
        
        # Save comparison data
        comparison_path = os.path.join(output_base_dir, 'latent_analysis_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        if verbose:
            print(f"\n📊 Comparison Analysis:")
            print(f"   Best Silhouette: {comparison_analysis['best_silhouette_experiment']['experiment_name']} ({comparison_analysis['best_silhouette_experiment']['mean_silhouette_score']:.4f})")
            print(f"   Mean Silhouette: {comparison_analysis['overall_statistics']['mean_silhouette_score']:.4f}")
            print(f"   Loss-Silhouette Correlation: {comparison_analysis['overall_statistics']['correlation_loss_silhouette']:.4f}")
    
    # Compile final results
    final_results = {
        'individual_analyses': analysis_results,
        'comparison_analysis': comparison_analysis,
        'analysis_metadata': {
            'total_experiments_analyzed': len(experiments_to_analyze),
            'successful_analyses': len([r for r in analysis_results.values() if 'error' not in r]),
            'failed_analyses': len([r for r in analysis_results.values() if 'error' in r]),
            'output_base_directory': output_base_dir,
            'include_interpolations': include_interpolations,
            'include_traversals': include_traversals
        }
    }
    
    # Save comprehensive results
    comprehensive_path = os.path.join(output_base_dir, 'systematic_latent_analysis_summary.json')
    with open(comprehensive_path, 'w') as f:
        json.dump(convert_to_json_serializable(final_results), f, indent=2)
    
    if verbose:
        print("=" * 80)
        print("🎉 Systematic latent space analysis completed!")
        print(f"   📂 All results saved to: {output_base_dir}")
        metadata = final_results['analysis_metadata']
        print(f"   ✅ Successful: {metadata['successful_analyses']}/{metadata['total_experiments_analyzed']}")
        if metadata['failed_analyses'] > 0:
            print(f"   ❌ Failed: {metadata['failed_analyses']}")
    
    return final_results 

def run_optuna_experiment_optimization(
    experiment_config: Dict[str, Any],
    optimization_config: Dict[str, Any],
    n_trials: int = 100,
    output_dir: str = "optuna_experiment_optimization",
    device: Optional[str] = None,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization integrated with the experiment framework.
    
    This function provides a high-level interface for running Optuna optimization
    using the existing experiment infrastructure and dataset configurations.
    
    Args:
        experiment_config: Configuration for experiment parameters
        optimization_config: Configuration for optimization settings
        n_trials: Number of optimization trials to run
        output_dir: Directory to save optimization results
        device: PyTorch device for training
        random_seed: Random seed for reproducibility
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing optimization results and analysis
    """
    from .optuna_optimization import run_optuna_optimization
    
    if verbose:
        print("🎯 Starting Optuna Experiment Optimization")
        print("=" * 70)
        print(f"   Trials: {n_trials}")
        print(f"   Output: {output_dir}")
        print(f"   Device: {device}")
    
    # Extract dataset configuration from experiment config
    dataset_config = experiment_config.get('dataset_config', {})
    
    # Run Optuna optimization
    optimization_results = run_optuna_optimization(
        dataset_config=dataset_config,
        optimization_config=optimization_config,
        n_trials=n_trials,
        output_dir=output_dir,
        random_seed=random_seed,
        verbose=verbose
    )
    
    # Add experiment metadata
    optimization_results['experiment_metadata'] = {
        'experiment_config': experiment_config,
        'optimization_integration': 'autoencoder_lib.experiment.wrappers',
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print("=" * 70)
        print("✅ Optuna experiment optimization complete!")
    
    return optimization_results


def run_multi_metric_optuna_optimization(
    experiment_config: Dict[str, Any],
    metrics_to_optimize: List[str],
    n_trials_per_metric: int = 50,
    output_dir: str = "multi_metric_optuna_optimization",
    device: Optional[str] = None,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Optuna optimization for multiple metrics separately and compare results.
    
    This function optimizes different metrics (e.g., test_loss, silhouette_score)
    separately and provides comparative analysis of the optimization outcomes.
    
    Args:
        experiment_config: Configuration for experiment parameters
        metrics_to_optimize: List of metric names to optimize
        n_trials_per_metric: Number of trials for each metric optimization
        output_dir: Directory to save optimization results
        device: PyTorch device for training
        random_seed: Random seed for reproducibility
        verbose: Whether to show detailed progress
        
    Returns:
        Dictionary containing results from all metric optimizations
    """
    from .optuna_optimization import run_optuna_systematic_optimization
    
    if verbose:
        print("🎯 Starting Multi-Metric Optuna Optimization")
        print("=" * 80)
        print(f"   Metrics: {metrics_to_optimize}")
        print(f"   Trials per metric: {n_trials_per_metric}")
        print(f"   Total trials: {len(metrics_to_optimize) * n_trials_per_metric}")
    
    # Create optimization configurations for each metric
    optimization_configs = []
    for metric in metrics_to_optimize:
        # Determine if we should minimize or maximize this metric
        minimize_metric = metric.endswith('_loss') or metric.endswith('_error')
        
        config = {
            'name': f'optimize_{metric}',
            'metric': metric,
            'minimize': minimize_metric,
            'architectures': experiment_config.get('architectures', ['simple_linear', 'deeper_linear']),
            'fixed_params': {
                'epochs': experiment_config.get('epochs', 50),
                'save_visualizations': False,  # Skip visualizations for optimization trials
                'save_model': False
            }
        }
        optimization_configs.append(config)
    
    # Extract dataset configuration
    dataset_config = experiment_config.get('dataset_config', {})
    
    # Run systematic optimization
    optimization_results = run_optuna_systematic_optimization(
        dataset_config=dataset_config,
        optimization_configs=optimization_configs,
        n_trials_per_config=n_trials_per_metric,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Add experiment metadata
    optimization_results['experiment_metadata'] = {
        'experiment_config': experiment_config,
        'metrics_optimized': metrics_to_optimize,
        'optimization_integration': 'autoencoder_lib.experiment.wrappers',
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print("=" * 80)
        print("✅ Multi-metric Optuna optimization complete!")
        
        # Show summary of best results for each metric
        if 'comparative_analysis' in optimization_results:
            best_config = optimization_results['comparative_analysis'].get('best_configuration', {})
            print(f"   🏆 Overall best configuration: {best_config.get('name', 'N/A')}")
            print(f"   📊 Best value: {best_config.get('best_value', 'N/A')}")
    
    return optimization_results


def create_optuna_configuration_from_experiment(
    experiment_config: Dict[str, Any],
    optimization_metric: str = 'final_test_loss',
    architectures: Optional[List[str]] = None,
    additional_fixed_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an Optuna optimization configuration from an experiment configuration.
    
    This helper function translates experiment configurations into Optuna-compatible
    optimization configurations, making it easier to integrate optimization with
    existing experimental setups.
    
    Args:
        experiment_config: Base experiment configuration
        optimization_metric: Metric to optimize
        architectures: Architectures to include in optimization search space
        additional_fixed_params: Additional parameters to fix during optimization
        
    Returns:
        Dictionary containing Optuna optimization configuration
    """
    # Default architectures from available models
    if architectures is None:
        architectures = ['simple_linear', 'deeper_linear', 'convolutional', 'deeper_convolutional']
    
    # Determine optimization direction
    minimize_metric = (
        optimization_metric.endswith('_loss') or 
        optimization_metric.endswith('_error') or
        optimization_metric in ['reconstruction_loss', 'total_loss']
    )
    
    # Fixed parameters from experiment config
    fixed_params = {
        'epochs': experiment_config.get('epochs', 50),
        'save_visualizations': False,  # Skip visualizations during optimization
        'save_model': False,  # Don't save models for every trial
        'verbose': False  # Reduce verbosity for trials
    }
    
    # Add additional fixed parameters
    if additional_fixed_params:
        fixed_params.update(additional_fixed_params)
    
    # Create optimization configuration
    optimization_config = {
        'metric': optimization_metric,
        'minimize': minimize_metric,
        'architectures': architectures,
        'fixed_params': fixed_params,
        'search_space': {
            'learning_rate_range': (1e-5, 1e-2),
            'latent_dims': [2, 4, 8, 16, 32, 64],
            'batch_sizes': [16, 32, 64, 128]
        }
    }
    
    return optimization_config 
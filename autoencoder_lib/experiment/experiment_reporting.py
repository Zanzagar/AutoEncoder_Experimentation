"""
Experiment Reporting Functions

Functions specifically for analyzing and reporting on systematic autoencoder experiments.
These functions handle experiment result aggregation, comparison tables, and comprehensive
reporting that goes beyond individual training visualizations.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
import torch

# Import core visualization functions that we'll orchestrate
from ..visualization import (
    plot_training_curves,
    plot_performance_grid,
    plot_latent_dimension_analysis,
    visualize_reconstructions,
    plot_reconstruction_loss_grid,
    compare_reconstruction_quality,
    plot_reconstruction_error_heatmap
)


def create_comparison_tables(systematic_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Create comparison tables for experiment results.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        
    Returns:
        DataFrame with formatted experiment results
    """
    # Collect all results into a list
    all_results = []
    for architecture, results in systematic_results.items():
        for result in results:
            metrics = result['metrics']
            all_results.append({
                'Architecture': architecture,
                'Latent Dim': result['latent_dim'],
                'Learning Rate': result.get('learning_rate', 'N/A'),
                'Epochs': result.get('epochs', 'N/A'),
                'Train Loss': f"{metrics.get('final_train_loss', 0):.4f}",
                'Test Loss': f"{metrics.get('final_test_loss', 0):.4f}",
                'Train Silhouette': f"{metrics.get('final_train_silhouette', 0):.4f}",
                'Test Silhouette': f"{metrics.get('final_silhouette', 0):.4f}",
                'Training Time': f"{metrics.get('training_time', 0):.2f}s"
            })
    
    df = pd.DataFrame(all_results)
    
    # Print comparison tables
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # Sort by test loss (best reconstruction)
    print("\n📊 Best Models by Reconstruction (Test Loss):")
    print("-" * 60)
    print(df.sort_values('Test Loss').to_string(index=False))
    
    # Sort by test silhouette (best separation)
    print("\n🎯 Best Models by Cluster Separation (Test Silhouette):")
    print("-" * 60)
    print(df.sort_values('Test Silhouette', ascending=False).to_string(index=False))
    
    return df


def save_experiment_summary(systematic_results: Dict[str, List[Dict[str, Any]]],
                           save_dir: str) -> str:
    """
    Save experiment summary to CSV file.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        save_dir: Directory to save the CSV file
        
    Returns:
        Path to saved CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(save_dir) / f'experiment_summary_{timestamp}.csv'
    
    # Prepare data for CSV
    csv_data = []
    for architecture, results in systematic_results.items():
        for result in results:
            metrics = result['metrics']
            csv_data.append({
                'Architecture': architecture,
                'Latent_Dim': result['latent_dim'],
                'Learning_Rate': result.get('learning_rate', 'N/A'),
                'Epochs': result.get('epochs', 'N/A'),
                'Train_Loss': metrics.get('final_train_loss', 'N/A'),
                'Test_Loss': metrics.get('final_test_loss', 'N/A'),
                'Train_Silhouette': metrics.get('final_train_silhouette', 'N/A'),
                'Test_Silhouette': metrics.get('final_silhouette', 'N/A'),
                'Training_Time_s': metrics.get('training_time', 'N/A'),
                'Timestamp': timestamp
            })
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = csv_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"📁 Experiment summary saved to: {csv_path}")
    return str(csv_path)


def generate_comprehensive_report(systematic_results: Dict[str, List[Dict[str, Any]]],
                                 output_dir: str,
                                 show_plots: bool = True) -> Dict[str, str]:
    """
    Generate a comprehensive visualization report by orchestrating core visualization functions.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        output_dir: Directory to save all visualizations and reports
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with paths to all generated files
    """
    print("\n🎨 Generating Comprehensive Visualization Report...")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    # 1. Generate training curves for each experiment using core function
    print("📊 Creating individual training curves...")
    for architecture, results in systematic_results.items():
        for result in results:
            if 'history' in result and result['history']:
                # Use core visualization function
                save_path = Path(output_dir) / f"{result['experiment_name']}_training_curves.png" if output_dir else None
                plot_training_curves(
                    history=result['history'],
                    save_path=str(save_path) if save_path else None
                )
    
    # 2. Create performance grid using core function
    print("🔥 Creating performance analysis...")
    # Prepare data for core performance grid function
    performance_data = {}
    for architecture, results in systematic_results.items():
        for result in results:
            model_key = f"{architecture}_latent{result['latent_dim']}"
            performance_data[model_key] = result['metrics']
    
    # Use core visualization function
    save_path = Path(output_dir) / f"performance_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_performance_grid(
        results=performance_data,
        save_path=str(save_path)
    )
    
    # 3. Create latent dimension analysis using core function  
    print("🏗️ Creating latent dimension analysis...")
    for architecture, results in systematic_results.items():
        # Organize data by latent dimension
        latent_dims = []
        metrics_dict = {
            'final_test_loss': [],
            'final_silhouette': [],
            'training_time': []
        }
        
        sorted_results = sorted(results, key=lambda x: x['latent_dim'])
        for result in sorted_results:
            latent_dims.append(result['latent_dim'])
            metrics = result['metrics']
            metrics_dict['final_test_loss'].append(metrics.get('final_test_loss', 0))
            metrics_dict['final_silhouette'].append(metrics.get('final_silhouette', 0))
            metrics_dict['training_time'].append(metrics.get('training_time', 0))
        
        if latent_dims:
            # Use core visualization function
            plot_latent_dimension_analysis(
                latent_dims=latent_dims,
                metrics_dict=metrics_dict
            )
    
    # 4. Create comparison tables
    print("📋 Creating comparison tables...")
    df = create_comparison_tables(systematic_results)
    
    # 5. Save experiment summary
    print("💾 Saving experiment summary...")
    csv_path = save_experiment_summary(systematic_results, save_dir=output_dir)
    generated_files['summary_csv'] = csv_path
    
    print("\n✅ Comprehensive visualization report complete!")
    print(f"📁 All files saved to: {output_dir}")
    
    return generated_files 


def analyze_reconstruction_quality(experiment_results: Dict[str, List[Dict[str, Any]]],
                                 dataset_samples: Optional[Dict] = None,
                                 show_best_worst: bool = True,
                                 num_samples: int = 8) -> Dict[str, Any]:
    """
    Analyze reconstruction quality across different experiments using core visualization functions.
    
    Args:
        experiment_results: Results from systematic experiments
        dataset_samples: Optional dataset samples for reconstruction testing
        show_best_worst: Whether to show best and worst reconstruction examples
        num_samples: Number of samples to show in visualizations
        
    Returns:
        Dictionary with reconstruction quality analysis results
    """
    print("\n🔍 Analyzing Reconstruction Quality...")
    print("=" * 50)
    
    analysis_results = {
        'best_reconstruction_model': None,
        'worst_reconstruction_model': None,
        'architecture_rankings': [],
        'reconstruction_metrics': {}
    }
    
    # Collect reconstruction metrics from all experiments
    all_models = []
    for architecture, results in experiment_results.items():
        for result in results:
            metrics = result['metrics']
            model_info = {
                'architecture': architecture,
                'latent_dim': result['latent_dim'],
                'model_name': result['experiment_name'],
                'test_loss': metrics.get('final_test_loss', float('inf')),
                'train_loss': metrics.get('final_train_loss', float('inf')),
                'model': result.get('model'),  # If models are stored
                'result': result
            }
            all_models.append(model_info)
    
    # Sort by reconstruction quality (test loss)
    sorted_models = sorted(all_models, key=lambda x: x['test_loss'])
    
    # Identify best and worst models
    if sorted_models:
        analysis_results['best_reconstruction_model'] = sorted_models[0]
        analysis_results['worst_reconstruction_model'] = sorted_models[-1]
        
        print(f"🏆 Best Reconstruction: {sorted_models[0]['model_name']} (Loss: {sorted_models[0]['test_loss']:.4f})")
        print(f"⚠️ Worst Reconstruction: {sorted_models[-1]['model_name']} (Loss: {sorted_models[-1]['test_loss']:.4f})")
    
    # Create architecture rankings
    arch_scores = {}
    for model in all_models:
        arch = model['architecture']
        if arch not in arch_scores:
            arch_scores[arch] = {'losses': [], 'count': 0}
        arch_scores[arch]['losses'].append(model['test_loss'])
        arch_scores[arch]['count'] += 1
    
    # Calculate average scores per architecture
    for arch, data in arch_scores.items():
        avg_loss = np.mean(data['losses'])
        analysis_results['architecture_rankings'].append({
            'architecture': arch,
            'avg_reconstruction_loss': avg_loss,
            'model_count': data['count']
        })
    
    # Sort architectures by performance
    analysis_results['architecture_rankings'].sort(key=lambda x: x['avg_reconstruction_loss'])
    
    print("\n📊 Architecture Rankings (by avg reconstruction loss):")
    for i, arch_data in enumerate(analysis_results['architecture_rankings'], 1):
        print(f"{i}. {arch_data['architecture']}: {arch_data['avg_reconstruction_loss']:.4f} "
              f"(from {arch_data['model_count']} models)")
    
    return analysis_results


def generate_reconstruction_comparison_report(experiment_results: Dict[str, List[Dict[str, Any]]],
                                            dataset_samples: Optional[Dict] = None,
                                            output_dir: Optional[str] = None,
                                            show_visualizations: bool = True) -> Dict[str, str]:
    """
    Generate comprehensive reconstruction comparison report using core visualization functions.
    
    Args:
        experiment_results: Results from systematic experiments
        dataset_samples: Optional dataset samples for reconstruction visualization
        output_dir: Directory to save visualizations
        show_visualizations: Whether to display plots
        
    Returns:
        Dictionary with paths to generated files
    """
    print("\n🎨 Generating Reconstruction Comparison Report...")
    print("=" * 55)
    
    generated_files = {}
    
    # 1. Analyze reconstruction quality
    quality_analysis = analyze_reconstruction_quality(experiment_results, dataset_samples)
    
    # 2. If we have dataset samples and models, generate reconstruction visualizations
    if dataset_samples and 'test_data' in dataset_samples:
        test_data = dataset_samples['test_data']
        test_labels = dataset_samples.get('test_labels')
        class_names = dataset_samples.get('class_names')
        
        # Get models for comparison (best, worst, and a few in between)
        models_to_compare = []
        model_names_to_compare = []
        
        # Add best model
        if quality_analysis['best_reconstruction_model'] and quality_analysis['best_reconstruction_model'].get('model'):
            models_to_compare.append(quality_analysis['best_reconstruction_model']['model'])
            model_names_to_compare.append(f"Best: {quality_analysis['best_reconstruction_model']['model_name']}")
        
        # Add worst model
        if quality_analysis['worst_reconstruction_model'] and quality_analysis['worst_reconstruction_model'].get('model'):
            models_to_compare.append(quality_analysis['worst_reconstruction_model']['model'])
            model_names_to_compare.append(f"Worst: {quality_analysis['worst_reconstruction_model']['model_name']}")
        
        # Generate comparison visualizations using core functions
        if models_to_compare and len(models_to_compare) > 1:
            print("🔄 Generating model reconstruction comparisons...")
            
            # Select a few representative samples for visualization
            num_samples = min(8, len(test_data))
            sample_indices = np.linspace(0, len(test_data) - 1, num_samples, dtype=int)
            
            for sample_idx in sample_indices:
                # Get reconstructions from each model
                reconstructions_list = []
                for model in models_to_compare:
                    model.eval()
                    with torch.no_grad():
                        sample_input = test_data[sample_idx:sample_idx+1]
                        if torch.cuda.is_available():
                            sample_input = sample_input.cuda()
                            model = model.cuda()
                        _, reconstruction = model(sample_input)
                        reconstructions_list.append(reconstruction.cpu())
                
                # Use core visualization function
                compare_reconstruction_quality(
                    originals=test_data[sample_idx:sample_idx+1],
                    reconstructions_list=reconstructions_list,
                    model_names=model_names_to_compare,
                    sample_idx=0,
                    figure_size=(15, 5)
                )
    
    # 3. Generate reconstruction loss analysis across latent dimensions
    print("📈 Generating reconstruction loss analysis...")
    for architecture in experiment_results.keys():
        arch_results = experiment_results[architecture]
        if len(arch_results) > 1:  # Only if we have multiple latent dimensions
            
            # Sort by latent dimension
            sorted_results = sorted(arch_results, key=lambda x: x['latent_dim'])
            
            latent_dims = [r['latent_dim'] for r in sorted_results]
            reconstruction_losses = [r['metrics'].get('final_test_loss', 0) for r in sorted_results]
            
            # Plot reconstruction loss vs latent dimension
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(latent_dims, reconstruction_losses, marker='o', linewidth=2, markersize=8)
            plt.xlabel('Latent Dimension')
            plt.ylabel('Reconstruction Loss (Test)')
            plt.title(f'Reconstruction Quality vs Latent Dimension: {architecture}')
            plt.grid(True, alpha=0.3)
            
            # Save if output directory provided
            if output_dir:
                save_path = Path(output_dir) / f'reconstruction_analysis_{architecture}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                generated_files[f'reconstruction_analysis_{architecture}'] = str(save_path)
            
            if show_visualizations:
                plt.show()
            plt.close()
    
    print("✅ Reconstruction comparison report complete!")
    
    return generated_files


def create_reconstruction_visualization_batch(models: List,
                                            model_names: List[str],
                                            test_data,
                                            test_labels=None,
                                            class_names=None,
                                            num_samples: int = 8,
                                            output_dir: Optional[str] = None) -> List[str]:
    """
    Create batch reconstruction visualizations using core visualization functions.
    
    Args:
        models: List of trained models
        model_names: Names corresponding to each model
        test_data: Test dataset for reconstruction
        test_labels: Optional test labels
        class_names: Optional class names
        num_samples: Number of samples to visualize
        output_dir: Optional directory to save visualizations
        
    Returns:
        List of paths to generated visualization files
    """
    import torch
    
    print(f"\n🖼️ Creating reconstruction visualizations for {len(models)} models...")
    
    generated_files = []
    
    # Prepare test samples
    num_samples = min(num_samples, len(test_data))
    sample_indices = np.linspace(0, len(test_data) - 1, num_samples, dtype=int)
    
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        print(f"📸 Processing model {i+1}/{len(models)}: {model_name}")
        
        model.eval()
        with torch.no_grad():
            # Get reconstructions for selected samples
            test_samples = test_data[sample_indices]
            if torch.cuda.is_available():
                test_samples = test_samples.cuda()
                model = model.cuda()
            
            _, reconstructions = model(test_samples)
            
            # Move back to CPU for visualization
            test_samples = test_samples.cpu()
            reconstructions = reconstructions.cpu()
            
            # Prepare labels if available
            sample_labels = test_labels[sample_indices] if test_labels is not None else None
            
            # Use core visualization function
            visualize_reconstructions(
                originals=test_samples,
                reconstructions=reconstructions,
                labels=sample_labels,
                class_names=class_names,
                num_samples=num_samples,
                title=f"Reconstructions: {model_name}"
            )
            
            # Generate reconstruction loss grid using core function
            # Calculate individual reconstruction losses
            mse_losses = torch.nn.functional.mse_loss(
                reconstructions.view(reconstructions.size(0), -1),
                test_samples.view(test_samples.size(0), -1),
                reduction='none'
            ).mean(dim=1)
            
            plot_reconstruction_loss_grid(
                originals=test_samples,
                reconstructions=reconstructions,
                losses=mse_losses,
                labels=sample_labels,
                class_names=class_names,
                num_samples=num_samples
            )
            
            # Generate error heatmap using core function
            plot_reconstruction_error_heatmap(
                originals=test_samples[:4],  # Show first 4 samples
                reconstructions=reconstructions[:4]
            )
    
    return generated_files 
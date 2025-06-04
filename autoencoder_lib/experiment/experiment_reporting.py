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


def create_performance_heatmaps(experiment_results: Dict[str, List[Dict[str, Any]]],
                                metrics: List[str] = ['final_test_loss', 'final_silhouette', 'training_time'],
                                output_dir: Optional[str] = None,
                                show_plots: bool = True) -> Dict[str, str]:
    """
    Create comprehensive performance heatmaps showing Architecture × Latent Dimension matrices.
    
    Args:
        experiment_results: Results from systematic experiments
        metrics: List of metrics to visualize in heatmaps
        output_dir: Directory to save heatmap visualizations
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with paths to generated heatmap files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n🔥 Creating Performance Heatmaps...")
    print("=" * 45)
    
    generated_files = {}
    
    # Organize data into matrix format
    architectures = list(experiment_results.keys())
    all_latent_dims = set()
    
    # Collect all latent dimensions
    for arch_results in experiment_results.values():
        for result in arch_results:
            all_latent_dims.add(result['latent_dim'])
    
    all_latent_dims = sorted(list(all_latent_dims))
    
    # Create heatmaps for each metric
    for metric in metrics:
        print(f"📊 Creating heatmap for: {metric}")
        
        # Create matrix for this metric
        heatmap_data = np.zeros((len(architectures), len(all_latent_dims)))
        heatmap_data.fill(np.nan)  # Start with NaN for missing values
        
        for i, architecture in enumerate(architectures):
            arch_results = experiment_results[architecture]
            for result in arch_results:
                latent_dim = result['latent_dim']
                j = all_latent_dims.index(latent_dim)
                metric_value = result['metrics'].get(metric, np.nan)
                heatmap_data[i, j] = metric_value
        
        # Create DataFrame for seaborn
        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=architectures,
            columns=[f'{dim}D' for dim in all_latent_dims]
        )
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        
        # Choose colormap based on metric type
        if 'loss' in metric.lower() or 'time' in metric.lower():
            # Lower is better - use reverse colormap
            cmap = 'RdYlGn_r'
            cbar_label = f'{metric.replace("_", " ").title()} (Lower is Better)'
        else:
            # Higher is better
            cmap = 'RdYlGn'
            cbar_label = f'{metric.replace("_", " ").title()} (Higher is Better)'
        
        # Create heatmap with annotations
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='.4f',
            cmap=cmap,
            center=None,
            square=False,
            linewidths=0.5,
            cbar_kws={'label': cbar_label}
        )
        
        plt.title(f'Performance Heatmap: {metric.replace("_", " ").title()}\n'
                  f'Architecture × Latent Dimension Analysis', fontsize=14, pad=20)
        plt.xlabel('Latent Dimension', fontsize=12)
        plt.ylabel('Architecture', fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Highlight best performance
        if not np.isnan(heatmap_data).all():
            if 'loss' in metric.lower() or 'time' in metric.lower():
                best_coords = np.unravel_index(np.nanargmin(heatmap_data), heatmap_data.shape)
            else:
                best_coords = np.unravel_index(np.nanargmax(heatmap_data), heatmap_data.shape)
            
            # Add border around best cell
            plt.gca().add_patch(plt.Rectangle(
                (best_coords[1], best_coords[0]), 1, 1,
                fill=False, edgecolor='black', linewidth=3
            ))
        
        plt.tight_layout()
        
        # Save if output directory provided
        if output_dir:
            save_path = Path(output_dir) / f'performance_heatmap_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            generated_files[f'heatmap_{metric}'] = str(save_path)
            print(f"  💾 Saved: {save_path}")
        
        if show_plots:
            plt.show()
        plt.close()
    
    # Create combined overview heatmap
    if len(metrics) > 1:
        print("📊 Creating combined performance overview...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics[:4]):  # Show up to 4 metrics
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Recreate heatmap data for this metric
            heatmap_data = np.zeros((len(architectures), len(all_latent_dims)))
            heatmap_data.fill(np.nan)
            
            for i, architecture in enumerate(architectures):
                arch_results = experiment_results[architecture]
                for result in arch_results:
                    latent_dim = result['latent_dim']
                    j = all_latent_dims.index(latent_dim)
                    metric_value = result['metrics'].get(metric, np.nan)
                    heatmap_data[i, j] = metric_value
            
            heatmap_df = pd.DataFrame(
                heatmap_data,
                index=architectures,
                columns=[f'{dim}D' for dim in all_latent_dims]
            )
            
            # Choose colormap
            if 'loss' in metric.lower() or 'time' in metric.lower():
                cmap = 'RdYlGn_r'
            else:
                cmap = 'RdYlGn'
            
            sns.heatmap(
                heatmap_df,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                ax=ax,
                square=False,
                linewidths=0.5,
                cbar_kws={'label': metric.replace("_", " ").title()}
            )
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.set_xlabel('Latent Dimension' if idx >= len(metrics) - 2 else '', fontsize=10)
            ax.set_ylabel('Architecture' if idx % 2 == 0 else '', fontsize=10)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Performance Analysis Overview: Architecture × Latent Dimension', 
                     fontsize=16, y=0.98)
        plt.tight_layout()
        
        if output_dir:
            save_path = Path(output_dir) / 'performance_heatmaps_overview.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            generated_files['heatmaps_overview'] = str(save_path)
        
        if show_plots:
            plt.show()
        plt.close()
    
    print("✅ Performance heatmaps complete!")
    return generated_files


def analyze_hyperparameter_sensitivity(experiment_results: Dict[str, List[Dict[str, Any]]],
                                      metrics: List[str] = ['final_test_loss', 'final_silhouette'],
                                      verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze hyperparameter sensitivity through statistical analysis.
    
    Args:
        experiment_results: Results from systematic experiments
        metrics: List of metrics to analyze
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    print("\n📈 Analyzing Hyperparameter Sensitivity...")
    print("=" * 50)
    
    sensitivity_results = {
        'architecture_effects': {},
        'latent_dim_effects': {},
        'interaction_effects': {},
        'parameter_importance': {}
    }
    
    # Organize data for analysis
    all_data = []
    for architecture, arch_results in experiment_results.items():
        for result in arch_results:
            data_point = {
                'architecture': architecture,
                'latent_dim': result['latent_dim'],
                **result['metrics']
            }
            all_data.append(data_point)
    
    if not all_data:
        print("⚠️ No data available for sensitivity analysis")
        return sensitivity_results
    
    df = pd.DataFrame(all_data)
    
    # Analyze each metric
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        print(f"\n📊 Analyzing {metric}...")
        
        # Architecture effects
        arch_effects = {}
        architectures = df['architecture'].unique()
        for arch in architectures:
            arch_data = df[df['architecture'] == arch][metric].dropna()
            if len(arch_data) > 0:
                arch_effects[arch] = {
                    'mean': arch_data.mean(),
                    'std': arch_data.std(),
                    'min': arch_data.min(),
                    'max': arch_data.max(),
                    'count': len(arch_data)
                }
        
        sensitivity_results['architecture_effects'][metric] = arch_effects
        
        # Latent dimension effects
        latent_effects = {}
        latent_dims = sorted(df['latent_dim'].unique())
        for latent_dim in latent_dims:
            latent_data = df[df['latent_dim'] == latent_dim][metric].dropna()
            if len(latent_data) > 0:
                latent_effects[latent_dim] = {
                    'mean': latent_data.mean(),
                    'std': latent_data.std(),
                    'min': latent_data.min(),
                    'max': latent_data.max(),
                    'count': len(latent_data)
                }
        
        sensitivity_results['latent_dim_effects'][metric] = latent_effects
        
        # Calculate parameter importance (variance explained)
        total_variance = df[metric].var()
        
        # Architecture variance
        arch_means = [arch_effects[arch]['mean'] for arch in architectures if arch in arch_effects]
        arch_variance = np.var(arch_means) if len(arch_means) > 1 else 0
        arch_importance = arch_variance / total_variance if total_variance > 0 else 0
        
        # Latent dimension variance
        latent_means = [latent_effects[ld]['mean'] for ld in latent_dims if ld in latent_effects]
        latent_variance = np.var(latent_means) if len(latent_means) > 1 else 0
        latent_importance = latent_variance / total_variance if total_variance > 0 else 0
        
        sensitivity_results['parameter_importance'][metric] = {
            'architecture': arch_importance,
            'latent_dim': latent_importance,
            'total_explained': min(1.0, arch_importance + latent_importance)
        }
        
        if verbose:
            print(f"  Architecture Importance: {arch_importance:.3f} ({arch_importance*100:.1f}%)")
            print(f"  Latent Dim Importance: {latent_importance:.3f} ({latent_importance*100:.1f}%)")
            print(f"  Total Explained Variance: {sensitivity_results['parameter_importance'][metric]['total_explained']*100:.1f}%")
    
    # Print summary rankings
    if verbose:
        print(f"\n🏆 Parameter Importance Rankings:")
        print("-" * 35)
        for metric in metrics:
            if metric in sensitivity_results['parameter_importance']:
                importance = sensitivity_results['parameter_importance'][metric]
                arch_pct = importance['architecture'] * 100
                latent_pct = importance['latent_dim'] * 100
                
                print(f"\n{metric.replace('_', ' ').title()}:")
                if arch_pct > latent_pct:
                    print(f"  1. Architecture ({arch_pct:.1f}%)")
                    print(f"  2. Latent Dimension ({latent_pct:.1f}%)")
                else:
                    print(f"  1. Latent Dimension ({latent_pct:.1f}%)")
                    print(f"  2. Architecture ({arch_pct:.1f}%)")
    
    return sensitivity_results


def identify_optimal_configurations(experiment_results: Dict[str, List[Dict[str, Any]]],
                                   primary_metric: str = 'final_test_loss',
                                   minimize_metric: bool = True,
                                   top_n: int = 5,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Identify optimal hyperparameter configurations with statistical confidence.
    
    Args:
        experiment_results: Results from systematic experiments
        primary_metric: Primary metric for optimization
        minimize_metric: Whether to minimize the metric (True for loss, False for accuracy)
        top_n: Number of top configurations to return
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with optimal configurations and analysis
    """
    print(f"\n🎯 Identifying Optimal Configurations...")
    print(f"Primary Metric: {primary_metric} ({'minimize' if minimize_metric else 'maximize'})")
    print("=" * 60)
    
    # Collect all configurations with their performance
    all_configs = []
    for architecture, arch_results in experiment_results.items():
        for result in arch_results:
            metric_value = result['metrics'].get(primary_metric)
            if metric_value is not None:
                config = {
                    'architecture': architecture,
                    'latent_dim': result['latent_dim'],
                    'learning_rate': result.get('learning_rate', 'N/A'),
                    'epochs': result.get('epochs', 'N/A'),
                    'primary_metric_value': metric_value,
                    'all_metrics': result['metrics'],
                    'experiment_name': result.get('experiment_name', 'unknown')
                }
                all_configs.append(config)
    
    if not all_configs:
        print("⚠️ No valid configurations found")
        return {'error': 'No valid configurations'}
    
    # Sort configurations by primary metric
    sorted_configs = sorted(all_configs, 
                          key=lambda x: x['primary_metric_value'], 
                          reverse=not minimize_metric)
    
    # Get top configurations
    top_configs = sorted_configs[:top_n]
    
    # Calculate performance statistics
    all_values = [config['primary_metric_value'] for config in all_configs]
    mean_performance = np.mean(all_values)
    std_performance = np.std(all_values)
    
    optimal_results = {
        'top_configurations': top_configs,
        'best_configuration': top_configs[0] if top_configs else None,
        'performance_statistics': {
            'mean': mean_performance,
            'std': std_performance,
            'min': min(all_values),
            'max': max(all_values),
            'total_configs': len(all_configs)
        },
        'improvement_analysis': {}
    }
    
    # Analyze improvement over baseline (mean)
    if top_configs:
        best_value = top_configs[0]['primary_metric_value']
        if minimize_metric:
            improvement = ((mean_performance - best_value) / mean_performance) * 100
        else:
            improvement = ((best_value - mean_performance) / mean_performance) * 100
        
        optimal_results['improvement_analysis'] = {
            'improvement_over_mean': improvement,
            'standard_deviations_better': abs(best_value - mean_performance) / std_performance if std_performance > 0 else 0
        }
    
    if verbose:
        print(f"🏆 Top {len(top_configs)} Configurations:")
        print("-" * 50)
        
        for i, config in enumerate(top_configs, 1):
            print(f"\n{i}. {config['architecture']} (Latent: {config['latent_dim']}D)")
            print(f"   {primary_metric}: {config['primary_metric_value']:.6f}")
            print(f"   Experiment: {config['experiment_name']}")
            
            # Show other key metrics
            other_metrics = ['final_silhouette', 'training_time']
            for metric in other_metrics:
                if metric in config['all_metrics'] and metric != primary_metric:
                    value = config['all_metrics'][metric]
                    print(f"   {metric}: {value:.4f}")
        
        # Performance summary
        best_config = optimal_results['best_configuration']
        improvement = optimal_results['improvement_analysis']['improvement_over_mean']
        std_better = optimal_results['improvement_analysis']['standard_deviations_better']
        
        print(f"\n📊 Performance Summary:")
        print(f"   Best Performance: {best_config['primary_metric_value']:.6f}")
        print(f"   Mean Performance: {mean_performance:.6f}")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   Standard Deviations Better: {std_better:.2f}σ")
        
        # Architecture and latent dimension analysis
        arch_counts = {}
        latent_counts = {}
        for config in top_configs:
            arch = config['architecture']
            latent = config['latent_dim']
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
            latent_counts[latent] = latent_counts.get(latent, 0) + 1
        
        print(f"\n🏗️ Top Architecture Distribution:")
        for arch, count in sorted(arch_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {arch}: {count}/{top_n} ({count/top_n*100:.1f}%)")
        
        print(f"\n🔢 Top Latent Dimension Distribution:")
        for latent, count in sorted(latent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {latent}D: {count}/{top_n} ({count/top_n*100:.1f}%)")
    
    return optimal_results


def generate_performance_surfaces(experiment_results: Dict[str, List[Dict[str, Any]]],
                                 metric: str = 'final_test_loss',
                                 output_dir: Optional[str] = None,
                                 show_plots: bool = True) -> Optional[str]:
    """
    Generate 3D performance surface visualizations across hyperparameter space.
    
    Args:
        experiment_results: Results from systematic experiments
        metric: Metric to visualize as surface
        output_dir: Directory to save visualization
        show_plots: Whether to display plots
        
    Returns:
        Path to saved visualization file
    """
    print(f"\n🌄 Generating 3D Performance Surface: {metric}")
    print("=" * 55)
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import griddata
        
        # Collect data points
        architectures = list(experiment_results.keys())
        data_points = []
        
        # Map architectures to numeric values for 3D plotting
        arch_mapping = {arch: i for i, arch in enumerate(architectures)}
        
        for architecture, arch_results in experiment_results.items():
            arch_idx = arch_mapping[architecture]
            for result in arch_results:
                metric_value = result['metrics'].get(metric)
                if metric_value is not None:
                    data_points.append({
                        'arch_idx': arch_idx,
                        'latent_dim': result['latent_dim'],
                        'metric_value': metric_value,
                        'architecture': architecture
                    })
        
        if len(data_points) < 4:
            print("⚠️ Insufficient data points for surface generation")
            return None
        
        # Extract coordinates and values
        x = [dp['arch_idx'] for dp in data_points]  # Architecture indices
        y = [dp['latent_dim'] for dp in data_points]  # Latent dimensions
        z = [dp['metric_value'] for dp in data_points]  # Metric values
        
        # Create interpolation grid
        x_range = np.linspace(min(x), max(x), len(architectures))
        y_range = np.linspace(min(y), max(y), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate surface
        Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=np.nan)
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Plot original data points
        ax.scatter(x, y, z, c='red', s=50, alpha=1, label='Experiments')
        
        # Customize plot
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Latent Dimension')
        ax.set_zlabel(metric.replace('_', ' ').title())
        ax.set_title(f'Performance Surface: {metric.replace("_", " ").title()}\n'
                    f'Across Architecture and Latent Dimension Space', pad=20)
        
        # Set architecture labels
        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels(architectures, rotation=45, ha='right')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5, 
                    label=metric.replace('_', ' ').title())
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save if output directory provided
        save_path = None
        if output_dir:
            save_path = Path(output_dir) / f'performance_surface_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved 3D surface: {save_path}")
        
        if show_plots:
            plt.show()
        plt.close()
        
        # Create contour projection
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        contour_lines = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.4, linewidths=0.5)
        
        # Plot original data points
        ax.scatter(x, y, c=z, s=100, edgecolors='white', linewidth=2, 
                  cmap='viridis', label='Experiments')
        
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label(metric.replace('_', ' ').title())
        
        # Customize plot
        ax.set_xlabel('Architecture Index')
        ax.set_ylabel('Latent Dimension')
        ax.set_title(f'Performance Contour Map: {metric.replace("_", " ").title()}')
        
        # Set architecture labels
        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels(architectures, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save contour plot
        if output_dir:
            contour_path = Path(output_dir) / f'performance_contour_{metric}.png'
            plt.savefig(contour_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved contour map: {contour_path}")
        
        if show_plots:
            plt.show()
        plt.close()
        
        print("✅ Performance surface generation complete!")
        return str(save_path) if save_path else None
        
    except ImportError as e:
        print(f"⚠️ 3D plotting requirements not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Error generating performance surface: {e}")
        return None 
"""
Experiment Reporting Functions

Functions specifically for analyzing and reporting on systematic autoencoder experiments.
These functions handle experiment result aggregation, comparison tables, and comprehensive
reporting that goes beyond individual training visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import csv
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import core visualization functions
from ..visualization.training_viz import (
    plot_latent_dimension_analysis,
    plot_metrics_vs_latent_dim
)
from ..visualization.reconstruction_viz import visualize_reconstructions as plot_reconstruction_comparison

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
    
    # 1. Create latent dimension analysis using core function  
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
    
    # 2. Create new metric vs latent dimension plots (Train/Test Loss and Train/Test Silhouette)
    print("📊 Creating metrics vs latent dimension plots...")
    # Convert systematic_results to the format expected by plot_metrics_vs_latent_dim
    all_results_for_metrics = {}
    for architecture, results in systematic_results.items():
        all_results_for_metrics[architecture] = []
        for result in results:
            # Create dummy model object and use history directly
            model_placeholder = None
            history = result  # The result dict contains all the metrics we need
            all_results_for_metrics[architecture].append((model_placeholder, history))
    
    # Generate the metrics vs latent dimension plot
    plot_metrics_vs_latent_dim(
        all_results=all_results_for_metrics,
        save_path=output_dir,
        session_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    # 3. Create comparison tables
    print("📋 Creating comparison tables...")
    df = create_comparison_tables(systematic_results)
    
    # 4. Save experiment summary
    print("💾 Saving experiment summary...")
    csv_path = save_experiment_summary(systematic_results, save_dir=output_dir)
    generated_files['summary_csv'] = csv_path
    
    print("\n✅ Comprehensive visualization report complete!")
    print(f"📁 All files saved to: {output_dir}")
    
    return generated_files


def analyze_reconstruction_quality(
    results_dict: Dict[str, Dict],
    dataset_name: str = "dataset",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze reconstruction quality across different experiments.
    
    Args:
        results_dict: Dictionary containing experiment results
        dataset_name: Name of the dataset being analyzed
        save_dir: Directory to save analysis outputs
        
    Returns:
        Dictionary containing reconstruction quality analysis
    """
    print("🔍 Analyzing reconstruction quality across experiments...")
    
    quality_analysis = {
        'reconstruction_scores': {},
        'best_configurations': {},
        'quality_trends': {}
    }
    
    # Extract reconstruction data for analysis
    reconstruction_data = {}
    
    for exp_name, exp_results in results_dict.items():
        if 'reconstructions' in exp_results:
            recon_data = exp_results['reconstructions']
            quality_analysis['reconstruction_scores'][exp_name] = {
                'mse_scores': recon_data.get('mse_scores', []),
                'ssim_scores': recon_data.get('ssim_scores', []),
                'avg_mse': np.mean(recon_data.get('mse_scores', [])) if recon_data.get('mse_scores') else None,
                'avg_ssim': np.mean(recon_data.get('ssim_scores', [])) if recon_data.get('ssim_scores') else None
            }
            
            reconstruction_data[exp_name] = recon_data
    
    # Identify best configurations
    if quality_analysis['reconstruction_scores']:
        best_mse = min(
            [(name, data['avg_mse']) for name, data in quality_analysis['reconstruction_scores'].items() 
             if data['avg_mse'] is not None],
            key=lambda x: x[1],
            default=(None, None)
        )
        
        best_ssim = max(
            [(name, data['avg_ssim']) for name, data in quality_analysis['reconstruction_scores'].items() 
             if data['avg_ssim'] is not None],
            key=lambda x: x[1],
            default=(None, None)
        )
        
        quality_analysis['best_configurations'] = {
            'best_mse': {'experiment': best_mse[0], 'score': best_mse[1]},
            'best_ssim': {'experiment': best_ssim[0], 'score': best_ssim[1]}
        }
    
    # Save analysis results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        analysis_path = os.path.join(save_dir, f'{dataset_name}_reconstruction_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(quality_analysis, f, indent=2, default=str)
        print(f"💾 Reconstruction analysis saved to: {analysis_path}")
    
    return quality_analysis


def generate_reconstruction_comparison_report(
    results_dict: Dict[str, Dict],
    dataset_name: str = "dataset",
    save_dir: Optional[str] = None,
    max_comparisons: int = 4
) -> bool:
    """
    Generate comprehensive reconstruction comparison visualizations.
    
    Args:
        results_dict: Dictionary containing experiment results
        dataset_name: Name of the dataset
        save_dir: Directory to save visualizations
        max_comparisons: Maximum number of experiments to compare
        
    Returns:
        Boolean indicating success
    """
    print("📊 Generating reconstruction comparison visualizations...")
    
    try:
        # Identify experiments with reconstruction data
        experiments_with_reconstructions = {
            name: results for name, results in results_dict.items()
            if 'reconstructions' in results and results['reconstructions']
        }
        
        if not experiments_with_reconstructions:
            print("❌ No reconstruction data found in experiment results")
            return False
        
        # Limit to max comparisons
        exp_names = list(experiments_with_reconstructions.keys())[:max_comparisons]
        
        # Generate comparison for each experiment
        for exp_name in exp_names:
            exp_results = experiments_with_reconstructions[exp_name]
            recon_data = exp_results['reconstructions']
            
            # Prepare save path
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{dataset_name}_{exp_name}_reconstruction_comparison.png')
            
            # Use core visualization function
            plot_reconstruction_comparison(
                original_images=recon_data.get('original_images', []),
                reconstructed_images=recon_data.get('reconstructed_images', []),
                class_names=recon_data.get('class_names', []),
                mse_scores=recon_data.get('mse_scores', []),
                ssim_scores=recon_data.get('ssim_scores', []),
                experiment_name=exp_name,
                save_path=save_path
            )
        
        print(f"✅ Generated reconstruction comparisons for {len(exp_names)} experiments")
        return True
        
    except Exception as e:
        print(f"❌ Error generating reconstruction comparison report: {e}")
        return False


def create_reconstruction_visualization_batch(
    results_dict: Dict[str, Dict],
    output_dir: str,
    dataset_name: str = "dataset"
) -> Dict[str, str]:
    """
    Create reconstruction visualizations for all experiments in batch.
    
    Args:
        results_dict: Dictionary containing experiment results
        output_dir: Directory for output files
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary mapping experiment names to visualization paths
    """
    print("🎨 Creating reconstruction visualization batch...")
    
    visualization_paths = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for exp_name, exp_results in results_dict.items():
        if 'reconstructions' not in exp_results:
            continue
            
        recon_data = exp_results['reconstructions']
        
        # Create visualization for this experiment
        viz_path = os.path.join(output_dir, f'{dataset_name}_{exp_name}_reconstructions.png')
        
        success = plot_reconstruction_comparison(
            original_images=recon_data.get('original_images', []),
            reconstructed_images=recon_data.get('reconstructed_images', []),
            class_names=recon_data.get('class_names', []),
            mse_scores=recon_data.get('mse_scores', []),
            ssim_scores=recon_data.get('ssim_scores', []),
            experiment_name=exp_name,
            save_path=viz_path,
            show_plot=False
        )
        
        if success:
            visualization_paths[exp_name] = viz_path
    
    print(f"✅ Created {len(visualization_paths)} reconstruction visualizations")
    return visualization_paths


def analyze_hyperparameter_sensitivity(
    results_dict: Dict[str, Dict],
    target_metric: str = 'final_test_loss',
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze sensitivity to different hyperparameters.
    
    Args:
        results_dict: Dictionary containing experiment results
        target_metric: Metric to analyze sensitivity for
        save_path: Path to save analysis results
        
    Returns:
        Dictionary containing sensitivity analysis
    """
    print(f"🔬 Analyzing hyperparameter sensitivity for {target_metric}...")
    
    # Extract hyperparameter configurations and corresponding metric values
    configurations = []
    metric_values = []
    
    for exp_name, exp_data in results_dict.items():
        config = exp_data.get('config', {})
        metric_value = exp_data.get(target_metric)
        
        if metric_value is not None:
            configurations.append(config)
            metric_values.append(metric_value)
    
    if not configurations:
        print(f"❌ No data found for metric: {target_metric}")
        return {}
    
    # Analyze parameter importance
    sensitivity_analysis = {
        'parameter_ranges': {},
        'parameter_correlations': {},
        'optimal_ranges': {}
    }
    
    # Get unique parameter names
    all_params = set()
    for config in configurations:
        all_params.update(config.keys())
    
    for param in all_params:
        param_values = [config.get(param) for config in configurations]
        
        # Skip if parameter has only one unique value or contains non-numeric data
        unique_values = list(set([v for v in param_values if v is not None]))
        if len(unique_values) <= 1:
            continue
            
        # Try to analyze numeric parameters
        try:
            numeric_values = [float(v) for v in param_values if v is not None]
            if len(numeric_values) != len(param_values):
                continue  # Skip mixed or non-numeric parameters
                
            # Calculate correlation with target metric
            if len(numeric_values) > 1 and len(set(numeric_values)) > 1:
                correlation = np.corrcoef(numeric_values, metric_values)[0, 1]
                
                sensitivity_analysis['parameter_correlations'][param] = {
                    'correlation': correlation,
                    'parameter_range': (min(numeric_values), max(numeric_values)),
                    'metric_range': (min(metric_values), max(metric_values))
                }
                
                # Find optimal range (values that produce best results)
                if 'loss' in target_metric.lower():
                    # Lower is better
                    best_indices = np.argsort(metric_values)[:len(metric_values)//3]
                else:
                    # Higher is better
                    best_indices = np.argsort(metric_values)[-len(metric_values)//3:]
                
                optimal_param_values = [numeric_values[i] for i in best_indices]
                sensitivity_analysis['optimal_ranges'][param] = {
                    'optimal_min': min(optimal_param_values),
                    'optimal_max': max(optimal_param_values),
                    'optimal_mean': np.mean(optimal_param_values)
                }
                
        except (ValueError, TypeError):
            # Handle categorical parameters
            value_performance = {}
            for i, value in enumerate(param_values):
                if value is not None:
                    if value not in value_performance:
                        value_performance[value] = []
                    value_performance[value].append(metric_values[i])
            
            # Calculate average performance for each categorical value
            avg_performance = {
                value: np.mean(performances) 
                for value, performances in value_performance.items()
            }
            
            sensitivity_analysis['parameter_ranges'][param] = {
                'categorical_performance': avg_performance,
                'best_value': min(avg_performance.items(), key=lambda x: x[1])[0] 
                           if 'loss' in target_metric.lower() else 
                           max(avg_performance.items(), key=lambda x: x[1])[0]
            }
    
    # Save analysis if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(sensitivity_analysis, f, indent=2, default=str)
        print(f"💾 Sensitivity analysis saved to: {save_path}")
    
    return sensitivity_analysis


def identify_optimal_configurations(
    results_dict: Dict[str, Dict],
    target_metrics: List[str] = ['final_test_loss', 'test_silhouette'],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identify optimal configurations based on multiple metrics.
    
    Args:
        results_dict: Dictionary containing experiment results
        target_metrics: List of metrics to consider for optimization
        save_path: Path to save optimization results
        
    Returns:
        Dictionary containing optimal configuration analysis
    """
    print(f"🎯 Identifying optimal configurations based on: {target_metrics}")
    
    # Extract configurations and metric values
    configurations = []
    metric_data = {metric: [] for metric in target_metrics}
    experiment_names = []
    
    for exp_name, exp_data in results_dict.items():
        config = exp_data.get('config', {})
        
        # Check if all target metrics are available
        metric_values = {}
        valid_config = True
        
        for metric in target_metrics:
            value = exp_data.get(metric)
            if value is not None:
                metric_values[metric] = value
            else:
                valid_config = False
                break
        
        if valid_config:
            configurations.append(config)
            experiment_names.append(exp_name)
            for metric, value in metric_values.items():
                metric_data[metric].append(value)
    
    if not configurations:
        print("❌ No configurations with all required metrics found")
        return {}
    
    # Normalize metrics (0-1 scale) and combine
    normalized_metrics = {}
    for metric in target_metrics:
        values = metric_data[metric]
        min_val, max_val = min(values), max(values)
        
        if max_val == min_val:
            normalized_metrics[metric] = [0.5] * len(values)  # All equal
        else:
            if 'loss' in metric.lower() or 'time' in metric.lower():
                # Lower is better - invert normalization
                normalized_metrics[metric] = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                # Higher is better
                normalized_metrics[metric] = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Calculate composite scores (equal weighting)
    composite_scores = []
    for i in range(len(configurations)):
        score = np.mean([normalized_metrics[metric][i] for metric in target_metrics])
        composite_scores.append(score)
    
    # Identify top configurations
    top_indices = np.argsort(composite_scores)[-5:][::-1]  # Top 5
    
    optimal_analysis = {
        'top_configurations': [],
        'metric_statistics': {},
        'configuration_rankings': {}
    }
    
    # Store top configurations
    for rank, idx in enumerate(top_indices):
        config_info = {
            'rank': rank + 1,
            'experiment_name': experiment_names[idx],
            'configuration': configurations[idx],
            'composite_score': composite_scores[idx],
            'metrics': {metric: metric_data[metric][idx] for metric in target_metrics},
            'normalized_metrics': {metric: normalized_metrics[metric][idx] for metric in target_metrics}
        }
        optimal_analysis['top_configurations'].append(config_info)
    
    # Calculate metric statistics
    for metric in target_metrics:
        values = metric_data[metric]
        optimal_analysis['metric_statistics'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'best_experiment': experiment_names[np.argmin(values) if 'loss' in metric.lower() else np.argmax(values)]
        }
    
    # Store all rankings
    sorted_indices = np.argsort(composite_scores)[::-1]
    for rank, idx in enumerate(sorted_indices):
        optimal_analysis['configuration_rankings'][experiment_names[idx]] = {
            'rank': rank + 1,
            'composite_score': composite_scores[idx]
        }
    
    # Save analysis if path provided
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(optimal_analysis, f, indent=2, default=str)
        print(f"💾 Optimal configuration analysis saved to: {save_path}")
    
    return optimal_analysis 
"""
Comprehensive visualization functions for autoencoder experiment results.

Provides visualization capabilities that match the output of the original
AutoEncoderJupyterTest.ipynb notebook, including loss curves, performance
comparisons, heatmaps, and analysis tables.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from matplotlib.colors import LogNorm
import csv
from datetime import datetime


def plot_loss_curves(experiment_results: Dict[str, Any], 
                    save_dir: Optional[str] = None,
                    show_plots: bool = True) -> None:
    """
    Plot training and validation loss curves for an experiment.
    
    Args:
        experiment_results: Results dictionary from run_single_experiment
        save_dir: Directory to save plots (optional)
        show_plots: Whether to display plots
    """
    history = experiment_results['history']
    experiment_name = experiment_results['experiment_name']
    
    # Extract loss data
    train_losses = history.get('train_loss', [])
    test_losses = history.get('test_loss', [])
    
    if not train_losses and not test_losses:
        print(f"No loss data available for {experiment_name}")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if train_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    # Plot test loss
    if test_losses:
        epochs = range(1, len(test_losses) + 1)
        plt.plot(epochs, test_losses, 'r--', label='Test Loss', linewidth=2)
    
    plt.title(f'Loss Curves: {experiment_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot if directory specified
    if save_dir:
        save_path = Path(save_dir) / f"{experiment_name}_loss_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_metrics_vs_latent_dim(systematic_results: Dict[str, List[Dict[str, Any]]],
                              save_dir: Optional[str] = None,
                              show_plots: bool = True) -> None:
    """
    Plot metrics (test/train loss and silhouette scores) vs latent dimension.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    architectures = list(systematic_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = [
        ('final_train_loss', 'Train Loss', axes[0, 0], '-'),
        ('final_test_loss', 'Test Loss', axes[0, 1], '--'),
        ('final_train_silhouette', 'Train Silhouette Score', axes[1, 0], '-'),
        ('final_silhouette', 'Test Silhouette Score', axes[1, 1], '--')
    ]
    
    for i, (architecture, results) in enumerate(systematic_results.items()):
        # Sort results by latent dimension
        sorted_results = sorted(results, key=lambda x: x['latent_dim'])
        
        for metric_key, metric_label, ax, linestyle in metrics:
            latent_dims = []
            metric_values = []
            
            for result in sorted_results:
                if metric_key in result['metrics']:
                    latent_dims.append(result['latent_dim'])
                    metric_values.append(result['metrics'][metric_key])
            
            if latent_dims and metric_values:
                ax.plot(latent_dims, metric_values, 'o-',
                       label=architecture,
                       color=colors[i],
                       linestyle=linestyle)
                
                ax.set_title(f'{metric_label} vs. Latent Dimension')
                ax.set_xlabel('Latent Dimension')
                ax.set_ylabel(metric_label)
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log', base=2)
                ax.legend()
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.suptitle(f"Metrics Comparison - {timestamp}", fontsize=14)
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        save_path = Path(save_dir) / f'metrics_vs_latent_dim_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_performance_heatmaps(systematic_results: Dict[str, List[Dict[str, Any]]],
                               save_dir: Optional[str] = None,
                               show_plots: bool = True) -> None:
    """
    Create heatmaps showing performance metrics across architectures and latent dimensions.
    
    Args:
        systematic_results: Results from run_systematic_experiments
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    # Get all unique latent dimensions
    all_dims = set()
    for results in systematic_results.values():
        for result in results:
            all_dims.add(result['latent_dim'])
    
    all_dims = sorted(list(all_dims))
    architectures = list(systematic_results.keys())
    
    # Prepare data for heatmaps
    test_loss_data = np.zeros((len(architectures), len(all_dims)))
    silhouette_data = np.zeros((len(architectures), len(all_dims)))
    
    # Fill in the data
    for i, architecture in enumerate(architectures):
        results = systematic_results[architecture]
        for result in results:
            latent_dim = result['latent_dim']
            dim_idx = all_dims.index(latent_dim)
            
            test_loss_data[i, dim_idx] = result['metrics'].get('final_test_loss', 0)
            silhouette_data[i, dim_idx] = result['metrics'].get('final_silhouette', 0)
    
    # Create heatmap figures
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Test Loss Heatmap
    im1 = axes[0].imshow(test_loss_data, cmap='viridis_r')
    axes[0].set_title('Test Loss across Architectures and Latent Dimensions')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Architecture')
    axes[0].set_xticks(np.arange(len(all_dims)))
    axes[0].set_yticks(np.arange(len(architectures)))
    axes[0].set_xticklabels(all_dims)
    axes[0].set_yticklabels(architectures)
    
    cbar1 = fig.colorbar(im1, ax=axes[0])
    cbar1.set_label('Test Loss (lower is better)')
    
    # Add values to heatmap
    for i in range(len(architectures)):
        for j in range(len(all_dims)):
            text = axes[0].text(j, i, f"{test_loss_data[i, j]:.4f}",
                               ha="center", va="center", 
                               color="w" if test_loss_data[i, j] > np.mean(test_loss_data) else "black")
    
    # Silhouette Score Heatmap
    im2 = axes[1].imshow(silhouette_data, cmap='viridis')
    axes[1].set_title('Silhouette Score across Architectures and Latent Dimensions')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Architecture')
    axes[1].set_xticks(np.arange(len(all_dims)))
    axes[1].set_yticks(np.arange(len(architectures)))
    axes[1].set_xticklabels(all_dims)
    axes[1].set_yticklabels(architectures)
    
    cbar2 = fig.colorbar(im2, ax=axes[1])
    cbar2.set_label('Silhouette Score (higher is better)')
    
    # Add values to heatmap
    for i in range(len(architectures)):
        for j in range(len(all_dims)):
            text = axes[1].text(j, i, f"{silhouette_data[i, j]:.4f}",
                               ha="center", va="center",
                               color="w" if silhouette_data[i, j] < np.mean(silhouette_data) else "black")
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.suptitle(f"Performance Heatmaps - {timestamp}", fontsize=14)
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        save_path = Path(save_dir) / f'performance_heatmaps_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmaps saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


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


def plot_architecture_comparison(systematic_results: Dict[str, List[Dict[str, Any]]],
                                save_dir: Optional[str] = None,
                                show_plots: bool = True) -> None:
    """
    Create plots comparing different architectures across latent dimensions.
    
    Args:
        systematic_results: Results from run_systematic_experiments  
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    architectures = list(systematic_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(architectures)))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot test loss vs latent dimension
    plt.subplot(2, 1, 1)
    for i, (architecture, results) in enumerate(systematic_results.items()):
        latent_dims = []
        test_losses = []
        
        sorted_results = sorted(results, key=lambda x: x['latent_dim'])
        for result in sorted_results:
            latent_dims.append(result['latent_dim'])
            test_losses.append(result['metrics'].get('final_test_loss', 0))
        
        plt.plot(latent_dims, test_losses, 'o-',
                label=architecture,
                color=colors[i],
                linestyle='--')
    
    plt.title('Test Loss vs. Latent Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Final Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Plot silhouette score vs latent dimension
    plt.subplot(2, 1, 2)
    for i, (architecture, results) in enumerate(systematic_results.items()):
        latent_dims = []
        silhouette_scores = []
        
        sorted_results = sorted(results, key=lambda x: x['latent_dim'])
        for result in sorted_results:
            latent_dims.append(result['latent_dim'])
            silhouette_scores.append(result['metrics'].get('final_silhouette', 0))
        
        plt.plot(latent_dims, silhouette_scores, 'o-',
                label=architecture,
                color=colors[i],
                linestyle='--')
    
    plt.title('Cluster Separation (Silhouette Score) vs. Latent Dimension')
    plt.xlabel('Latent Dimension')  
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.suptitle(f"Architecture Comparison - {timestamp}", fontsize=12)
    plt.tight_layout()
    
    # Save plot
    if save_dir:
        save_path = Path(save_dir) / f'architecture_comparison_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture comparison saved to: {save_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def generate_comprehensive_report(systematic_results: Dict[str, List[Dict[str, Any]]],
                                 output_dir: str,
                                 show_plots: bool = True) -> Dict[str, str]:
    """
    Generate a comprehensive visualization report matching AutoEncoderJupyterTest output.
    
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
    
    # 1. Plot metrics vs latent dimensions
    print("📊 Creating metrics comparison plots...")
    plot_metrics_vs_latent_dim(systematic_results, save_dir=output_dir, show_plots=show_plots)
    
    # 2. Create performance heatmaps
    print("🔥 Creating performance heatmaps...")
    create_performance_heatmaps(systematic_results, save_dir=output_dir, show_plots=show_plots)
    
    # 3. Plot architecture comparison
    print("🏗️ Creating architecture comparison plots...")
    plot_architecture_comparison(systematic_results, save_dir=output_dir, show_plots=show_plots)
    
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
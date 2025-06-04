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

# Import core visualization functions that we'll orchestrate
from ..visualization import (
    plot_training_curves,
    plot_performance_grid,
    plot_latent_dimension_analysis
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
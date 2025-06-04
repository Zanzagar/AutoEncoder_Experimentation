"""
Latent Space Analysis Integration for Experiments

This module provides high-level functions that orchestrate latent space analysis
using the existing core visualization functions from autoencoder_lib.visualization.
These functions are designed to integrate with the systematic experiment framework.
"""

import numpy as np
import torch
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from pathlib import Path

# Import core visualization functions
from autoencoder_lib.visualization import (
    visualize_latent_space_2d,
    compare_latent_spaces,
    plot_latent_distribution,
    plot_latent_interpolation,
    analyze_latent_clustering,
    plot_latent_variance_analysis,
    visualize_side_by_side_latent_spaces
)


def analyze_latent_space(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    device: torch.device = torch.device('cpu'),
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract latent representations and perform basic analysis.
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels tensor
        test_data: Test data tensor
        test_labels: Test labels tensor
        class_names: Names for each class
        device: Device to run computations on
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing latent representations and analysis results
    """
    print("🧠 Analyzing latent space representations...")
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Extract latent representations
        print("   Extracting training set latent representations...")
        train_latent = []
        batch_size = 100  # Process in batches to avoid memory issues
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size].to(device)
            if hasattr(model, 'encoder'):
                latent_batch = model.encoder(batch)
            else:
                # For models without explicit encoder attribute
                latent_batch = model.encode(batch) if hasattr(model, 'encode') else model(batch)[1]
            train_latent.append(latent_batch.cpu())
        
        train_latent = torch.cat(train_latent, dim=0)
        
        print("   Extracting test set latent representations...")
        test_latent = []
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size].to(device)
            if hasattr(model, 'encoder'):
                latent_batch = model.encoder(batch)
            else:
                latent_batch = model.encode(batch) if hasattr(model, 'encode') else model(batch)[1]
            test_latent.append(latent_batch.cpu())
        
        test_latent = torch.cat(test_latent, dim=0)
    
    # Prepare analysis results
    analysis_results = {
        'train_latent': train_latent.numpy(),
        'test_latent': test_latent.numpy(),
        'train_labels': train_labels.numpy(),
        'test_labels': test_labels.numpy(),
        'latent_dim': train_latent.shape[1],
        'num_train_samples': len(train_latent),
        'num_test_samples': len(test_latent),
        'class_names': class_names
    }
    
    print(f"   ✅ Extracted latent representations: {train_latent.shape[1]}D space")
    print(f"   📊 Train samples: {len(train_latent)}, Test samples: {len(test_latent)}")
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Save latent representations (without numpy arrays for JSON)
        metadata = {k: v for k, v in analysis_results.items() 
                   if not isinstance(v, np.ndarray)}
        
        with open(os.path.join(output_dir, 'latent_analysis_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save numpy arrays separately
        np.savez(os.path.join(output_dir, 'latent_representations.npz'),
                train_latent=analysis_results['train_latent'],
                test_latent=analysis_results['test_latent'],
                train_labels=analysis_results['train_labels'],
                test_labels=analysis_results['test_labels'])
        
        print(f"   💾 Saved latent analysis to {output_dir}")
    
    return analysis_results


def create_latent_tsne_analysis(
    analysis_results: Dict[str, Any],
    output_dir: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Generate comprehensive t-SNE analysis of latent spaces.
    
    Args:
        analysis_results: Results from analyze_latent_space()
        output_dir: Directory to save visualizations
        random_state: Random seed for reproducibility (note: function uses fixed seed)
        
    Returns:
        Dictionary containing t-SNE quality metrics
    """
    print("🔬 Creating latent space t-SNE analysis...")
    
    train_latent = analysis_results['train_latent']
    test_latent = analysis_results['test_latent']
    train_labels = analysis_results['train_labels']
    test_labels = analysis_results['test_labels']
    class_names = analysis_results.get('class_names')
    
    # Generate side-by-side latent space comparison using latent data directly
    print("   Generating side-by-side t-SNE visualization...")
    
    # Set save paths if output directory specified
    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'latent_tsne_comparison.png')
    
    # Use the latent space visualization function directly
    # Train t-SNE
    _, train_silhouette = visualize_latent_space_2d(
        latent_vectors=train_latent,
        labels=train_labels,
        class_names=class_names,
        method='tsne',
        title='Training Set - Latent Space t-SNE',
        save_path=save_path.replace('.png', '_train.png') if save_path else None
    )
    
    # Test t-SNE
    _, test_silhouette = visualize_latent_space_2d(
        latent_vectors=test_latent,
        labels=test_labels,
        class_names=class_names,
        method='tsne',
        title='Test Set - Latent Space t-SNE',
        save_path=save_path.replace('.png', '_test.png') if save_path else None
    )
    
    # Generate individual t-SNE visualizations for more detailed analysis
    print("   Generating detailed t-SNE analyses...")
    
    if output_dir:
        # Training set detailed analysis
        train_save_path = os.path.join(output_dir, 'latent_tsne_train_detailed.png')
        _, train_detailed_silhouette = visualize_latent_space_2d(
            latent_vectors=train_latent,
            labels=train_labels,
            class_names=class_names,
            method='tsne',
            title='Training Set - Latent Space t-SNE Analysis',
            save_path=train_save_path
        )
        
        # Test set detailed analysis
        test_save_path = os.path.join(output_dir, 'latent_tsne_test_detailed.png')
        _, test_detailed_silhouette = visualize_latent_space_2d(
            latent_vectors=test_latent,
            labels=test_labels,
            class_names=class_names,
            method='tsne',
            title='Test Set - Latent Space t-SNE Analysis',
            save_path=test_save_path
        )
    else:
        # Just calculate silhouettes without saving
        _, train_detailed_silhouette = visualize_latent_space_2d(
            latent_vectors=train_latent,
            labels=train_labels,
            class_names=class_names,
            method='tsne',
            title='Training Set - Latent Space t-SNE Analysis'
        )
        
        _, test_detailed_silhouette = visualize_latent_space_2d(
            latent_vectors=test_latent,
            labels=test_labels,
            class_names=class_names,
            method='tsne',
            title='Test Set - Latent Space t-SNE Analysis'
        )
    
    # Compile results
    tsne_results = {
        'train_silhouette_score': float(train_silhouette),
        'test_silhouette_score': float(test_silhouette),
        'train_detailed_silhouette': float(train_detailed_silhouette),
        'test_detailed_silhouette': float(test_detailed_silhouette),
        'mean_silhouette_score': float((train_silhouette + test_silhouette) / 2)
    }
    
    print(f"   📈 t-SNE Quality Metrics:")
    print(f"     • Train Silhouette: {train_silhouette:.4f}")
    print(f"     • Test Silhouette: {test_silhouette:.4f}")
    print(f"     • Mean Silhouette: {tsne_results['mean_silhouette_score']:.4f}")
    
    # Save results
    if output_dir:
        with open(os.path.join(output_dir, 'tsne_analysis_results.json'), 'w') as f:
            json.dump(tsne_results, f, indent=2)
        print(f"   💾 Saved t-SNE analysis results to {output_dir}")
    
    return tsne_results


def perform_latent_clustering(
    analysis_results: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform clustering analysis on latent space representations.
    
    Args:
        analysis_results: Results from analyze_latent_space()
        output_dir: Directory to save clustering results
        
    Returns:
        Dictionary containing clustering analysis results
    """
    print("🎯 Performing latent space clustering analysis...")
    
    train_latent = analysis_results['train_latent']
    test_latent = analysis_results['test_latent']
    train_labels = analysis_results['train_labels']
    test_labels = analysis_results['test_labels']
    class_names = analysis_results.get('class_names')
    
    # Analyze training set clustering
    print("   Analyzing training set clustering...")
    train_clustering_results = analyze_latent_clustering(
        latent_vectors=train_latent,
        labels=train_labels,
        class_names=class_names,
        figure_size=(15, 10)
    )
    
    # Analyze test set clustering  
    print("   Analyzing test set clustering...")
    test_clustering_results = analyze_latent_clustering(
        latent_vectors=test_latent,
        labels=test_labels,
        class_names=class_names,
        figure_size=(15, 10)
    )
    
    # Combine results
    clustering_results = {
        'train_clustering': train_clustering_results,
        'test_clustering': test_clustering_results,
        'analysis_summary': {
            'train_best_silhouette': float(train_clustering_results.get('best_silhouette_score', 0)),
            'test_best_silhouette': float(test_clustering_results.get('best_silhouette_score', 0)),
            'train_optimal_clusters': int(train_clustering_results.get('optimal_num_clusters', 0)),
            'test_optimal_clusters': int(test_clustering_results.get('optimal_num_clusters', 0))
        }
    }
    
    print(f"   📊 Clustering Analysis Results:")
    print(f"     • Train - Optimal clusters: {clustering_results['analysis_summary']['train_optimal_clusters']}")
    print(f"     • Train - Best silhouette: {clustering_results['analysis_summary']['train_best_silhouette']:.4f}")
    print(f"     • Test - Optimal clusters: {clustering_results['analysis_summary']['test_optimal_clusters']}")
    print(f"     • Test - Best silhouette: {clustering_results['analysis_summary']['test_best_silhouette']:.4f}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'clustering_analysis_results.json'), 'w') as f:
            json.dump(clustering_results, f, indent=2)
        print(f"   💾 Saved clustering analysis to {output_dir}")
    
    return clustering_results


def generate_latent_interpolations(
    model: torch.nn.Module,
    analysis_results: Dict[str, Any],
    num_interpolations: int = 5,
    num_steps: int = 8,
    device: torch.device = torch.device('cpu'),
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate latent space interpolations between different class examples.
    
    Args:
        model: Trained autoencoder model  
        analysis_results: Results from analyze_latent_space()
        num_interpolations: Number of interpolation pairs to generate
        num_steps: Number of steps in each interpolation
        device: Device to run computations on
        output_dir: Directory to save interpolation visualizations
        
    Returns:
        Dictionary containing interpolation analysis results
    """
    print("🌉 Generating latent space interpolations...")
    
    train_latent = torch.tensor(analysis_results['train_latent'])
    train_labels = analysis_results['train_labels']
    class_names = analysis_results.get('class_names')
    unique_classes = np.unique(train_labels)
    
    model.eval()
    model = model.to(device)
    
    interpolation_results = []
    
    # Generate interpolations between different classes
    for i in range(min(num_interpolations, len(unique_classes))):
        for j in range(i + 1, len(unique_classes)):
            class1, class2 = unique_classes[i], unique_classes[j]
            
            # Find examples from each class
            class1_indices = np.where(train_labels == class1)[0]
            class2_indices = np.where(train_labels == class2)[0]
            
            if len(class1_indices) > 0 and len(class2_indices) > 0:
                # Select random examples
                idx1 = np.random.choice(class1_indices)
                idx2 = np.random.choice(class2_indices)
                
                start_latent = train_latent[idx1]
                end_latent = train_latent[idx2]
                
                class1_name = class_names[class1] if class_names else f"Class {class1}"
                class2_name = class_names[class2] if class_names else f"Class {class2}"
                
                print(f"   Interpolating between {class1_name} and {class2_name}...")
                
                # Generate interpolation visualization
                plot_latent_interpolation(
                    model=model,
                    start_latent=start_latent,
                    end_latent=end_latent,
                    num_steps=num_steps,
                    device=device,
                    figure_size=(16, 4)
                )
                
                interpolation_results.append({
                    'class1': int(class1),
                    'class2': int(class2),
                    'class1_name': class1_name,
                    'class2_name': class2_name,
                    'start_latent_idx': int(idx1),
                    'end_latent_idx': int(idx2)
                })
    
    print(f"   ✅ Generated {len(interpolation_results)} interpolation visualizations")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'interpolation_results.json'), 'w') as f:
            json.dump({'interpolations': interpolation_results}, f, indent=2)
        print(f"   💾 Saved interpolation analysis to {output_dir}")
    
    return {'interpolations': interpolation_results}


def analyze_latent_traversals(
    model: torch.nn.Module,
    analysis_results: Dict[str, Any],
    num_traversals: int = 6,
    traversal_range: float = 3.0,
    num_steps: int = 9,
    device: torch.device = torch.device('cpu'),
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze individual latent dimensions through systematic traversal.
    
    Args:
        model: Trained autoencoder model
        analysis_results: Results from analyze_latent_space()
        num_traversals: Number of latent dimensions to traverse
        traversal_range: Range of values to traverse (+/- range)
        num_steps: Number of steps in traversal
        device: Device to run computations on
        output_dir: Directory to save traversal visualizations
        
    Returns:
        Dictionary containing traversal analysis results
    """
    print("🎭 Analyzing latent dimension traversals...")
    
    train_latent = torch.tensor(analysis_results['train_latent'])
    latent_dim = analysis_results['latent_dim']
    
    model.eval()
    model = model.to(device)
    
    # Select a representative latent vector (mean of training set)
    base_latent = torch.mean(train_latent, dim=0).to(device)
    
    # Select dimensions to traverse
    dims_to_traverse = min(num_traversals, latent_dim)
    traversal_results = []
    
    with torch.no_grad():
        for dim in range(dims_to_traverse):
            print(f"   Traversing latent dimension {dim}...")
            
            # Create traversal sequence
            traversal_values = torch.linspace(-traversal_range, traversal_range, num_steps)
            traversal_latents = []
            
            for value in traversal_values:
                modified_latent = base_latent.clone()
                modified_latent[dim] = value
                traversal_latents.append(modified_latent)
            
            traversal_latents = torch.stack(traversal_latents)
            
            # Generate reconstructions
            if hasattr(model, 'decoder'):
                reconstructions = model.decoder(traversal_latents)
            else:
                reconstructions = model.decode(traversal_latents) if hasattr(model, 'decode') else model(traversal_latents)
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, num_steps, figsize=(16, 3))
            if num_steps == 1:
                axes = [axes]
            
            for i, (value, reconstruction) in enumerate(zip(traversal_values, reconstructions)):
                ax = axes[i]
                img = reconstruction.cpu().squeeze()
                
                if len(img.shape) == 3:  # Color image
                    img = img.permute(1, 2, 0)
                
                ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                ax.set_title(f'{value:.2f}', fontsize=10)
                ax.axis('off')
            
            plt.suptitle(f'Latent Dimension {dim} Traversal', fontsize=14)
            plt.tight_layout()
            
            if output_dir:
                save_path = os.path.join(output_dir, f'latent_traversal_dim_{dim}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            plt.show()
            
            traversal_results.append({
                'dimension': int(dim),
                'traversal_range': float(traversal_range),
                'num_steps': int(num_steps),
                'base_latent_value': float(base_latent[dim])
            })
    
    print(f"   ✅ Analyzed {len(traversal_results)} latent dimension traversals")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'traversal_results.json'), 'w') as f:
            json.dump({'traversals': traversal_results}, f, indent=2)
        print(f"   💾 Saved traversal analysis to {output_dir}")
    
    return {'traversals': traversal_results}


def calculate_latent_metrics(
    analysis_results: Dict[str, Any],
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive latent space quality metrics.
    
    Args:
        analysis_results: Results from analyze_latent_space()
        output_dir: Directory to save metric analysis
        
    Returns:
        Dictionary containing comprehensive latent space metrics
    """
    print("📏 Calculating comprehensive latent space metrics...")
    
    train_latent = analysis_results['train_latent']
    test_latent = analysis_results['test_latent']
    train_labels = analysis_results['train_labels']
    test_labels = analysis_results['test_labels']
    class_names = analysis_results.get('class_names')
    
    # Generate latent distribution analysis
    print("   Analyzing latent dimension distributions...")
    plot_latent_distribution(
        latent_vectors=train_latent,
        labels=train_labels,
        class_names=class_names,
        max_dims=6,
        figure_size=(15, 10)
    )
    
    # Generate variance analysis
    print("   Analyzing latent dimension variance...")
    plot_latent_variance_analysis(
        latent_vectors=train_latent,
        labels=train_labels,
        class_names=class_names,
        figure_size=(15, 8)
    )
    
    # Calculate statistical metrics
    train_stats = {
        'mean': np.mean(train_latent, axis=0).tolist(),
        'std': np.std(train_latent, axis=0).tolist(),
        'min': np.min(train_latent, axis=0).tolist(),
        'max': np.max(train_latent, axis=0).tolist()
    }
    
    test_stats = {
        'mean': np.mean(test_latent, axis=0).tolist(),
        'std': np.std(test_latent, axis=0).tolist(),
        'min': np.min(test_latent, axis=0).tolist(),
        'max': np.max(test_latent, axis=0).tolist()
    }
    
    # Calculate class separability metrics
    class_separability = {}
    unique_classes = np.unique(train_labels)
    
    for i, class_id in enumerate(unique_classes):
        class_mask = train_labels == class_id
        class_latents = train_latent[class_mask]
        
        if len(class_latents) > 0:
            class_name = class_names[class_id] if class_names else f"Class {class_id}"
            class_separability[class_name] = {
                'num_samples': int(len(class_latents)),
                'mean_distance_to_origin': float(np.mean(np.linalg.norm(class_latents, axis=1))),
                'std_distance_to_origin': float(np.std(np.linalg.norm(class_latents, axis=1))),
                'intra_class_variance': float(np.mean(np.var(class_latents, axis=0)))
            }
    
    # Compile comprehensive metrics
    comprehensive_metrics = {
        'latent_dimension': int(analysis_results['latent_dim']),
        'num_train_samples': int(analysis_results['num_train_samples']),
        'num_test_samples': int(analysis_results['num_test_samples']),
        'train_statistics': train_stats,
        'test_statistics': test_stats,
        'class_separability': class_separability,
        'overall_metrics': {
            'train_latent_variance': float(np.mean(np.var(train_latent, axis=0))),
            'test_latent_variance': float(np.mean(np.var(test_latent, axis=0))),
            'train_mean_norm': float(np.mean(np.linalg.norm(train_latent, axis=1))),
            'test_mean_norm': float(np.mean(np.linalg.norm(test_latent, axis=1)))
        }
    }
    
    print(f"   📊 Calculated comprehensive latent space metrics")
    print(f"     • Mean latent variance (train): {comprehensive_metrics['overall_metrics']['train_latent_variance']:.4f}")
    print(f"     • Mean latent norm (train): {comprehensive_metrics['overall_metrics']['train_mean_norm']:.4f}")
    
    # Save comprehensive results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'comprehensive_latent_metrics.json'), 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        print(f"   💾 Saved comprehensive metrics to {output_dir}")
    
    return comprehensive_metrics


def run_complete_latent_analysis(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    device: torch.device = torch.device('cpu'),
    output_dir: Optional[str] = None,
    include_interpolations: bool = True,
    include_traversals: bool = True
) -> Dict[str, Any]:
    """
    Run complete latent space analysis pipeline.
    
    Args:
        model: Trained autoencoder model
        train_data: Training data tensor
        train_labels: Training labels tensor
        test_data: Test data tensor
        test_labels: Test labels tensor
        class_names: Names for each class
        device: Device to run computations on
        output_dir: Directory to save all analysis results
        include_interpolations: Whether to generate interpolation analysis
        include_traversals: Whether to generate traversal analysis
        
    Returns:
        Dictionary containing all analysis results
    """
    print("🎯 Running complete latent space analysis pipeline...")
    print("=" * 60)
    
    # Step 1: Extract latent representations
    analysis_results = analyze_latent_space(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        class_names=class_names,
        device=device,
        output_dir=output_dir
    )
    
    # Step 2: t-SNE analysis
    tsne_results = create_latent_tsne_analysis(
        analysis_results=analysis_results,
        output_dir=output_dir
    )
    
    # Step 3: Clustering analysis
    clustering_results = perform_latent_clustering(
        analysis_results=analysis_results,
        output_dir=output_dir
    )
    
    # Step 4: Comprehensive metrics
    metrics_results = calculate_latent_metrics(
        analysis_results=analysis_results,
        output_dir=output_dir
    )
    
    # Step 5: Interpolations (optional)
    interpolation_results = {}
    if include_interpolations:
        interpolation_results = generate_latent_interpolations(
            model=model,
            analysis_results=analysis_results,
            device=device,
            output_dir=output_dir
        )
    
    # Step 6: Traversals (optional)
    traversal_results = {}
    if include_traversals:
        traversal_results = analyze_latent_traversals(
            model=model,
            analysis_results=analysis_results,
            device=device,
            output_dir=output_dir
        )
    
    # Compile complete results
    complete_results = {
        'base_analysis': analysis_results,
        'tsne_analysis': tsne_results,
        'clustering_analysis': clustering_results,
        'comprehensive_metrics': metrics_results,
        'interpolation_analysis': interpolation_results,
        'traversal_analysis': traversal_results,
        'analysis_summary': {
            'latent_dimension': analysis_results['latent_dim'],
            'mean_silhouette_score': tsne_results['mean_silhouette_score'],
            'train_optimal_clusters': clustering_results['analysis_summary']['train_optimal_clusters'],
            'test_optimal_clusters': clustering_results['analysis_summary']['test_optimal_clusters'],
            'latent_variance': metrics_results['overall_metrics']['train_latent_variance'],
            'num_interpolations': len(interpolation_results.get('interpolations', [])),
            'num_traversals': len(traversal_results.get('traversals', []))
        }
    }
    
    # Save master results file
    if output_dir:
        with open(os.path.join(output_dir, 'complete_latent_analysis.json'), 'w') as f:
            # Create a JSON-serializable version
            json_results = {k: v for k, v in complete_results.items() 
                           if k != 'base_analysis'}  # Skip numpy arrays
            json_results['analysis_metadata'] = {
                'latent_dimension': complete_results['base_analysis']['latent_dim'],
                'num_train_samples': complete_results['base_analysis']['num_train_samples'],
                'num_test_samples': complete_results['base_analysis']['num_test_samples']
            }
            json.dump(json_results, f, indent=2)
    
    print("=" * 60)
    print("🎉 Complete latent space analysis finished!")
    print(f"   📊 Key Results:")
    summary = complete_results['analysis_summary']
    print(f"     • Latent Dimension: {summary['latent_dimension']}")
    print(f"     • Mean Silhouette Score: {summary['mean_silhouette_score']:.4f}")
    print(f"     • Optimal Clusters (Train/Test): {summary['train_optimal_clusters']}/{summary['test_optimal_clusters']}")
    print(f"     • Latent Variance: {summary['latent_variance']:.4f}")
    if include_interpolations:
        print(f"     • Interpolations Generated: {summary['num_interpolations']}")
    if include_traversals:
        print(f"     • Traversals Analyzed: {summary['num_traversals']}")
    
    return complete_results 
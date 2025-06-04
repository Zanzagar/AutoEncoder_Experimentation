#!/usr/bin/env python3
"""
Simple test for latent analysis tools integration using existing dataset.
"""

import torch
import numpy as np
from autoencoder_lib.experiment.wrappers import run_single_experiment
from autoencoder_lib.experiment.latent_analysis import run_complete_latent_analysis

def test_latent_analysis_with_existing_data():
    print("=" * 80)
    print("🔬 TESTING LATENT ANALYSIS WITH EXISTING DATASET")
    print("=" * 80)
    
    # Run a quick experiment to get a trained model
    print("🧪 Running quick experiment to get trained model...")
    
    experiment_config = {
        'dataset_path': 'demo_wrapper_test_dataset',
        'architecture_name': 'simple_linear',
        'latent_dim': 8,
        'epochs': 1,  # Just 1 epoch for speed
        'batch_size': 16,
        'learning_rate': 0.001,
        'output_dir': 'test_latent_experiment',
        'save_model': True,  # Need to save model for loading
        'random_seed': 42,
        'verbose': False  # Keep it quiet
    }
    
    try:
        # Run experiment
        result = run_single_experiment(**experiment_config)
        
        if not result.get('success', False):
            print(f"   ❌ Experiment failed: {result.get('error', 'Unknown error')}")
            return False
        
        print(f"   ✅ Experiment completed: {result['experiment_name']}")
        
        # Load the trained model
        model_path = result['model_path']
        if not model_path:
            print("   ❌ No model path in results")
            return False
        
        # Load model
        from autoencoder_lib.models import create_autoencoder
        
        # Get model config from results
        history = result['history']
        latent_dim = result['latent_dim']
        input_shape = result['dataset_info']['input_shape']
        
        print(f"   📐 Input shape: {input_shape}")
        print(f"   🧠 Latent dim: {latent_dim}")
        
        # Create model and load weights
        model = create_autoencoder(
            architecture_name='simple_linear',
            input_shape=tuple(input_shape),
            latent_dim=latent_dim
        )
        
        # Load saved weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✅ Model loaded from: {model_path}")
        
        # Create simple test data from the dataset directory
        from pathlib import Path
        import os
        from PIL import Image
        
        dataset_dir = Path('demo_wrapper_test_dataset')
        if not dataset_dir.exists():
            print(f"   ❌ Dataset directory not found: {dataset_dir}")
            return False
        
        # Load some images from the dataset
        train_data = []
        train_labels = []
        class_names = []
        
        for class_idx, class_dir in enumerate(dataset_dir.iterdir()):
            if class_dir.is_dir():
                class_names.append(class_dir.name)
                image_files = list(class_dir.glob("*.png"))[:10]  # Just 10 images per class
                
                for img_file in image_files:
                    img = Image.open(img_file).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    train_data.append(img_array)
                    train_labels.append(class_idx)
        
        if not train_data:
            print(f"   ❌ No images found in dataset")
            return False
        
        # Convert to tensors
        train_data = torch.tensor(np.array(train_data), dtype=torch.float32).unsqueeze(1)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        
        # Use first half as train, second half as test
        mid_point = len(train_data) // 2
        test_data = train_data[mid_point:]
        test_labels = train_labels[mid_point:]
        train_data = train_data[:mid_point]
        train_labels = train_labels[:mid_point]
        
        print(f"   📊 Data loaded: {len(train_data)} train, {len(test_data)} test")
        print(f"   🏷️ Classes: {class_names}")
        
    except Exception as e:
        print(f"   ❌ Error setting up experiment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Now test the latent analysis
    print("\n🔬 Running latent analysis...")
    try:
        analysis_results = run_complete_latent_analysis(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            class_names=class_names,
            device=torch.device('cpu'),
            output_dir='test_latent_analysis_output',
            include_interpolations=False,  # Skip interpolations for speed
            include_traversals=False  # Skip traversals for speed
        )
        
        print(f"   ✅ Latent analysis completed!")
        
        # Check results structure
        print("\n📋 Analysis Results Summary:")
        print(f"   📊 Top-level keys: {list(analysis_results.keys())}")
        
        summary = analysis_results.get('analysis_summary', {})
        if summary:
            print(f"   🧠 Latent Dimension: {summary.get('latent_dimension', 'N/A')}")
            print(f"   📈 Mean Silhouette Score: {summary.get('mean_silhouette_score', 'N/A'):.4f}")
            print(f"   🎯 Train Optimal Clusters: {summary.get('train_optimal_clusters', 'N/A')}")
            print(f"   🎯 Test Optimal Clusters: {summary.get('test_optimal_clusters', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in latent analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_latent_analysis_with_existing_data()
    if success:
        print("\n🎉 Latent Analysis Integration Test PASSED!")
    else:
        print("\n💥 Latent Analysis Integration Test FAILED!") 
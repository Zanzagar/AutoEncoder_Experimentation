#!/usr/bin/env python3
"""
Test latent analysis tools integration.
"""

import torch
import numpy as np
from autoencoder_lib.experiment.latent_analysis import run_complete_latent_analysis
from autoencoder_lib.models import create_autoencoder

def test_latent_analysis_integration():
    print("=" * 80)
    print("🔬 TESTING LATENT ANALYSIS TOOLS INTEGRATION")
    print("=" * 80)
    
    # Create a simple test dataset
    print("🧪 Creating test dataset...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic 2D data with 2 classes
    n_samples = 100
    latent_dim = 8
    
    # Class 0: circles around origin
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r = np.random.normal(2, 0.5, n_samples//2)
    class0_data = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    
    # Class 1: circles around (4, 4)
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r = np.random.normal(2, 0.5, n_samples//2)
    class1_data = np.stack([4 + r * np.cos(theta), 4 + r * np.sin(theta)], axis=1)
    
    # Convert to image-like tensors (1, 16, 16) - smaller for simple_linear
    def create_image_from_2d(points):
        images = []
        for point in points:
            img = np.zeros((16, 16))  # Smaller image size
            x, y = int(8 + point[0]), int(8 + point[1])
            x, y = np.clip(x, 0, 15), np.clip(y, 0, 15)
            img[y-1:y+2, x-1:x+2] = 1.0  # Small blob at the point
            images.append(img)
        return np.array(images)
    
    class0_images = create_image_from_2d(class0_data)
    class1_images = create_image_from_2d(class1_data)
    
    # Combine and create labels
    train_data = torch.tensor(np.concatenate([class0_images, class1_images]), dtype=torch.float32).unsqueeze(1)
    train_labels = torch.tensor([0] * (n_samples//2) + [1] * (n_samples//2), dtype=torch.long)
    
    # Create smaller test set
    test_data = train_data[:20]  # First 20 samples
    test_labels = train_labels[:20]
    
    class_names = ['circles_center', 'circles_offset']
    
    print(f"   📊 Dataset: {len(train_data)} train, {len(test_data)} test samples")
    print(f"   🏷️ Classes: {class_names}")
    print(f"   📐 Image shape: {train_data.shape}")
    
    # Create and train a simple model
    print("\n🏗️ Creating and training model...")
    try:
        model = create_autoencoder(
            architecture_name='simple_linear',
            input_shape=(1, 16, 16),  # Smaller input shape
            latent_dim=latent_dim
        )
        
        print(f"   ✅ Model created: {model.__class__.__name__}")
        print(f"   🧠 Latent dimension: {latent_dim}")
        
        # Quick training (just a few steps to get a reasonable model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(5):  # Very quick training
            total_loss = 0
            for i in range(0, len(train_data), 16):  # Small batches
                batch = train_data[i:i+16]
                optimizer.zero_grad()
                
                reconstructed, latent = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"   Epoch {epoch}: Loss = {total_loss/len(train_data)*16:.4f}")
        
        print(f"   ✅ Model trained successfully!")
        
    except Exception as e:
        print(f"   ❌ Error creating/training model: {e}")
        return False
    
    # Test latent analysis integration
    print("\n🔬 Running complete latent analysis...")
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
            include_interpolations=True,
            include_traversals=True
        )
        
        print(f"   ✅ Complete latent analysis finished!")
        
        # Check results structure
        print("\n📋 Analysis Results Summary:")
        print(f"   📊 Analysis Summary Keys: {list(analysis_results.get('analysis_summary', {}).keys())}")
        
        summary = analysis_results.get('analysis_summary', {})
        if summary:
            print(f"   🧠 Latent Dimension: {summary.get('latent_dimension', 'N/A')}")
            print(f"   📈 Mean Silhouette Score: {summary.get('mean_silhouette_score', 'N/A'):.4f}")
            print(f"   🎯 Train Optimal Clusters: {summary.get('train_optimal_clusters', 'N/A')}")
            print(f"   🎯 Test Optimal Clusters: {summary.get('test_optimal_clusters', 'N/A')}")
        
        if 'latent_tsne_analysis' in analysis_results:
            tsne_results = analysis_results['latent_tsne_analysis']
            print(f"   🎨 t-SNE Train Silhouette: {tsne_results.get('train_silhouette', 'N/A'):.4f}")
            print(f"   🎨 t-SNE Test Silhouette: {tsne_results.get('test_silhouette', 'N/A'):.4f}")
        
        if 'interpolation_analysis' in analysis_results:
            interp_results = analysis_results['interpolation_analysis']
            print(f"   🔀 Interpolations: {interp_results.get('num_interpolations', 'N/A')}")
        
        if 'traversal_analysis' in analysis_results:
            trav_results = analysis_results['traversal_analysis']
            print(f"   🚶 Traversals: {trav_results.get('num_traversals', 'N/A')}")
        
        print(f"\n   📁 Output saved to: test_latent_analysis_output/")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in latent analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_latent_analysis_integration()
    if success:
        print("\n🎉 Latent Analysis Integration Test PASSED!")
    else:
        print("\n💥 Latent Analysis Integration Test FAILED!") 
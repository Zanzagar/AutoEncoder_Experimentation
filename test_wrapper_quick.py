#!/usr/bin/env python3
"""
Quick test script for ExperimentRunnerWrapper
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os

# Test basic imports
print("Testing imports...")
from autoencoder_lib.experiment import ExperimentRunner, run_single_experiment
from autoencoder_lib.models import create_autoencoder, MODEL_ARCHITECTURES
from autoencoder_lib.data import generate_dataset
print("✅ All imports successful!")

# Test data generation
print("\nTesting data generation...")
dataset_info = generate_dataset(
    dataset_type='layered_geological',
    output_dir='quick_test_dataset',
    num_samples_per_class=5,
    image_size=32,
    num_classes=2
)
print("✅ Data generation successful!")

# Test model creation
print("\nTesting model creation...")
model = create_autoencoder(
    architecture_name='simple_linear',
    input_size=32*32,
    latent_dim=8
)
print(f"✅ Model created: {type(model).__name__}")

# Test experiment runner
print("\nTesting ExperimentRunner...")
runner = ExperimentRunner(
    output_dir="quick_test_results",
    random_seed=42
)
print("✅ ExperimentRunner initialized!")

print("\n🎉 All basic tests passed! The wrapper components are working correctly.")
print(f"Available architectures: {list(MODEL_ARCHITECTURES.keys())}")

# Quick test for the fixed single experiment wrapper
def test_quick_experiment():
    """Test a very quick single experiment to verify all fixes work"""
    print("=== Quick Single Experiment Test ===")
    
    try:
        from autoencoder_lib.experiment import run_single_experiment
        
        # Configuration using existing dataset
        config = {
            'dataset_path': 'demo_wrapper_test_dataset',  # Use existing dataset
            'architecture_name': 'simple_linear',
            'latent_dim': 8,  # Small latent dim for quick test
            'epochs': 1,      # Just 1 epoch for speed
            'batch_size': 16, # Small batch for speed
            'learning_rate': 0.001,
            'output_dir': 'quick_test_results',
            'save_model': True,  # Save model to test .pth functionality
            'random_seed': 42,
            'verbose': True
        }
        
        print("Running quick experiment...")
        print(f"Config: {config}")
        
        # Run the experiment
        result = run_single_experiment(**config)
        
        # Check results
        if result.get('success', False):
            print("✅ Quick experiment completed successfully!")
            print(f"  Architecture: {result.get('architecture')}")
            print(f"  Latent dim: {result.get('latent_dim')}")
            print(f"  Final test loss: {result.get('metrics', {}).get('final_test_loss', 'N/A')}")
            return True
        else:
            print(f"❌ Experiment failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Quick Test for Fixed Wrapper Functions")
    print("=" * 50)
    
    success = test_quick_experiment()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Quick test passed! The fixes are working.")
        print("You can now run your comprehensive demo notebook.")
    else:
        print("⚠️  Quick test failed. Check the errors above.")
    
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Test script to verify the fix for plot_training_curves import error
in systematic experiments.
"""

import sys
import os
sys.path.append('autoencoder_lib')

from autoencoder_lib.experiment.wrappers import run_systematic_experiments

def test_systematic_experiments_fix():
    """Test the systematic experiments fix."""
    print("🧪 Testing systematic experiments fix...")
    
    # Quick test configuration
    test_config = {
        'dataset_path': 'demo_wrapper_test_dataset',
        'architectures': ['simple_linear'],
        'latent_dims': [4, 8],
        'learning_rates': [0.001],  # Fixed: plural and list
        'epochs': 1,  # Just 1 epoch for speed
        'batch_size': 16,
        'output_dir': 'test_systematic_fix',
        'random_seed': 42,
        'verbose': False  # Keep it quiet
    }
    
    try:
        results = run_systematic_experiments(**test_config)
        print("✅ SUCCESS: Systematic experiments completed without error!")
        print(f"📊 Generated results for {len(results)} architectures")
        
        # Check if all expected parts completed
        for arch, arch_results in results.items():
            print(f"   {arch}: {len(arch_results)} latent dimension experiments")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_systematic_experiments_fix()
    print("\n" + "="*50)
    if success:
        print("🎉 Fix verified successfully!")
    else:
        print("💥 Fix needs more work!") 
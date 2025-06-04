#!/usr/bin/env python3
"""
Simple test for systematic experiments to reproduce and fix the plotting issue.
"""

from autoencoder_lib.experiment.wrappers import run_systematic_experiments

def test_systematic_experiments():
    print("=" * 60)
    print("🧪 TESTING SYSTEMATIC EXPERIMENTS (Plotting Fix)")
    print("=" * 60)
    
    # Configuration for a very quick systematic test 
    systematic_config = {
        'dataset_path': 'demo_wrapper_test_dataset',  
        'architectures': ['simple_linear', 'deeper_linear'],  # Only 2 architectures
        'latent_dims': [4, 8],  # Only 2 latent dims 
        'learning_rates': [0.001],  # Single learning rate
        'epochs': 2,  # Very few epochs
        'batch_size': 16,  # Small batch for speed
        'output_dir': 'test_systematic_results',
        'random_seed': 42,
        'generate_visualizations': True,  # Test visualization generation
        'verbose': True
    }
    
    print("Configuration:")
    for key, value in systematic_config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        print("🚀 Starting systematic experiments...")
        systematic_results = run_systematic_experiments(**systematic_config)
        
        print(f"✅ Systematic experiments completed successfully!")
        print(f"Results summary:")
        for arch, results in systematic_results.items():
            print(f"  {arch}: {len(results)} experiments")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in systematic experiments: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_systematic_experiments()
    if success:
        print("🎉 Systematic test passed!")
    else:
        print("💥 Systematic test failed!") 
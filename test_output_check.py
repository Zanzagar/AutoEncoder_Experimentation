#!/usr/bin/env python3
"""
Simple test script to show the improved final output organization.
This will run a very quick single experiment to demonstrate the
consolidated final statistics and clear epoch labeling.
"""

from autoencoder_lib.experiment.wrappers import run_single_experiment

def test_final_output():
    print("=" * 60)
    print("🧪 TESTING IMPROVED FINAL OUTPUT ORGANIZATION")
    print("=" * 60)
    
    # Configuration for a very quick test
    config = {
        'dataset_path': 'demo_wrapper_test_dataset',
        'architecture_name': 'simple_linear',
        'latent_dim': 4,  # Very small for speed
        'epochs': 1,      # Single epoch 
        'batch_size': 32, # Larger batch for speed
        'learning_rate': 0.001,
        'output_dir': 'test_output_demo',
        'save_model': False,  # Skip saving for speed
        'random_seed': 42,
        'verbose': False,  # Reduce verbosity to focus on final output
        'visualization_interval': 10000,  # No mid-training visualization
        'num_visualizations': 0  # Only final visualization
    }
    
    print("Running experiment with improved output formatting...")
    print("Watch for:")
    print("  ✅ Clear epoch numbering (1/1 instead of 0/1)")
    print("  ✅ Consolidated final statistics section")
    print("  ✅ Clear distinction between training progress and final results")
    print("=" * 60)
    
    # Run the experiment
    try:
        results = run_single_experiment(**config)
        print("\n🎉 SUCCESS! Check the output above to see:")
        print("  📊 Organized final statistics")
        print("  🎯 Clear final results labeling")
        print("  📈 Proper epoch numbering")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_final_output()
    print(f"\nTest {'PASSED' if success else 'FAILED'}!") 
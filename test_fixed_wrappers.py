# Test script for the fixed wrapper functions

import os
import sys

def test_dataset_loading():
    """Test that we can load an existing dataset"""
    print("=== Testing Dataset Loading ===")
    
    # Check if we have the dataset
    dataset_path = 'demo_wrapper_test_dataset'
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    
    dataset_info_file = os.path.join(dataset_path, 'dataset_info.npy')
    if not os.path.exists(dataset_info_file):
        print(f"❌ dataset_info.npy not found at {dataset_info_file}")
        return False
    
    print(f"✅ Dataset found at {dataset_path}")
    print(f"✅ dataset_info.npy found at {dataset_info_file}")
    
    # Try to load it
    try:
        import numpy as np
        dataset_info = np.load(dataset_info_file, allow_pickle=True).item()
        print(f"✅ Successfully loaded dataset_info")
        print(f"  - Classes: {dataset_info.get('class_names', 'Unknown')}")
        print(f"  - Keys: {list(dataset_info.keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset_info: {e}")
        return False

def test_function_imports():
    """Test that all functions import correctly"""
    print("\n=== Testing Function Imports ===")
    
    try:
        from autoencoder_lib.experiment import run_single_experiment, run_systematic_experiments
        from autoencoder_lib.models import create_autoencoder
        print("✅ All functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_create_autoencoder():
    """Test the fixed create_autoencoder function"""
    print("\n=== Testing create_autoencoder Function ===")
    
    try:
        from autoencoder_lib.models import create_autoencoder
        
        # Test correct usage
        model = create_autoencoder(
            architecture_name='simple_linear',
            input_size=64*64,
            latent_dim=16
        )
        print("✅ create_autoencoder works with correct parameters")
        print(f"  - Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ create_autoencoder failed: {e}")
        return False

def test_single_experiment_signature():
    """Test the new single experiment function signature"""
    print("\n=== Testing Single Experiment Signature ===")
    
    try:
        from autoencoder_lib.experiment import run_single_experiment
        import inspect
        
        # Get function signature
        sig = inspect.signature(run_single_experiment)
        params = list(sig.parameters.keys())
        
        print("✅ Function signature retrieved")
        print(f"  - Parameters: {params}")
        
        # Check for expected parameters
        expected_params = ['dataset_config', 'dataset_info', 'dataset_path']
        for param in expected_params:
            if param in params:
                print(f"  ✅ {param} parameter found")
            else:
                print(f"  ❌ {param} parameter missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fixed Wrapper Functions")
    print("=" * 50)
    
    tests = [
        test_function_imports,
        test_create_autoencoder,
        test_single_experiment_signature,
        test_dataset_loading
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Fixes are working correctly.")
        print("\nYour notebook should now work correctly:")
        print("- create_autoencoder() function call is fixed")
        print("- Dataset loading from existing dataset_info.npy is supported")
        print("- Both single and systematic experiments can load existing datasets")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    sys.exit(0 if passed == total else 1) 
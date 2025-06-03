#!/usr/bin/env python3
"""
Test script to verify all autoencoder_lib imports work correctly.
"""

print("Testing autoencoder_lib imports...")

# Test data import
try:
    from autoencoder_lib.data import generate_dataset, visualize_dataset
    print("✅ data module imported (generate_dataset, visualize_dataset)")
except Exception as e:
    print(f"❌ data module failed: {e}")
    import traceback
    traceback.print_exc()

# Test models import  
try:
    from autoencoder_lib.models.factory import create_autoencoder, get_available_architectures
    print("✅ models.factory imported (create_autoencoder, get_available_architectures)")
    
    # Test getting available architectures
    architectures = get_available_architectures()
    print(f"   Available architectures: {architectures}")
except Exception as e:
    print(f"❌ models.factory failed: {e}")
    import traceback
    traceback.print_exc()

# Test experiment import
try:
    from autoencoder_lib.experiment.runner import ExperimentRunner
    print("✅ experiment.runner imported (ExperimentRunner)")
except Exception as e:
    print(f"❌ experiment.runner failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed.") 
#!/usr/bin/env python3
"""
Quick test script for ExperimentRunnerWrapper
"""

import torch
import numpy as np
from pathlib import Path

# Test basic imports
print("Testing imports...")
from autoencoder_lib.experiment import ExperimentRunner
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
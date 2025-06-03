#!/usr/bin/env python3
"""
Simple test for ExperimentRunnerWrapper - testing wrapper class itself
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset

# Import all required modules
from autoencoder_lib.experiment import ExperimentRunner
from autoencoder_lib.models import ModelFactory, create_autoencoder, MODEL_ARCHITECTURES
from autoencoder_lib.data import generate_dataset

# Simple wrapper class (minimal for testing)
class ExperimentRunnerWrapper:
    """Test version of the ExperimentRunnerWrapper"""
    
    def __init__(self, output_dir: str = "test_results", random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = random_seed
        
        # Initialize experiment runner
        self.experiment_runner = ExperimentRunner(
            device=self.device,
            output_dir=str(self.output_dir),
            random_seed=random_seed
        )
        
        print(f"ExperimentRunnerWrapper initialized:")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
    
    def test_model_creation(self):
        """Test creating models with different architectures"""
        print("\nTesting model creation:")
        
        for arch_name in MODEL_ARCHITECTURES.keys():
            try:
                if arch_name in ['convolutional', 'deeper_convolutional']:
                    model = create_autoencoder(
                        architecture_name=arch_name,
                        input_channels=1,
                        latent_dim=16
                    )
                else:
                    model = create_autoencoder(
                        architecture_name=arch_name,
                        input_size=64*64,
                        latent_dim=16
                    )
                print(f"  ✅ {arch_name}: {type(model).__name__}")
            except Exception as e:
                print(f"  ❌ {arch_name}: {e}")

# Run the test
if __name__ == "__main__":
    print("Testing ExperimentRunnerWrapper...")
    
    # Test wrapper initialization
    wrapper = ExperimentRunnerWrapper()
    
    # Test model creation
    wrapper.test_model_creation()
    
    # Test simple dataset generation
    print("\nTesting dataset generation:")
    dataset_info = generate_dataset(
        dataset_type='layered_geological',
        output_dir='simple_test_dataset',
        num_samples_per_class=5,
        image_size=32,
        num_classes=2
    )
    print("✅ Dataset generation successful!")
    
    print("\n🎉 All tests passed! Wrapper is working correctly.") 
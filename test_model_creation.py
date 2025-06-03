#!/usr/bin/env python3
"""
Test script to verify model creation works correctly with different image sizes.
"""

import torch
print("Torch imported successfully")

try:
    from autoencoder_lib.models.factory import create_autoencoder, get_available_architectures
    print("Factory imports successful")
except Exception as e:
    print(f"Factory import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("Testing model creation with different image sizes...")

# Test parameters
test_cases = [
    {"image_size": 16, "input_size": 16*16, "latent_dim": 5},
    {"image_size": 32, "input_size": 32*32, "latent_dim": 10}, 
    {"image_size": 64, "input_size": 64*64, "latent_dim": 15}
]

try:
    architectures = get_available_architectures()
    print(f"Available architectures: {architectures}")
except Exception as e:
    print(f"Failed to get architectures: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

for arch in architectures:
    print(f"\nTesting {arch} architecture:")
    
    for test_case in test_cases:
        img_size = test_case["image_size"]
        input_size = test_case["input_size"]
        latent_dim = test_case["latent_dim"]
        
        print(f"  Testing {img_size}x{img_size} image...")
        
        try:
            # Create model with appropriate parameters
            if 'convolutional' in arch:
                print(f"    Creating convolutional model...")
                model = create_autoencoder(
                    architecture_name=arch,
                    input_channels=1,
                    latent_dim=latent_dim,
                    input_size=(img_size, img_size)
                )
                # Test with sample data
                test_input = torch.randn(1, 1, img_size, img_size)
            else:
                print(f"    Creating linear model...")
                model = create_autoencoder(
                    architecture_name=arch,
                    input_size=input_size,
                    latent_dim=latent_dim
                )
                # Test with sample data
                test_input = torch.randn(1, 1, img_size, img_size)
            
            print(f"    Model created, testing forward pass...")
            # Test forward pass
            encoded, decoded = model(test_input)
            
            print(f"  ✅ {img_size}x{img_size}: input {test_input.shape} -> encoded {encoded.shape} -> decoded {decoded.shape}")
            
        except Exception as e:
            print(f"  ❌ {img_size}x{img_size}: {e}")
            import traceback
            traceback.print_exc()

print("\nModel creation test completed.") 
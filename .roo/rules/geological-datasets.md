---
description: 
globs: 
alwaysApply: false
---
# Dataset Guidelines

## Dataset Progression Strategy

- Start with datasets containing classes with extremely small differences
- Progressively add more complex features and classes
- Build datasets that incrementally challenge the latent space representation
- Compare performance across datasets of increasing complexity

## Geological Dataset Structure

- `layered_geologic_patterns_dataset/consistent_layers/`: Contains geological patterns with uniform layer thickness and properties
- `layered_geologic_patterns_dataset/variable_layers/`: Contains geological patterns with varying layer thickness, angles, and properties

## Additional Dataset Types

- Simple geometric shapes with subtle variations
- Texture patterns with progressive complexity
- Domain-specific image datasets (e.g., medical, astronomical)
- Synthetic datasets with controllable feature complexity
- Consider using generative models to create datasets with precise feature control

## Image Format and Preprocessing

- Images should be processed as numpy arrays for input to the autoencoder
- Normalize pixel values to the range [0,1] before feeding to neural networks
- Apply consistent preprocessing across all datasets for fair comparison
- Consider applying data augmentation techniques to expand the training dataset

## Feature Extraction and Analysis

- Focus on extracting features related to:
  - Shape boundaries and contours
  - Texture patterns and gradients
  - Spatial relationships between elements
  - Color distributions (if applicable)
- Document which features are expected to be discriminated in each dataset
- Track which features are successfully encoded in the latent space

## Dataset Split Guidelines

- Use consistent train/validation/test splits for reproducible results
- Document the random seed used for splitting
- Ensure balanced representation of different pattern types in each split
- Consider cross-validation for robust evaluation across dataset variations




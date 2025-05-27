---
description: 
globs: 
alwaysApply: false
---
# Autoencoder Architecture Guidelines

## Architecture Exploration

- Experiment with multiple autoencoder architectures to compare latent space representations
- Consider these architecture variants:
  - Standard symmetric autoencoders
  - Variational autoencoders (VAEs) for probabilistic latent spaces
  - Convolutional autoencoders for spatial feature preservation
  - Adversarial autoencoders for improved reconstruction quality
  - Sparse autoencoders for more efficient representations

## Core Architecture

- Use a symmetric architecture where encoder and decoder mirror each other
- Consider the following layer types:
  - Convolutional layers for spatial feature extraction
  - Pooling layers for dimensionality reduction
  - Dense layers for the bottleneck (latent space)
  - Transposed convolutional layers for upsampling
- Document tradeoffs between architecture complexity and discrimination capabilities

## Latent Space Investigation

- Experiment with different latent space dimensions to find the optimal bottleneck size
- Systematically reduce latent dimensions to find the minimum required for discrimination
- Test the boundaries where the latent space fails to discriminate between similar classes
- Visualize latent spaces using:
  - t-SNE or UMAP for high-dimensional visualization
  - PCA for understanding principal components
  - Latent space traversal to interpret encoded features
- Store exploration results in appropriate directories for each dataset type

## Discrimination Analysis

- Implement classification methods on the latent representations
- Analyze which features are preserved and which are lost in the compression
- Compare latent space clusters across different model architectures
- Study how incremental changes in the input affect the latent representation
- Document when the latent space fails to separate different classes

## Loss Functions

- Use Mean Squared Error (MSE) for basic reconstruction
- Consider structural similarity index (SSIM) for preserving key structural features
- Experiment with perceptual losses for better feature preservation
- For specialized applications, design custom loss functions that penalize errors in discriminative features
- Consider adding classification-based loss terms to enhance discrimination

## Training Protocol

- Use appropriate batch sizes based on memory constraints and dataset size
- Document the learning rate and optimizer configuration
- Implement early stopping to prevent overfitting
- Save model checkpoints at regular intervals
- Use consistent training protocols across different architecture experiments for fair comparison

## Evaluation Metrics

- Reconstruction error (MSE, MAE)
- Structural similarity metrics
- Latent space discrimination metrics (e.g., silhouette score, classification accuracy)
- Visual comparison of original vs. reconstructed patterns
- Feature preservation assessment for different levels of feature complexity
- Quantitative comparison of different architectures on the same datasets




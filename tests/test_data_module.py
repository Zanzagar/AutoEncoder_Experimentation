#!/usr/bin/env python3
"""
Unit Tests for Data Module

This module contains comprehensive tests for all data module components:
- BaseDataset and BaseDataLoader abstract classes
- LayeredGeologicalDataset implementation
- StandardDataLoader functionality
- Data preprocessing utilities
- Visualization functions
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autoencoder_lib.data import (
    BaseDataset, BaseDataLoader,
    LayeredGeologicalDataset, StandardDataLoader, ShapeDataset,
    MinMaxNormalizer, StandardNormalizer, RobustNormalizer,
    DataAugmenter, PreprocessingPipeline,
    create_standard_pipeline, preprocess_for_pytorch,
    calculate_data_statistics
)
from autoencoder_lib.visualization import (
    visualize_dataset_samples, plot_class_distribution,
    show_sample_grid, visualize_raw_data_tsne
)


class TestBaseClasses(unittest.TestCase):
    """Test abstract base classes."""
    
    def test_base_dataset_is_abstract(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseDataset()
    
    def test_base_dataloader_is_abstract(self):
        """Test that BaseDataLoader cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseDataLoader()


class TestLayeredGeologicalDataset(unittest.TestCase):
    """Test LayeredGeologicalDataset implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = LayeredGeologicalDataset(name="test_dataset")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        self.assertEqual(self.dataset.name, "test_dataset")
        self.assertIsNone(self.dataset.dataset_info)
    
    def test_dataset_generation(self):
        """Test dataset generation with all pattern types."""
        dataset_info = self.dataset.generate(
            output_dir=None,  # Generate in memory
            image_size=32,
            num_samples_per_class=5,
            random_seed=42
        )
        
        # Check basic structure
        self.assertIn('images', dataset_info)
        self.assertIn('labels', dataset_info)
        self.assertIn('class_names', dataset_info)
        self.assertIn('metadata', dataset_info)
        
        # Check data shapes and types
        images = dataset_info['images']
        labels = dataset_info['labels']
        
        self.assertEqual(len(images), 25)  # 5 classes * 5 samples
        self.assertEqual(len(labels), 25)
        self.assertEqual(images.shape, (25, 32, 32))
        self.assertEqual(len(dataset_info['class_names']), 5)
        
        # Check data ranges
        self.assertTrue(np.all(images >= 0))
        self.assertTrue(np.all(images <= 1))
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.assertEqual(len(unique_labels), 5)
        self.assertTrue(np.all(counts == 5))
    
    def test_dataset_reproducibility(self):
        """Test that dataset generation is reproducible with same seed."""
        dataset_info1 = self.dataset.generate(
            output_dir=None,
            image_size=16,
            num_samples_per_class=3,
            random_seed=123
        )
        
        dataset_info2 = self.dataset.generate(
            output_dir=None,
            image_size=16,
            num_samples_per_class=3,
            random_seed=123
        )
        
        # Should be identical
        np.testing.assert_array_equal(dataset_info1['images'], dataset_info2['images'])
        np.testing.assert_array_equal(dataset_info1['labels'], dataset_info2['labels'])
    
    def test_dataset_save_load(self):
        """Test dataset saving and loading."""
        # Generate dataset
        dataset_info = self.dataset.generate(
            output_dir=self.temp_dir,
            image_size=16,
            num_samples_per_class=3,
            random_seed=42
        )
        
        # Check files were created
        dataset_file = os.path.join(self.temp_dir, "dataset_info.npy")
        images_file = os.path.join(self.temp_dir, "images.npy")
        labels_file = os.path.join(self.temp_dir, "labels.npy")
        metadata_file = os.path.join(self.temp_dir, "metadata.json")
        
        self.assertTrue(os.path.exists(dataset_file))
        self.assertTrue(os.path.exists(images_file))
        self.assertTrue(os.path.exists(labels_file))
        self.assertTrue(os.path.exists(metadata_file))
        
        # Load dataset from directory
        loaded_info = self.dataset.load(self.temp_dir)
        
        # Compare original and loaded
        np.testing.assert_array_equal(dataset_info['images'], loaded_info['images'])
        np.testing.assert_array_equal(dataset_info['labels'], loaded_info['labels'])
        self.assertEqual(dataset_info['class_names'], loaded_info['class_names'])
    
    def test_pattern_types(self):
        """Test that all geological pattern types are generated correctly."""
        dataset_info = self.dataset.generate(
            output_dir=None,
            image_size=24,
            num_samples_per_class=2,
            random_seed=42
        )
        
        expected_patterns = [
            'horizontal_layers',
            'folded_layers', 
            'faulted_layers',
            'intrusion_patterns',
            'unconformity_patterns'
        ]
        
        self.assertEqual(dataset_info['class_names'], expected_patterns)
        
        # Check that different patterns produce different images
        images = dataset_info['images']
        labels = dataset_info['labels']
        
        # Get one sample from each class
        class_samples = []
        for class_id in range(5):
            class_indices = np.where(labels == class_id)[0]
            class_samples.append(images[class_indices[0]])
        
        # Verify they're different (not identical)
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertFalse(np.array_equal(class_samples[i], class_samples[j]))


class TestStandardDataLoader(unittest.TestCase):
    """Test StandardDataLoader implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.dataset = LayeredGeologicalDataset(name="test_loader")
        self.dataset_info = self.dataset.generate(
            output_dir=None,
            image_size=16,
            num_samples_per_class=10,
            random_seed=42
        )
        
        self.loader = StandardDataLoader()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_splitting(self):
        """Test train/test data splitting."""
        split_info = self.loader.create_split(
            self.dataset_info,
            train_ratio=0.7,
            test_ratio=0.3,
            random_seed=42
        )
        
        # Check split structure
        self.assertIn('train_indices', split_info)
        self.assertIn('test_indices', split_info)
        self.assertIn('metadata', split_info)
        
        # Check split sizes
        total_samples = len(self.dataset_info['images'])
        expected_train = int(0.7 * total_samples)
        expected_test = total_samples - expected_train
        
        self.assertEqual(len(split_info['train_indices']), expected_train)
        self.assertEqual(len(split_info['test_indices']), expected_test)
        
        # Check no overlap
        train_set = set(split_info['train_indices'])
        test_set = set(split_info['test_indices'])
        self.assertEqual(len(train_set.intersection(test_set)), 0)
    
    def test_split_reproducibility(self):
        """Test that splits are reproducible with same seed."""
        split1 = self.loader.create_split(
            self.dataset_info,
            train_ratio=0.8,
            test_ratio=0.2,
            random_seed=123
        )
        
        split2 = self.loader.create_split(
            self.dataset_info,
            train_ratio=0.8,
            test_ratio=0.2,
            random_seed=123
        )
        
        np.testing.assert_array_equal(split1['train_indices'], split2['train_indices'])
        np.testing.assert_array_equal(split1['test_indices'], split2['test_indices'])
    
    def test_get_train_test_data(self):
        """Test getting train and test data."""
        split_info = self.loader.create_split(
            self.dataset_info,
            train_ratio=0.6,
            test_ratio=0.4,
            random_seed=42
        )
        
        train_data = self.loader.get_train_data(self.dataset_info, split_info)
        test_data = self.loader.get_test_data(self.dataset_info, split_info)
        
        # Check data structure
        self.assertIn('images', train_data)
        self.assertIn('labels', train_data)
        self.assertIn('images', test_data)
        self.assertIn('labels', test_data)
        
        # Check data consistency
        total_samples = len(self.dataset_info['images'])
        self.assertEqual(
            len(train_data['images']) + len(test_data['images']),
            total_samples
        )
    
    def test_pytorch_dataset(self):
        """Test PyTorch Dataset wrapper."""
        images = self.dataset_info['images']
        labels = self.dataset_info['labels']
        
        pytorch_dataset = ShapeDataset(images, labels)
        
        # Check dataset length
        self.assertEqual(len(pytorch_dataset), len(images))
        
        # Check data access
        sample_image, sample_target, sample_label = pytorch_dataset[0]
        self.assertIsInstance(sample_image, torch.Tensor)
        self.assertIsInstance(sample_target, torch.Tensor)
        self.assertIsInstance(sample_label, torch.Tensor)
        
        # Check data shapes
        expected_shape = (1, images.shape[1], images.shape[2])  # (C, H, W)
        self.assertEqual(sample_image.shape, expected_shape)
        self.assertEqual(sample_target.shape, expected_shape)  # Target should be same as input for autoencoders


class TestPreprocessing(unittest.TestCase):
    """Test data preprocessing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        self.test_data = np.random.rand(20, 16, 16) * 255  # Random data [0, 255]
        self.test_labels = np.random.randint(0, 3, 20)
    
    def test_minmax_normalizer(self):
        """Test MinMax normalization."""
        normalizer = MinMaxNormalizer(feature_range=(0, 1))
        
        # Test fit and transform
        normalized = normalizer.fit_transform(self.test_data)
        
        # Check range
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)
        
        # Test inverse transform
        recovered = normalizer.inverse_transform(normalized)
        np.testing.assert_allclose(self.test_data, recovered, rtol=1e-10)
    
    def test_standard_normalizer(self):
        """Test Standard (z-score) normalization."""
        normalizer = StandardNormalizer()
        
        normalized = normalizer.fit_transform(self.test_data)
        
        # Check mean and std
        self.assertAlmostEqual(normalized.mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized.std(), 1.0, places=5)
        
        # Test inverse transform
        recovered = normalizer.inverse_transform(normalized)
        np.testing.assert_allclose(self.test_data, recovered, rtol=1e-10)
    
    def test_robust_normalizer(self):
        """Test Robust normalization."""
        normalizer = RobustNormalizer()
        
        normalized = normalizer.fit_transform(self.test_data)
        
        # Test inverse transform
        recovered = normalizer.inverse_transform(normalized)
        np.testing.assert_allclose(self.test_data, recovered, rtol=1e-10)
    
    def test_data_augmenter(self):
        """Test data augmentation."""
        augmenter = DataAugmenter(random_seed=42)
        
        # Test individual augmentation methods
        test_image = self.test_data[0] / 255.0  # Normalize to [0, 1]
        
        # Test noise addition
        noisy = augmenter.add_noise(test_image, 'gaussian', 0.1)
        self.assertEqual(noisy.shape, test_image.shape)
        self.assertTrue(np.all(noisy >= 0))
        self.assertTrue(np.all(noisy <= 1))
        
        # Test rotation
        rotated = augmenter.rotate_image(test_image, (-10, 10))
        self.assertEqual(rotated.shape, test_image.shape)
        
        # Test flipping
        flipped = augmenter.flip_image(test_image)
        self.assertEqual(flipped.shape, test_image.shape)
        
        # Test batch augmentation
        test_images = self.test_data[:5] / 255.0
        test_labels = self.test_labels[:5]
        
        aug_images, aug_labels = augmenter.augment_batch(
            test_images, test_labels, augmentation_factor=2
        )
        
        self.assertEqual(len(aug_images), len(test_images) * 2)
        self.assertEqual(len(aug_labels), len(test_labels) * 2)
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        pipeline = create_standard_pipeline()
        
        # Test pipeline operations
        processed = pipeline.fit_transform(self.test_data)
        recovered = pipeline.inverse_transform(processed)
        
        # Check that pipeline preserves data through inverse transform
        np.testing.assert_allclose(self.test_data, recovered, rtol=1e-10)
    
    def test_pytorch_integration(self):
        """Test PyTorch integration functions."""
        normalizer = MinMaxNormalizer()
        
        # Test preprocessing for PyTorch
        data_tensor, labels_tensor = preprocess_for_pytorch(
            self.test_data / 255.0,  # Normalize input
            self.test_labels,
            normalizer=normalizer,
            augment=False
        )
        
        # Check tensor types and shapes
        self.assertIsInstance(data_tensor, torch.Tensor)
        self.assertIsInstance(labels_tensor, torch.Tensor)
        
        expected_shape = (len(self.test_data), 1, 16, 16)  # (N, C, H, W)
        self.assertEqual(data_tensor.shape, expected_shape)
        self.assertEqual(labels_tensor.shape, (len(self.test_labels),))
    
    def test_data_statistics(self):
        """Test data statistics calculation."""
        stats = calculate_data_statistics(self.test_data)
        
        # Check that all expected statistics are present
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'shape', 'dtype']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that statistics make sense
        self.assertLessEqual(stats['min'], stats['q25'])
        self.assertLessEqual(stats['q25'], stats['median'])
        self.assertLessEqual(stats['median'], stats['q75'])
        self.assertLessEqual(stats['q75'], stats['max'])


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test dataset
        dataset = LayeredGeologicalDataset(name="test_viz")
        self.dataset_info = dataset.generate(
            output_dir=None,
            image_size=16,
            num_samples_per_class=5,
            random_seed=42
        )
    
    def test_visualize_dataset_samples(self):
        """Test dataset sample visualization."""
        # This should not raise an exception
        try:
            visualize_dataset_samples(
                self.dataset_info,
                samples_per_class=2,
                figure_size=(8, 6),
                random_seed=42
            )
        except Exception as e:
            self.fail(f"visualize_dataset_samples raised an exception: {e}")
    
    def test_plot_class_distribution(self):
        """Test class distribution plotting."""
        try:
            plot_class_distribution(
                self.dataset_info,
                figure_size=(8, 6)
            )
        except Exception as e:
            self.fail(f"plot_class_distribution raised an exception: {e}")
    
    def test_show_sample_grid(self):
        """Test sample grid visualization."""
        images = self.dataset_info['images'][:9]  # First 9 images
        labels = self.dataset_info['labels'][:9]
        
        try:
            show_sample_grid(
                images,
                labels=labels,
                label_names=self.dataset_info['class_names'],
                grid_size=(3, 3),
                figure_size=(8, 8)
            )
        except Exception as e:
            self.fail(f"show_sample_grid raised an exception: {e}")
    
    def test_tsne_visualization(self):
        """Test t-SNE visualization."""
        try:
            # Test with small dataset to make it fast
            embedding, labels, score = visualize_raw_data_tsne(
                dataset_info=self.dataset_info,
                random_state=42,
                max_samples=20,  # Limit samples for speed
                figure_size=(8, 6)
            )
            
            # Check return values
            self.assertEqual(len(embedding), len(labels))
            self.assertIsInstance(score, float)
            
        except Exception as e:
            self.fail(f"visualize_raw_data_tsne raised an exception: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete data pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete data processing pipeline."""
        # 1. Generate dataset
        dataset = LayeredGeologicalDataset(name="integration_test")
        dataset_info = dataset.generate(
            output_dir=None,
            image_size=16,
            num_samples_per_class=8,
            random_seed=42
        )
        
        # 2. Create data loader and split data
        loader = StandardDataLoader()
        split_info = loader.create_split(
            dataset_info,
            train_ratio=0.7,
            test_ratio=0.3,
            random_seed=42
        )
        
        train_data = loader.get_train_data(dataset_info, split_info)
        test_data = loader.get_test_data(dataset_info, split_info)
        
        # 3. Preprocess data
        normalizer = MinMaxNormalizer()
        train_tensor, train_labels = preprocess_for_pytorch(
            train_data['images'],
            train_data['labels'],
            normalizer=normalizer,
            augment=True,
            augmentation_factor=2
        )
        
        test_tensor, test_labels = preprocess_for_pytorch(
            test_data['images'],
            test_data['labels'],
            normalizer=normalizer,
            augment=False
        )
        
        # 4. Verify pipeline results
        self.assertIsInstance(train_tensor, torch.Tensor)
        self.assertIsInstance(test_tensor, torch.Tensor)
        
        # Check that augmentation worked
        self.assertEqual(len(train_tensor), len(train_data['images']) * 2)
        self.assertEqual(len(test_tensor), len(test_data['images']))
        
        # Check data ranges
        self.assertTrue(torch.all(train_tensor >= 0))
        self.assertTrue(torch.all(train_tensor <= 1))
        self.assertTrue(torch.all(test_tensor >= 0))
        self.assertTrue(torch.all(test_tensor <= 1))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestBaseClasses,
        TestLayeredGeologicalDataset,
        TestStandardDataLoader,
        TestPreprocessing,
        TestVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code) 
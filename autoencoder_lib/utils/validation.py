"""
Parameter Validation Utilities

Functions for validating experiment configurations, model parameters, and data inputs.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging


def validate_positive_int(value: Any, name: str, min_value: int = 1) -> int:
    """
    Validate that a value is a positive integer.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        
    Returns:
        Validated integer value
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value)}")
    
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    return int(value)


def validate_positive_float(value: Any, name: str, min_value: float = 0.0,
                           max_value: Optional[float] = None) -> float:
    """
    Validate that a value is a positive float.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
        
    Returns:
        Validated float value
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a number, got {type(value)}")
    
    value = float(value)
    
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_choice(value: Any, name: str, choices: List[Any]) -> Any:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        choices: List of allowed values
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If validation fails
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")
    
    return value


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...],
                         name: str, allow_batch: bool = True) -> torch.Tensor:
    """
    Validate tensor shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (can use -1 for any dimension)
        name: Parameter name for error messages
        allow_batch: If True, allow additional batch dimension
        
    Returns:
        Validated tensor
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    actual_shape = tensor.shape
    
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        # Skip batch dimension
        actual_shape = actual_shape[1:]
    
    if len(actual_shape) != len(expected_shape):
        raise ValueError(f"{name} expected {len(expected_shape)} dimensions, "
                        f"got {len(actual_shape)}")
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(f"{name} dimension {i} expected {expected}, got {actual}")
    
    return tensor


def validate_data_consistency(train_data: torch.Tensor, test_data: torch.Tensor,
                            train_labels: Optional[torch.Tensor] = None,
                            test_labels: Optional[torch.Tensor] = None) -> None:
    """
    Validate consistency between training and test data.
    
    Args:
        train_data: Training data tensor
        test_data: Test data tensor
        train_labels: Training labels (optional)
        test_labels: Test labels (optional)
        
    Raises:
        ValueError: If validation fails
    """
    # Check that data tensors have compatible shapes
    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(f"Train and test data must have same feature dimensions, "
                        f"got train: {train_data.shape[1:]} vs test: {test_data.shape[1:]}")
    
    # Check label consistency if provided
    if train_labels is not None:
        if len(train_labels) != len(train_data):
            raise ValueError(f"Train labels length {len(train_labels)} doesn't match "
                           f"train data length {len(train_data)}")
    
    if test_labels is not None:
        if len(test_labels) != len(test_data):
            raise ValueError(f"Test labels length {len(test_labels)} doesn't match "
                           f"test data length {len(test_data)}")
    
    # Check that label ranges are consistent
    if train_labels is not None and test_labels is not None:
        train_classes = set(train_labels.unique().tolist())
        test_classes = set(test_labels.unique().tolist())
        
        if train_classes != test_classes:
            logging.warning(f"Train and test label classes differ: "
                          f"train={train_classes}, test={test_classes}")


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If validation fails
    """
    validated_config = config.copy()
    
    # Validate input dimensions
    if 'input_dim' in config:
        validated_config['input_dim'] = validate_positive_int(
            config['input_dim'], 'input_dim'
        )
    
    # Validate latent dimension
    if 'latent_dim' in config:
        validated_config['latent_dim'] = validate_positive_int(
            config['latent_dim'], 'latent_dim'
        )
    
    # Validate hidden dimensions
    if 'hidden_dims' in config:
        hidden_dims = config['hidden_dims']
        if not isinstance(hidden_dims, (list, tuple)):
            raise ValueError("hidden_dims must be a list or tuple")
        
        validated_hidden_dims = []
        for i, dim in enumerate(hidden_dims):
            validated_hidden_dims.append(
                validate_positive_int(dim, f'hidden_dims[{i}]')
            )
        validated_config['hidden_dims'] = validated_hidden_dims
    
    # Validate activation function
    if 'activation' in config:
        valid_activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        validated_config['activation'] = validate_choice(
            config['activation'].lower(), 'activation', valid_activations
        )
    
    # Validate dropout rate
    if 'dropout_rate' in config:
        validated_config['dropout_rate'] = validate_positive_float(
            config['dropout_rate'], 'dropout_rate', 0.0, 1.0
        )
    
    return validated_config


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate training configuration parameters.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If validation fails
    """
    validated_config = config.copy()
    
    # Validate epochs
    if 'epochs' in config:
        validated_config['epochs'] = validate_positive_int(
            config['epochs'], 'epochs'
        )
    
    # Validate learning rate
    if 'learning_rate' in config:
        validated_config['learning_rate'] = validate_positive_float(
            config['learning_rate'], 'learning_rate', 1e-10, 1.0
        )
    
    # Validate batch size
    if 'batch_size' in config:
        validated_config['batch_size'] = validate_positive_int(
            config['batch_size'], 'batch_size'
        )
    
    # Validate weight decay
    if 'weight_decay' in config:
        validated_config['weight_decay'] = validate_positive_float(
            config['weight_decay'], 'weight_decay', 0.0
        )
    
    # Validate scheduler parameters
    if 'scheduler_patience' in config:
        validated_config['scheduler_patience'] = validate_positive_int(
            config['scheduler_patience'], 'scheduler_patience'
        )
    
    if 'scheduler_factor' in config:
        validated_config['scheduler_factor'] = validate_positive_float(
            config['scheduler_factor'], 'scheduler_factor', 0.01, 1.0
        )
    
    return validated_config


def validate_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate visualization configuration parameters.
    
    Args:
        config: Visualization configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If validation fails
    """
    validated_config = config.copy()
    
    # Validate samples per class
    if 'samples_per_class' in config:
        validated_config['samples_per_class'] = validate_positive_int(
            config['samples_per_class'], 'samples_per_class'
        )
    
    # Validate visualization interval
    if 'visualization_interval' in config:
        validated_config['visualization_interval'] = validate_positive_int(
            config['visualization_interval'], 'visualization_interval'
        )
    
    # Validate number of visualizations
    if 'num_visualizations' in config:
        validated_config['num_visualizations'] = validate_positive_int(
            config['num_visualizations'], 'num_visualizations'
        )
    
    # Validate perplexity
    if 'perplexity' in config:
        validated_config['perplexity'] = validate_positive_float(
            config['perplexity'], 'perplexity', 5.0, 50.0
        )
    
    # Validate max samples
    if 'max_samples' in config:
        validated_config['max_samples'] = validate_positive_int(
            config['max_samples'], 'max_samples'
        )
    
    return validated_config


def validate_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate complete experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If validation fails
    """
    validated_config = config.copy()
    
    # Validate each section if present
    if 'model' in config:
        validated_config['model'] = validate_model_config(config['model'])
    
    if 'training' in config:
        validated_config['training'] = validate_training_config(config['training'])
    
    if 'visualization' in config:
        validated_config['visualization'] = validate_visualization_config(config['visualization'])
    
    # Validate experiment-specific parameters
    if 'random_seed' in config:
        validated_config['random_seed'] = validate_positive_int(
            config['random_seed'], 'random_seed', 0
        )
    
    # Validate output directory
    if 'output_dir' in config:
        if not isinstance(config['output_dir'], str):
            raise ValueError("output_dir must be a string")
        validated_config['output_dir'] = config['output_dir']
    
    return validated_config


def validate_latent_dimensions(latent_dims: List[int], max_input_dim: Optional[int] = None) -> List[int]:
    """
    Validate latent dimension values.
    
    Args:
        latent_dims: List of latent dimensions to validate
        max_input_dim: Maximum input dimension (for sanity checking)
        
    Returns:
        Validated list of latent dimensions
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(latent_dims, (list, tuple)):
        raise ValueError("latent_dims must be a list or tuple")
    
    validated_dims = []
    for i, dim in enumerate(latent_dims):
        validated_dim = validate_positive_int(dim, f'latent_dims[{i}]')
        
        if max_input_dim is not None and validated_dim > max_input_dim:
            logging.warning(f"Latent dimension {validated_dim} is larger than "
                          f"input dimension {max_input_dim}")
        
        validated_dims.append(validated_dim)
    
    # Check for duplicates
    if len(set(validated_dims)) != len(validated_dims):
        logging.warning("Duplicate latent dimensions found")
    
    return validated_dims


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """
    Validate and normalize device specification.
    
    Args:
        device: Device specification
        
    Returns:
        Validated torch.device
        
    Raises:
        ValueError: If validation fails
    """
    if isinstance(device, str):
        device = device.lower()
        if device in ['cpu', 'cuda']:
            return torch.device(device)
        elif device.startswith('cuda:'):
            return torch.device(device)
        else:
            raise ValueError(f"Invalid device string: {device}")
    elif isinstance(device, torch.device):
        return device
    else:
        raise ValueError(f"Device must be string or torch.device, got {type(device)}")


def validate_file_path(file_path: str, must_exist: bool = False,
                      must_be_writable: bool = False) -> str:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: If True, file must exist
        must_be_writable: If True, directory must be writable
        
    Returns:
        Validated file path
        
    Raises:
        ValueError: If validation fails
    """
    import os
    
    if not isinstance(file_path, str):
        raise ValueError("File path must be a string")
    
    if must_exist and not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    if must_be_writable:
        directory = os.path.dirname(file_path)
        if directory and not os.access(directory, os.W_OK):
            raise ValueError(f"Directory is not writable: {directory}")
    
    return file_path 
"""
Utils Module

Contains helper functions and utilities used across the autoencoder experimentation package.
Includes configuration management, file I/O, logging, reproducibility, and validation utilities.
"""

# Configuration utilities
from .config import (
    ConfigManager,
    create_default_config,
    validate_config,
    merge_configs
)

# File I/O utilities
from .file_io import (
    ensure_dir,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_model,
    load_model,
    save_experiment_results,
    load_experiment_results,
    save_numpy_array,
    load_numpy_array,
    save_dataframe,
    load_dataframe,
    list_experiment_files,
    cleanup_old_files
)

# Logging utilities
from .logging import (
    setup_logging,
    get_experiment_logger,
    ProgressTracker,
    log_experiment_phase,
    ExperimentLogger
)

# Reproducibility utilities
from .reproducibility import (
    set_seed,
    get_random_state,
    set_random_state,
    SeedContext,
    make_deterministic,
    verify_reproducibility,
    create_experiment_seeds
)

# Validation utilities
from .validation import (
    validate_positive_int,
    validate_positive_float,
    validate_choice,
    validate_tensor_shape,
    validate_data_consistency,
    validate_model_config,
    validate_training_config,
    validate_visualization_config,
    validate_experiment_config,
    validate_latent_dimensions,
    validate_device,
    validate_file_path
)

__all__ = [
    # Configuration utilities
    'ConfigManager',
    'create_default_config',
    'validate_config',
    'merge_configs',
    
    # File I/O utilities
    'ensure_dir',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'save_model',
    'load_model',
    'save_experiment_results',
    'load_experiment_results',
    'save_numpy_array',
    'load_numpy_array',
    'save_dataframe',
    'load_dataframe',
    'list_experiment_files',
    'cleanup_old_files',
    
    # Logging utilities
    'setup_logging',
    'get_experiment_logger',
    'ProgressTracker',
    'log_experiment_phase',
    'ExperimentLogger',
    
    # Reproducibility utilities
    'set_seed',
    'get_random_state',
    'set_random_state',
    'SeedContext',
    'make_deterministic',
    'verify_reproducibility',
    'create_experiment_seeds',
    
    # Validation utilities
    'validate_positive_int',
    'validate_positive_float',
    'validate_choice',
    'validate_tensor_shape',
    'validate_data_consistency',
    'validate_model_config',
    'validate_training_config',
    'validate_visualization_config',
    'validate_experiment_config',
    'validate_latent_dimensions',
    'validate_device',
    'validate_file_path'
] 
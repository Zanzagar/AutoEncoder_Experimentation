"""
Reproducibility Utilities

Functions for managing random seeds and ensuring reproducible experiments.
"""

import random
import torch
import numpy as np
import os
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")


def get_random_state() -> dict:
    """
    Get current random state for all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Set random state for all libraries.
    
    Args:
        state: Dictionary containing random states from get_random_state()
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
    
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
    
    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])
    
    if torch.cuda.is_available():
        if 'torch_cuda_random' in state:
            torch.cuda.set_rng_state(state['torch_cuda_random'])
        
        if 'torch_cuda_random_all' in state and torch.cuda.device_count() > 1:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_all'])


class SeedContext:
    """
    Context manager for temporarily setting a random seed.
    
    Example:
        with SeedContext(42):
            # All random operations here use seed 42
            data = torch.randn(10, 10)
        # Original random state is restored
    """
    
    def __init__(self, seed: int):
        """
        Initialize the seed context.
        
        Args:
            seed: Temporary seed to use
        """
        self.seed = seed
        self.original_state = None
    
    def __enter__(self):
        """Save current state and set new seed."""
        self.original_state = get_random_state()
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original random state."""
        if self.original_state is not None:
            set_random_state(self.original_state)


def make_deterministic(seed: int = 42, warn: bool = True) -> None:
    """
    Set up deterministic behavior for PyTorch operations.
    
    Args:
        seed: Random seed to use
        warn: Whether to print warnings about potential performance impact
    """
    set_seed(seed)
    
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True)
    
    if warn:
        print("Warning: Deterministic algorithms may impact performance.")
        print("Set warn=False to suppress this message.")


def verify_reproducibility(func, seed: int = 42, n_trials: int = 3) -> bool:
    """
    Verify that a function produces reproducible results.
    
    Args:
        func: Function to test (should take no arguments)
        seed: Random seed to use for testing
        n_trials: Number of trials to run
        
    Returns:
        True if results are reproducible, False otherwise
    """
    results = []
    
    for trial in range(n_trials):
        with SeedContext(seed):
            result = func()
            if torch.is_tensor(result):
                result = result.clone().detach()
            elif isinstance(result, np.ndarray):
                result = result.copy()
            results.append(result)
    
    # Check if all results are identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        if torch.is_tensor(first_result):
            if not torch.equal(first_result, result):
                print(f"Trial {i+1} differs from trial 1")
                return False
        elif isinstance(first_result, np.ndarray):
            if not np.array_equal(first_result, result):
                print(f"Trial {i+1} differs from trial 1")
                return False
        else:
            if first_result != result:
                print(f"Trial {i+1} differs from trial 1")
                return False
    
    print(f"Function is reproducible across {n_trials} trials")
    return True


def create_experiment_seeds(base_seed: int = 42, n_experiments: int = 5) -> list:
    """
    Create a list of seeds for multiple experiments.
    
    Args:
        base_seed: Base seed for generating experiment seeds
        n_experiments: Number of experiment seeds to generate
        
    Returns:
        List of seeds for experiments
    """
    # Use the base seed to generate consistent experiment seeds
    with SeedContext(base_seed):
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(n_experiments)]
    
    return seeds 
"""
File I/O Utilities

Functions for saving and loading experiment results, models, and datasets.
"""

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
import logging


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        converted_data = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=indent, default=str)
        
        logging.info(f"Saved JSON data to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Loaded JSON data from {filepath}")
        return data
        
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return None


def save_pickle(data: Any, filepath: str) -> bool:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logging.info(f"Saved pickle data to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving pickle to {filepath}: {e}")
        return False


def load_pickle(filepath: str) -> Optional[Any]:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logging.info(f"Loaded pickle data from {filepath}")
        return data
        
    except Exception as e:
        logging.error(f"Error loading pickle from {filepath}: {e}")
        return None


def save_model(model: torch.nn.Module, filepath: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a PyTorch model with optional optimizer and metadata.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model
        optimizer: Optional optimizer state to save
        metadata: Optional metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            save_dict['optimizer_class'] = optimizer.__class__.__name__
        
        if metadata is not None:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, filepath)
        
        logging.info(f"Saved model to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving model to {filepath}: {e}")
        return False


def load_model(model: torch.nn.Module, filepath: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: Optional[torch.device] = None) -> bool:
    """
    Load a PyTorch model with optional optimizer state.
    
    Args:
        model: PyTorch model to load state into
        filepath: Path to the saved model
        optimizer: Optional optimizer to load state into
        device: Device to load the model onto
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logging.info(f"Loaded model from {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error loading model from {filepath}: {e}")
        return False


def save_experiment_results(results: Dict[str, Any], experiment_name: str,
                          output_dir: str = "experiment_results") -> str:
    """
    Save experiment results with automatic timestamp and organization.
    
    Args:
        results: Experiment results dictionary
        experiment_name: Name of the experiment
        output_dir: Base output directory
        
    Returns:
        Path to the saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}_results.json"
    filepath = os.path.join(output_dir, experiment_name, filename)
    
    # Add timestamp to results
    results['timestamp'] = timestamp
    results['experiment_name'] = experiment_name
    
    save_json(results, filepath)
    return filepath


def load_experiment_results(experiment_name: str, 
                          output_dir: str = "experiment_results",
                          latest: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load experiment results.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Base output directory
        latest: If True, load the latest results file
        
    Returns:
        Experiment results dictionary or None if not found
    """
    experiment_dir = os.path.join(output_dir, experiment_name)
    
    if not os.path.exists(experiment_dir):
        logging.warning(f"Experiment directory not found: {experiment_dir}")
        return None
    
    # Find results files
    results_files = [f for f in os.listdir(experiment_dir) 
                    if f.endswith('_results.json')]
    
    if not results_files:
        logging.warning(f"No results files found in {experiment_dir}")
        return None
    
    if latest:
        # Sort by timestamp and get the latest
        results_files.sort(reverse=True)
        filepath = os.path.join(experiment_dir, results_files[0])
    else:
        # Return the first one found
        filepath = os.path.join(experiment_dir, results_files[0])
    
    return load_json(filepath)


def save_numpy_array(array: np.ndarray, filepath: str, 
                     compressed: bool = True) -> bool:
    """
    Save a numpy array to disk.
    
    Args:
        array: Numpy array to save
        filepath: Path to save the array
        compressed: Whether to use compressed format
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        
        if compressed:
            np.savez_compressed(filepath, array=array)
        else:
            np.save(filepath, array)
        
        logging.info(f"Saved numpy array to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving numpy array to {filepath}: {e}")
        return False


def load_numpy_array(filepath: str) -> Optional[np.ndarray]:
    """
    Load a numpy array from disk.
    
    Args:
        filepath: Path to the array file
        
    Returns:
        Loaded numpy array or None if failed
    """
    try:
        if filepath.endswith('.npz'):
            data = np.load(filepath)
            array = data['array']
        else:
            array = np.load(filepath)
        
        logging.info(f"Loaded numpy array from {filepath}")
        return array
        
    except Exception as e:
        logging.error(f"Error loading numpy array from {filepath}: {e}")
        return None


def save_dataframe(df: pd.DataFrame, filepath: str, 
                   format: str = 'csv') -> bool:
    """
    Save a pandas DataFrame to disk.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the DataFrame
        format: Format to save in ('csv', 'parquet', 'excel')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Saved DataFrame to {filepath}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving DataFrame to {filepath}: {e}")
        return False


def load_dataframe(filepath: str, format: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load a pandas DataFrame from disk.
    
    Args:
        filepath: Path to the DataFrame file
        format: Format to load from (auto-detected if None)
        
    Returns:
        Loaded DataFrame or None if failed
    """
    try:
        if format is None:
            # Auto-detect format from extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.xlsx', '.xls']:
                format = 'excel'
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if format.lower() == 'csv':
            df = pd.read_csv(filepath)
        elif format.lower() == 'parquet':
            df = pd.read_parquet(filepath)
        elif format.lower() == 'excel':
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Loaded DataFrame from {filepath}")
        return df
        
    except Exception as e:
        logging.error(f"Error loading DataFrame from {filepath}: {e}")
        return None


def list_experiment_files(output_dir: str = "experiment_results") -> Dict[str, List[str]]:
    """
    List all experiment files in the output directory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dictionary mapping experiment names to lists of files
    """
    experiment_files = {}
    
    if not os.path.exists(output_dir):
        return experiment_files
    
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            experiment_files[item] = files
    
    return experiment_files


def cleanup_old_files(directory: str, max_age_days: int = 30,
                     pattern: str = "*") -> int:
    """
    Clean up old files in a directory.
    
    Args:
        directory: Directory to clean up
        max_age_days: Maximum age of files to keep (in days)
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(directory):
        return 0
    
    import glob
    import time
    
    pattern_path = os.path.join(directory, pattern)
    files = glob.glob(pattern_path)
    
    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    
    for file_path in files:
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.error(f"Error deleting file {file_path}: {e}")
    
    return deleted_count 
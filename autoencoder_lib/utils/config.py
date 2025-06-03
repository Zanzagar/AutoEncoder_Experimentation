"""
Configuration Management Utilities

Handles loading, saving, and validating configuration parameters for autoencoder experiments.
"""

import os
import json
import configparser
from typing import Dict, Any, Optional, Union
import logging


class ConfigManager:
    """
    Manages configuration for autoencoder experiments.
    Supports both JSON and INI file formats.
    """
    
    def __init__(self, config_file: Optional[str] = None, format: str = "json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file
            format: Configuration file format ("json" or "ini")
        """
        self.config_file = config_file
        self.format = format.lower()
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file (optional if set in constructor)
            
        Returns:
            Dictionary containing configuration parameters
        """
        if config_file:
            self.config_file = config_file
        
        if not self.config_file or not os.path.exists(self.config_file):
            logging.warning(f"Configuration file not found: {self.config_file}")
            return {}
        
        try:
            if self.format == "json":
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            elif self.format == "ini":
                parser = configparser.ConfigParser()
                parser.read(self.config_file)
                self.config = {section: dict(parser.items(section)) 
                              for section in parser.sections()}
            else:
                raise ValueError(f"Unsupported configuration format: {self.format}")
            
            logging.info(f"Configuration loaded from {self.config_file}")
            return self.config.copy()
            
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            return {}
    
    def save_config(self, config: Optional[Dict[str, Any]] = None, 
                    config_file: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save (optional if using internal config)
            config_file: Path to save configuration (optional if set in constructor)
            
        Returns:
            True if successful, False otherwise
        """
        if config is not None:
            self.config = config
        
        if config_file:
            self.config_file = config_file
        
        if not self.config_file:
            logging.error("No configuration file specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            if self.format == "json":
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif self.format == "ini":
                parser = configparser.ConfigParser()
                for section, options in self.config.items():
                    parser[section] = options
                with open(self.config_file, 'w') as f:
                    parser.write(f)
            else:
                raise ValueError(f"Unsupported configuration format: {self.format}")
            
            logging.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            section: Section name for INI format
            
        Returns:
            Configuration value or default
        """
        if section and section in self.config:
            return self.config[section].get(key, default)
        elif key in self.config:
            return self.config[key]
        else:
            # Try to find in any section
            for section_data in self.config.values():
                if isinstance(section_data, dict) and key in section_data:
                    return section_data[key]
            return default
    
    def set(self, key: str, value: Any, section: Optional[str] = None) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            section: Section name for INI format
        """
        if section:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
        else:
            self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary of configuration updates
        """
        self.config.update(config_dict)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary containing section configuration
        """
        return self.config.get(section, {})


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for autoencoder experiments.
    
    Returns:
        Dictionary containing default configuration parameters
    """
    return {
        "experiment": {
            "random_seed": 42,
            "output_dir": "experiment_results",
            "save_models": True,
            "save_visualizations": True
        },
        "training": {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32,
            "weight_decay": 1e-5,
            "scheduler_patience": 3,
            "scheduler_factor": 0.5
        },
        "visualization": {
            "samples_per_class": 2,
            "visualization_interval": 500,
            "num_visualizations": 5,
            "perplexity": 30,
            "max_samples": 500
        },
        "model": {
            "latent_dims": [2, 4, 8, 16, 32],
            "architectures": ["SimpleLinear", "DeeperLinear", "Conv", "DeeperConv"]
        },
        "evaluation": {
            "calculate_train_silhouette": True,
            "calculate_test_silhouette": True,
            "evaluation_batch_size": 64
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check required sections
        required_sections = ["experiment", "training", "visualization"]
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate specific parameters
        training = config.get("training", {})
        if "epochs" in training and training["epochs"] <= 0:
            logging.error("Training epochs must be positive")
            return False
        
        if "learning_rate" in training and training["learning_rate"] <= 0:
            logging.error("Learning rate must be positive")
            return False
        
        if "batch_size" in training and training["batch_size"] <= 0:
            logging.error("Batch size must be positive")
            return False
        
        # Validate visualization parameters
        visualization = config.get("visualization", {})
        if "samples_per_class" in visualization and visualization["samples_per_class"] <= 0:
            logging.error("Samples per class must be positive")
            return False
        
        logging.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Error validating configuration: {e}")
        return False


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged 
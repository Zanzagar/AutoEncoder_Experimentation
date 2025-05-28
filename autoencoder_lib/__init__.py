"""
AutoEncoder Experimentation Library

A Python package for autoencoder neural network experimentation and analysis.
Provides tools for dataset generation, model training, visualization, and analysis.
"""

__version__ = "0.1.0"
__author__ = "AutoEncoder Experimentation Project"

# Import main modules
from . import data
from . import models
from . import experiment
from . import visualization
from . import utils

__all__ = [
    "data",
    "models", 
    "experiment",
    "visualization",
    "utils"
] 
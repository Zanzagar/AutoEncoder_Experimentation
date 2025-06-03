"""
Logging Utilities

Functions for setting up logging and progress tracking for experiments.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager


def setup_logging(level: str = "INFO", 
                  log_file: Optional[str] = None,
                  format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration for experiments.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to save logs
        format_string: Custom format string for log messages
    """
    # Convert string level to logging constant
    level_dict = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_dict.get(level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")


def get_experiment_logger(experiment_name: str, 
                         output_dir: str = "experiment_results") -> logging.Logger:
    """
    Get a logger specifically configured for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Base output directory
        
    Returns:
        Configured logger for the experiment
    """
    logger_name = f"experiment.{experiment_name}"
    logger = logging.getLogger(logger_name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create experiment-specific log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = os.path.join(output_dir, experiment_name, "logs", log_filename)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    # File handler for experiment
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set logger level
    logger.setLevel(logging.INFO)
    
    logger.info(f"Experiment logger initialized for {experiment_name}")
    return logger


class ProgressTracker:
    """
    Track progress for long-running operations with logging.
    """
    
    def __init__(self, total_steps: int, description: str = "Progress",
                 log_interval: int = 10, logger: Optional[logging.Logger] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps expected
            description: Description of the operation
            log_interval: How often to log progress (every N steps)
            logger: Logger to use (creates new one if None)
        """
        self.total_steps = total_steps
        self.description = description
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = datetime.now()
        
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.logger.info(f"Starting {description} - Total steps: {total_steps}")
    
    def update(self, step: Optional[int] = None, message: Optional[str] = None) -> None:
        """
        Update progress.
        
        Args:
            step: Current step number (auto-increment if None)
            message: Optional additional message
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Log progress at intervals
        if self.current_step % self.log_interval == 0 or self.current_step == self.total_steps:
            progress_pct = (self.current_step / self.total_steps) * 100
            elapsed = datetime.now() - self.start_time
            
            # Estimate remaining time
            if self.current_step > 0:
                time_per_step = elapsed.total_seconds() / self.current_step
                remaining_steps = self.total_steps - self.current_step
                estimated_remaining = remaining_steps * time_per_step
                
                remaining_str = f", ETA: {estimated_remaining:.1f}s"
            else:
                remaining_str = ""
            
            log_msg = (f"{self.description}: {self.current_step}/{self.total_steps} "
                      f"({progress_pct:.1f}%) - Elapsed: {elapsed.total_seconds():.1f}s{remaining_str}")
            
            if message:
                log_msg += f" - {message}"
            
            self.logger.info(log_msg)
    
    def finish(self, message: Optional[str] = None) -> None:
        """
        Mark progress as finished.
        
        Args:
            message: Optional completion message
        """
        total_time = datetime.now() - self.start_time
        
        completion_msg = (f"{self.description} completed: {self.total_steps} steps "
                         f"in {total_time.total_seconds():.1f}s")
        
        if message:
            completion_msg += f" - {message}"
        
        self.logger.info(completion_msg)


@contextmanager
def log_experiment_phase(phase_name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for logging experiment phases.
    
    Args:
        phase_name: Name of the experiment phase
        logger: Logger to use
        
    Yields:
        The logger being used
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    logger.info(f"Starting {phase_name}")
    
    try:
        yield logger
        
    except Exception as e:
        logger.error(f"Error in {phase_name}: {e}")
        raise
        
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Completed {phase_name} in {duration.total_seconds():.2f}s")


class ExperimentLogger:
    """
    Specialized logger for experiment tracking with structured logging.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "experiment_results"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Base output directory
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.logger = get_experiment_logger(experiment_name, output_dir)
        self.metrics = {}
        self.phase_times = {}
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Experiment Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {'value': value, 'timestamp': datetime.now()}
        if step is not None:
            metric_entry['step'] = step
        
        self.metrics[name].append(metric_entry)
        
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metric - {name}: {value}{step_str}")
    
    def log_model_info(self, model, total_params: Optional[int] = None) -> None:
        """
        Log model information.
        
        Args:
            model: PyTorch model
            total_params: Total parameter count (calculated if None)
        """
        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info("Model Information:")
        self.logger.info(f"  Architecture: {model.__class__.__name__}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a phase of the experiment.
        
        Args:
            phase_name: Name of the phase
        """
        self.phase_times[phase_name] = {'start': datetime.now()}
        self.logger.info(f"Phase started: {phase_name}")
    
    def end_phase(self, phase_name: str) -> None:
        """
        End tracking a phase of the experiment.
        
        Args:
            phase_name: Name of the phase
        """
        if phase_name in self.phase_times:
            end_time = datetime.now()
            self.phase_times[phase_name]['end'] = end_time
            duration = end_time - self.phase_times[phase_name]['start']
            self.logger.info(f"Phase completed: {phase_name} ({duration.total_seconds():.2f}s)")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all logged metrics.
        
        Returns:
            Dictionary containing metrics summary
        """
        summary = {}
        for metric_name, metric_data in self.metrics.items():
            values = [entry['value'] for entry in metric_data]
            summary[metric_name] = {
                'count': len(values),
                'latest': values[-1] if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'avg': sum(values) / len(values) if values else None
            }
        
        return summary 
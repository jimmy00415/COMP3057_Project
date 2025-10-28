"""Centralized logging configuration."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs JSON structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record
        
        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class MetricsLogger:
    """Logger for training and evaluation metrics."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to save metric logs
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.logger = logging.getLogger("metrics")
    
    def log_training_step(self, 
                         epoch: int, 
                         step: int, 
                         loss: float,
                         learning_rate: float,
                         **kwargs):
        """Log training step metrics.
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics
        """
        metrics = {
            'type': 'training_step',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'loss': float(loss),
            'learning_rate': float(learning_rate),
            **kwargs
        }
        
        self._write_metrics(metrics)
    
    def log_evaluation(self, 
                      epoch: int,
                      wer: float,
                      cer: float,
                      **kwargs):
        """Log evaluation metrics.
        
        Args:
            epoch: Current epoch
            wer: Word Error Rate
            cer: Character Error Rate
            **kwargs: Additional metrics
        """
        metrics = {
            'type': 'evaluation',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'wer': float(wer),
            'cer': float(cer),
            **kwargs
        }
        
        self._write_metrics(metrics)
        self.logger.info(f"Epoch {epoch} - WER: {wer:.4f}, CER: {cer:.4f}")
    
    def log_memory(self, 
                   ram_used_gb: float,
                   ram_percent: float,
                   gpu_used_gb: Optional[float] = None,
                   gpu_percent: Optional[float] = None):
        """Log memory usage.
        
        Args:
            ram_used_gb: RAM used in GB
            ram_percent: RAM usage percentage
            gpu_used_gb: GPU memory used in GB
            gpu_percent: GPU memory usage percentage
        """
        metrics = {
            'type': 'memory',
            'timestamp': datetime.now().isoformat(),
            'ram_used_gb': float(ram_used_gb),
            'ram_percent': float(ram_percent),
        }
        
        if gpu_used_gb is not None:
            metrics['gpu_used_gb'] = float(gpu_used_gb)
        if gpu_percent is not None:
            metrics['gpu_percent'] = float(gpu_percent)
        
        self._write_metrics(metrics)
    
    def _write_metrics(self, metrics: dict):
        """Write metrics to file.
        
        Args:
            metrics: Metrics dictionary
        """
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write metrics: {e}")


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    use_structured: bool = False,
    module_levels: Optional[dict] = None
) -> MetricsLogger:
    """Setup centralized logging configuration.
    
    Args:
        log_level: Default log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        use_structured: Whether to use JSON structured logging
        module_levels: Dict of module-specific log levels
    
    Returns:
        MetricsLogger instance
    """
    # Create log directory
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if use_structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log_dir specified
    if log_dir:
        file_handler = logging.FileHandler(
            log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        if use_structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set module-specific levels
    if module_levels:
        for module, level in module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, level.upper()))
    
    # Create metrics logger
    metrics_logger = MetricsLogger(log_dir)
    
    logging.info("Logging configured successfully")
    
    return metrics_logger


# Convenience function for performance logging
def log_performance(func):
    """Decorator to log function performance.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper

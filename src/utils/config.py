"""Utility functions for configuration, logging, and reproducibility."""

import os
import yaml
import random
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def get_git_revision() -> str:
    """Get current git commit hash for reproducibility."""
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return git_hash
    except:
        return "unknown"


def get_device(device: str = "cuda") -> torch.device:
    """Get PyTorch device with fallback."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, metrics, path: str):
    """Save model checkpoint with metadata."""
    ensure_dir(os.path.dirname(path))
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'git_revision': get_git_revision()
    }
    
    torch.save(checkpoint, path)
    

def load_checkpoint(path: str, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


class ExperimentLogger:
    """Unified interface for experiment tracking."""
    
    def __init__(self, backend: str = "wandb", project_name: str = "whisper-asr", config: Dict = None):
        self.backend = backend
        self.config = config or {}
        
        if backend == "wandb":
            import wandb
            self.run = wandb.init(project=project_name, config=config)
        elif backend == "mlflow":
            import mlflow
            mlflow.set_experiment(project_name)
            mlflow.start_run()
            if config:
                mlflow.log_params(self._flatten_dict(config))
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"runs/{project_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to tracking backend."""
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step or 0)
    
    def log_artifact(self, path: str):
        """Log file artifact."""
        if self.backend == "wandb":
            import wandb
            wandb.log_artifact(path)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_artifact(path)
    
    def finish(self):
        """Finish tracking session."""
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
        elif self.backend == "tensorboard":
            self.writer.close()
    
    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentLogger._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

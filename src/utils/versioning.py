"""Versioning utilities for data and models."""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class DataVersionManager:
    """Manages data versioning and tracking."""
    
    def __init__(self, version_file: str = "data_versions.txt"):
        self.version_file = version_file
        
    def compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def log_dataset_version(self, dataset_name: str, filepath: str = None, 
                           version: str = None, metadata: Dict = None):
        """Log dataset version to tracking file."""
        entry = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'version': version or 'v1',
        }
        
        if filepath and os.path.exists(filepath):
            entry['hash'] = self.compute_file_hash(filepath)
            entry['filepath'] = filepath
        
        if metadata:
            entry['metadata'] = metadata
        
        with open(self.version_file, 'a') as f:
            f.write(f"{json.dumps(entry)}\n")
    
    def get_versions(self) -> List[Dict]:
        """Retrieve all logged dataset versions."""
        versions = []
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                for line in f:
                    versions.append(json.loads(line.strip()))
        return versions


class ModelRegistry:
    """Registry for tracking trained models."""
    
    def __init__(self, registry_file: str = "model_registry.json"):
        self.registry_file = registry_file
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'models': []}
    
    def _save_registry(self):
        """Save registry to disk."""
        os.makedirs(os.path.dirname(self.registry_file) or '.', exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_id: str, model_path: str, 
                      metrics: Dict, config: Dict, 
                      git_revision: str = None, dataset_version: str = None):
        """Register a trained model."""
        model_entry = {
            'model_id': model_id,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config,
            'git_revision': git_revision,
            'dataset_version': dataset_version
        }
        
        self.registry['models'].append(model_entry)
        self._save_registry()
        
        return model_id
    
    def get_best_model(self, metric: str = 'wer', mode: str = 'min'):
        """Retrieve best model based on metric."""
        if not self.registry['models']:
            return None
        
        models = [m for m in self.registry['models'] if metric in m.get('metrics', {})]
        
        if not models:
            return None
        
        if mode == 'min':
            best = min(models, key=lambda x: x['metrics'][metric])
        else:
            best = max(models, key=lambda x: x['metrics'][metric])
        
        return best
    
    def list_models(self) -> List[Dict]:
        """List all registered models."""
        return self.registry['models']

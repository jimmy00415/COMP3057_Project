"""Whisper model initialization and configuration."""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig
)
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class WhisperModelManager:
    """Manages Whisper model initialization and configuration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.processor = None
        self.model_name = None
    
    def initialize_model(self, variant: str = None, device: str = "cuda"):
        """Initialize Whisper model and processor."""
        if variant is None:
            variant = self.config['model']['default_variant']
        
        self.model_name = self.config['model']['variants'].get(variant)
        
        if self.model_name is None:
            raise ValueError(f"Unknown model variant: {variant}")
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Load processor
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        
        # Load model
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        
        # Configure forced decoder IDs for English transcription
        language = self.config['model']['language']
        task = self.config['model']['task']
        
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
        
        # Additional configurations
        self.model.config.suppress_tokens = []
        self.model.config.use_cache = True
        
        # Move to device
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        logger.info(f"Model loaded on {device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
        
        return self.model, self.processor
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def freeze_encoder(self):
        """Freeze encoder parameters for faster fine-tuning."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False
        
        logger.info("Encoder frozen")
        logger.info(f"Trainable parameters: {self.count_parameters():,}")
    
    def freeze_layers(self, n_layers: int):
        """Freeze first N encoder layers."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        for i, layer in enumerate(self.model.model.encoder.layers):
            if i < n_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        logger.info(f"Frozen {n_layers} encoder layers")
        logger.info(f"Trainable parameters: {self.count_parameters():,}")
    
    def prepare_for_training(self):
        """Prepare model for fine-tuning."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Apply freezing if configured
        if self.config['model'].get('freeze_encoder', False):
            self.freeze_encoder()
        elif self.config['model'].get('freeze_layers', 0) > 0:
            self.freeze_layers(self.config['model']['freeze_layers'])
        
        # Enable gradient checkpointing for memory efficiency
        self.model.config.use_cache = False
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model information summary."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.model_name,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': self.count_parameters(),
            'config': self.model.config.to_dict()
        }


def load_pretrained_model(model_path: str, device: str = "cuda"):
    """Load a fine-tuned model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, processor


def compare_models(config: Dict) -> Dict:
    """Generate comparison table of Whisper variants."""
    variants = {
        'tiny': {'params': '39M', 'speed': 'fastest', 'accuracy': 'lower'},
        'base': {'params': '74M', 'speed': 'fast', 'accuracy': 'good'},
        'small': {'params': '244M', 'speed': 'moderate', 'accuracy': 'better'},
        'medium': {'params': '769M', 'speed': 'slow', 'accuracy': 'best'},
        'distil': {'params': '~756M', 'speed': 'moderate', 'accuracy': 'best'}
    }
    
    return variants

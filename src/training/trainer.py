"""Training loop and fine-tuning utilities."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from typing import Dict, Optional
import logging
import os


logger = logging.getLogger(__name__)


class WhisperTrainer:
    """Handles Whisper model training."""
    
    def __init__(self, 
                 model,
                 processor,
                 train_loader,
                 val_loader,
                 config: Dict,
                 device: str = "cuda",
                 experiment_logger=None):
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.experiment_logger = experiment_logger
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.use_amp = config['training'].get('fp16', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training (FP16) enabled")
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
    
    def _setup_optimizer(self):
        """Initialize optimizer."""
        training_config = self.config['training']
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        logger.info(f"Optimizer: AdamW (lr={training_config['learning_rate']})")
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        training_config = self.config['training']
        
        total_steps = len(self.train_loader) * training_config['num_epochs']
        warmup_steps = training_config.get('warmup_steps', 500)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Scheduler: Linear warmup ({warmup_steps} steps) + decay")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize gradients before first backward
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with automatic mixed precision
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation (outside autocast)
            scaled_loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Track metrics (detach to avoid keeping graph)
            total_loss += loss.detach().item()
            
            # Gradient clipping and optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping (if using AMP)
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.config['training'].get('max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            num_batches += 1
            
            # Update progress bar (use detached loss)
            progress_bar.set_postfix({'loss': scaled_loss.detach().item()})
            
            # Log to experiment tracker
            if self.experiment_logger and self.global_step % 10 == 0:
                self.experiment_logger.log_metrics({
                    'train/loss': loss.detach().item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                }, step=self.global_step)
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def train(self, num_epochs: int = None):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}")
            
            # Log to experiment tracker
            if self.experiment_logger:
                self.experiment_logger.log_metrics({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'val/loss': val_metrics['loss']
                }, step=self.global_step)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_steps', 1000) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                logger.info(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['training'].get('early_stopping_patience', 3):
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("Training completed!")
        return self.best_val_loss
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save as HuggingFace format
        model_dir = checkpoint_path.replace('.pt', '_hf')
        self.model.save_pretrained(model_dir)
        self.processor.save_pretrained(model_dir)
        logger.info(f"Model saved in HuggingFace format: {model_dir}")
        
        # Cleanup old checkpoints to save disk space
        save_total_limit = self.config['training'].get('save_total_limit', None)
        if save_total_limit is not None and 'best_model' not in filename:
            self._cleanup_checkpoints(checkpoint_dir, save_total_limit)
    
    def _cleanup_checkpoints(self, checkpoint_dir: str, save_total_limit: int):
        """Remove old checkpoints keeping only the most recent ones."""
        import glob
        
        # Get all checkpoint files (excluding best_model)
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
        checkpoint_files.sort(key=os.path.getmtime)
        
        # Remove oldest checkpoints if we exceed the limit
        if len(checkpoint_files) > save_total_limit:
            files_to_remove = checkpoint_files[:-save_total_limit]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    # Also remove corresponding HF directory
                    hf_dir = file_path.replace('.pt', '_hf')
                    if os.path.exists(hf_dir):
                        import shutil
                        shutil.rmtree(hf_dir)
                    logger.info(f"Removed old checkpoint: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")

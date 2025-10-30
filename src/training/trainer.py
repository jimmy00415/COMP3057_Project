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

# Import new utilities
from src.utils.memory import MemoryManager
from src.utils.logging_config import MetricsLogger


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
        self.scaler = GradScaler(
            init_scale=2.**10,  # Lower initial scale for stability
            growth_interval=2000  # Slower growth
        ) if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training (FP16) enabled with conservative scaling")
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(threshold_gb=0.85)
        self.memory_manager.log_memory_usage("Training init: ")
        
        # Initialize metrics logger
        log_dir = config.get('log_dir', 'logs')
        self.metrics_logger = MetricsLogger(log_dir)
        
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
            try:
                # Move batch to device
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Check for NaN/Inf in inputs
                if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                    logger.error(f"NaN/Inf detected in input features at batch {batch_idx}")
                    continue
                
                # Context manager for gradient checkpointing compatibility
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps != 0
                
                # Forward pass with automatic mixed precision
                with autocast(device_type='cuda', enabled=self.use_amp):
                    outputs = self.model(input_features=input_features, labels=labels)
                    loss = outputs.loss / gradient_accumulation_steps
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                    self.optimizer.zero_grad()
                    continue
                
                # Backward pass - detach loss from graph immediately after backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Track metrics - store loss value before potential graph cleanup
                loss_value = loss.detach().item() * gradient_accumulation_steps
                total_loss += loss_value
                
                # Gradient clipping and optimizer step
                if not is_accumulation_step:
                    # Unscale gradients FIRST if using AMP
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    # Check gradients for NaN/Inf AFTER unscaling
                    has_inf_or_nan = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_inf_or_nan = True
                                break
                    
                    if has_inf_or_nan:
                        logger.error(f"NaN/Inf gradients detected at batch {batch_idx}, skipping update")
                        self.optimizer.zero_grad()
                        if self.use_amp:
                            self.scaler.update()  # Update scaler even on skip
                        continue
                    
                    # Gradient clipping
                    max_grad_norm = self.config['training'].get('max_grad_norm')
                    if max_grad_norm:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )
                        # Detect gradient explosion BEFORE clipping
                        if grad_norm > max_grad_norm * 20:
                            logger.warning(f"Extreme gradient norm: {grad_norm:.2f}, skipping")
                            self.optimizer.zero_grad()
                            if self.use_amp:
                                self.scaler.update()
                            continue
                    
                    # Optimizer step
                    if self.use_amp:
                        # Check if step will be skipped due to inf/nan
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss_value / gradient_accumulation_steps})
                
                # Memory management - check every 10 batches
                if batch_idx % 10 == 0:
                    self.memory_manager.check_and_cleanup()
                
                # Log to experiment tracker and metrics logger
                if self.experiment_logger and self.global_step % 10 == 0:
                    self.experiment_logger.log_metrics({
                        'train/loss': loss_value,
                        'train/learning_rate': self.scheduler.get_last_lr()[0]
                    }, step=self.global_step)
                
                # Log training step to metrics logger
                if self.global_step % 10 == 0:
                    self.metrics_logger.log_training_step(
                        epoch=self.current_epoch,
                        step=self.global_step,
                        loss=loss_value,
                        learning_rate=self.scheduler.get_last_lr()[0]
                    )
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at batch {batch_idx}, clearing cache and continuing")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue
                else:
                    logger.error(f"Runtime error at batch {batch_idx}: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error at batch {batch_idx}: {e}")
                raise
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self, compute_wer_cer: bool = False) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            compute_wer_cer: Whether to compute WER/CER (slower but more informative)
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # For WER/CER calculation
        all_predictions = []
        all_references = []
        
        # Log memory before validation
        self.memory_manager.log_memory_usage("Validation start: ")
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_features=input_features, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # Generate predictions for WER/CER if requested
                if compute_wer_cer:
                    predicted_ids = self.model.generate(input_features, max_length=225)
                    predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                    
                    # Decode reference labels
                    labels[labels == -100] = self.processor.tokenizer.pad_token_id
                    references = self.processor.batch_decode(labels, skip_special_tokens=True)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
        
        avg_loss = total_loss / num_batches
        metrics = {'loss': avg_loss}
        
        # Calculate WER/CER if requested
        if compute_wer_cer and all_predictions:
            from jiwer import wer, cer
            import re
            
            # Text normalization function
            def normalize_text(text):
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            # Normalize all texts
            norm_predictions = [normalize_text(p) for p in all_predictions]
            norm_references = [normalize_text(r) for r in all_references]
            
            # Calculate metrics
            try:
                wer_score = wer(norm_references, norm_predictions)
                cer_score = cer(norm_references, norm_predictions)
                metrics['wer'] = wer_score
                metrics['cer'] = cer_score
            except Exception as e:
                logger.warning(f"Failed to calculate WER/CER: {e}")
        
        # Log memory after validation
        mem_info = self.memory_manager.get_memory_info()
        self.metrics_logger.log_memory(
            ram_used_gb=mem_info['ram_used_gb'],
            ram_percent=mem_info['ram_percent'],
            gpu_used_gb=mem_info.get('gpu_used_gb'),
            gpu_percent=mem_info.get('gpu_percent')
        )
        
        return metrics
    
    def train(self, num_epochs: int = None, compute_metrics_every: int = 1):
        """Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            compute_metrics_every: Compute WER/CER every N epochs (0 to disable)
        
        Returns:
            Best validation loss achieved
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"WER/CER will be computed every {compute_metrics_every} epoch(s)" if compute_metrics_every > 0 else "WER/CER computation disabled")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_metrics['loss']:.4f}")
            
            # Validate - compute WER/CER periodically
            should_compute_wer_cer = compute_metrics_every > 0 and (epoch + 1) % compute_metrics_every == 0
            val_metrics = self.validate(compute_wer_cer=should_compute_wer_cer)
            
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_metrics['loss']:.4f}"
            if 'wer' in val_metrics:
                log_msg += f", WER: {val_metrics['wer']:.4f}, CER: {val_metrics['cer']:.4f}"
            logger.info(log_msg)
            
            # Log to experiment tracker
            metrics_to_log = {
                'epoch': epoch + 1,
                'train/epoch_loss': train_metrics['loss'],
                'val/loss': val_metrics['loss']
            }
            if 'wer' in val_metrics:
                metrics_to_log['val/wer'] = val_metrics['wer']
                metrics_to_log['val/cer'] = val_metrics['cer']
                
            if self.experiment_logger:
                self.experiment_logger.log_metrics(metrics_to_log, step=self.global_step)
            
            # Log to metrics logger
            if 'wer' in val_metrics:
                self.metrics_logger.log_evaluation(
                    epoch=epoch + 1,
                    wer=val_metrics['wer'],
                    cer=val_metrics['cer'],
                    val_loss=val_metrics['loss']
                )
            
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
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        return self.best_val_loss
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint with validation."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        temp_checkpoint_path = checkpoint_path + ".tmp"
        
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
                'config': self.config,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
            }
            
            # Save to temporary file first
            torch.save(checkpoint, temp_checkpoint_path)
            
            # Validate checkpoint can be loaded
            try:
                test_load = torch.load(temp_checkpoint_path, map_location='cpu')
                if 'model_state_dict' not in test_load:
                    raise ValueError("Checkpoint missing model_state_dict")
            except Exception as e:
                logger.error(f"Checkpoint validation failed: {e}")
                if os.path.exists(temp_checkpoint_path):
                    os.remove(temp_checkpoint_path)
                raise
            
            # Atomic rename
            if os.path.exists(checkpoint_path):
                os.replace(temp_checkpoint_path, checkpoint_path)
            else:
                os.rename(temp_checkpoint_path, checkpoint_path)
                
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filename}: {e}")
            if os.path.exists(temp_checkpoint_path):
                os.remove(temp_checkpoint_path)
            raise
        
        # Also save as HuggingFace format
        try:
            model_dir = checkpoint_path.replace('.pt', '_hf')
            self.model.save_pretrained(model_dir)
            self.processor.save_pretrained(model_dir)
            logger.info(f"Model saved in HuggingFace format: {model_dir}")
        except Exception as e:
            logger.error(f"Failed to save HuggingFace format: {e}")
        
        # Cleanup old checkpoints to save disk space
        save_total_limit = self.config['training'].get('save_total_limit', None)
        if save_total_limit is not None and 'best_model' not in filename:
            try:
                self._cleanup_checkpoints(checkpoint_dir, save_total_limit)
            except Exception as e:
                logger.error(f"Checkpoint cleanup failed: {e}")
    
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

"""Learning curve visualization for training monitoring."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LearningCurveVisualizer:
    """Visualize training metrics and learning curves."""
    
    def __init__(self, metrics_file: Optional[str] = None):
        """Initialize learning curve visualizer.
        
        Args:
            metrics_file: Path to metrics JSONL file
        """
        self.metrics_file = metrics_file
        self.training_data = []
        self.evaluation_data = []
        
        if metrics_file and Path(metrics_file).exists():
            self.load_metrics(metrics_file)
    
    def load_metrics(self, metrics_file: str):
        """Load metrics from JSONL file.
        
        Args:
            metrics_file: Path to metrics file
        """
        self.training_data = []
        self.evaluation_data = []
        
        try:
            with open(metrics_file, 'r') as f:
                for line in f:
                    try:
                        metric = json.loads(line.strip())
                        if metric.get('type') == 'training_step':
                            self.training_data.append(metric)
                        elif metric.get('type') == 'evaluation':
                            self.evaluation_data.append(metric)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(self.training_data)} training steps and {len(self.evaluation_data)} evaluation points")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def plot_learning_curves(self, 
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 10),
                            smooth_window: int = 10) -> plt.Figure:
        """Plot comprehensive learning curves.
        
        Args:
            save_path: Path to save figure (if None, displays only)
            figsize: Figure size (width, height)
            smooth_window: Window size for smoothing training loss
        
        Returns:
            Matplotlib figure object
        """
        if not self.training_data and not self.evaluation_data:
            raise ValueError("No metrics data loaded. Call load_metrics() first.")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Training Loss over Steps
        if self.training_data:
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_training_loss(ax1, smooth_window)
        
        # 2. Validation Loss over Epochs
        if self.evaluation_data:
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_validation_loss(ax2)
        
        # 3. WER/CER over Epochs
        if self.evaluation_data and any('wer' in d for d in self.evaluation_data):
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_wer_cer(ax3)
        
        # 4. Learning Rate Schedule
        if self.training_data:
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_learning_rate(ax4)
        
        # 5. Train vs Val Loss Comparison
        if self.training_data and self.evaluation_data:
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_train_val_comparison(ax5)
        
        plt.suptitle('Training Learning Curves', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        return fig
    
    def _plot_training_loss(self, ax: plt.Axes, smooth_window: int):
        """Plot training loss over steps."""
        steps = [d['step'] for d in self.training_data]
        losses = [d['loss'] for d in self.training_data]
        
        # Raw loss
        ax.plot(steps, losses, alpha=0.3, color='#3498db', linewidth=0.5, label='Raw')
        
        # Smoothed loss
        if len(losses) > smooth_window:
            smoothed = np.convolve(losses, np.ones(smooth_window)/smooth_window, mode='valid')
            smoothed_steps = steps[smooth_window-1:]
            ax.plot(smoothed_steps, smoothed, color='#2c3e50', linewidth=2, label=f'Smoothed (window={smooth_window})')
        
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
        ax.set_title('Training Loss Progression', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_validation_loss(self, ax: plt.Axes):
        """Plot validation loss over epochs."""
        epochs = [d['epoch'] for d in self.evaluation_data]
        val_losses = [d.get('val_loss', d.get('loss', 0)) for d in self.evaluation_data]
        
        ax.plot(epochs, val_losses, marker='o', color='#e74c3c', linewidth=2, markersize=8, label='Validation Loss')
        
        # Mark best epoch
        if val_losses:
            best_idx = np.argmin(val_losses)
            ax.plot(epochs[best_idx], val_losses[best_idx], marker='*', 
                   markersize=20, color='#27ae60', label=f'Best (Epoch {epochs[best_idx]})')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Validation Loss over Epochs', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_wer_cer(self, ax: plt.Axes):
        """Plot WER and CER over epochs."""
        epochs = [d['epoch'] for d in self.evaluation_data if 'wer' in d]
        wers = [d['wer'] for d in self.evaluation_data if 'wer' in d]
        cers = [d['cer'] for d in self.evaluation_data if 'cer' in d]
        
        if epochs:
            ax.plot(epochs, wers, marker='o', color='#3498db', linewidth=2, markersize=8, label='WER')
            ax.plot(epochs, cers, marker='s', color='#9b59b6', linewidth=2, markersize=8, label='CER')
            
            # Mark best WER
            best_wer_idx = np.argmin(wers)
            ax.plot(epochs[best_wer_idx], wers[best_wer_idx], marker='*', 
                   markersize=20, color='#27ae60', label=f'Best WER (Epoch {epochs[best_wer_idx]})')
            
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
            ax.set_title('WER & CER over Epochs', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
    
    def _plot_learning_rate(self, ax: plt.Axes):
        """Plot learning rate schedule."""
        steps = [d['step'] for d in self.training_data]
        lrs = [d['learning_rate'] for d in self.training_data]
        
        ax.plot(steps, lrs, color='#f39c12', linewidth=2)
        ax.set_xlabel('Training Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
    
    def _plot_train_val_comparison(self, ax: plt.Axes):
        """Plot training vs validation loss comparison."""
        # Get epoch-level training loss (average per epoch)
        train_losses_by_epoch = {}
        for d in self.training_data:
            epoch = d['epoch']
            if epoch not in train_losses_by_epoch:
                train_losses_by_epoch[epoch] = []
            train_losses_by_epoch[epoch].append(d['loss'])
        
        train_epochs = sorted(train_losses_by_epoch.keys())
        train_losses = [np.mean(train_losses_by_epoch[e]) for e in train_epochs]
        
        # Validation losses
        val_epochs = [d['epoch'] for d in self.evaluation_data]
        val_losses = [d.get('val_loss', d.get('loss', 0)) for d in self.evaluation_data]
        
        ax.plot(train_epochs, train_losses, marker='o', color='#3498db', 
               linewidth=2, markersize=6, label='Train Loss (avg)')
        ax.plot(val_epochs, val_losses, marker='s', color='#e74c3c', 
               linewidth=2, markersize=6, label='Validation Loss')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Train vs Validation Loss', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        if self.training_data:
            losses = [d['loss'] for d in self.training_data]
            stats['train'] = {
                'total_steps': len(self.training_data),
                'final_loss': losses[-1] if losses else None,
                'min_loss': min(losses) if losses else None,
                'max_loss': max(losses) if losses else None,
                'avg_loss': np.mean(losses) if losses else None
            }
        
        if self.evaluation_data:
            val_losses = [d.get('val_loss', d.get('loss', 0)) for d in self.evaluation_data]
            epochs = [d['epoch'] for d in self.evaluation_data]
            
            stats['validation'] = {
                'total_epochs': max(epochs) if epochs else 0,
                'final_loss': val_losses[-1] if val_losses else None,
                'best_loss': min(val_losses) if val_losses else None,
                'best_epoch': epochs[np.argmin(val_losses)] if val_losses else None
            }
            
            # WER/CER stats if available
            wers = [d['wer'] for d in self.evaluation_data if 'wer' in d]
            cers = [d['cer'] for d in self.evaluation_data if 'cer' in d]
            
            if wers:
                wer_epochs = [d['epoch'] for d in self.evaluation_data if 'wer' in d]
                stats['metrics'] = {
                    'final_wer': wers[-1],
                    'final_cer': cers[-1] if cers else None,
                    'best_wer': min(wers),
                    'best_cer': min(cers) if cers else None,
                    'best_wer_epoch': wer_epochs[np.argmin(wers)]
                }
        
        return stats
    
    def print_summary(self):
        """Print formatted summary of training metrics."""
        stats = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        if 'train' in stats:
            print(f"\n📊 Training:")
            print(f"  • Total Steps: {stats['train']['total_steps']}")
            print(f"  • Final Loss: {stats['train']['final_loss']:.4f}")
            print(f"  • Best Loss: {stats['train']['min_loss']:.4f}")
            print(f"  • Average Loss: {stats['train']['avg_loss']:.4f}")
        
        if 'validation' in stats:
            print(f"\n📈 Validation:")
            print(f"  • Total Epochs: {stats['validation']['total_epochs']}")
            print(f"  • Final Loss: {stats['validation']['final_loss']:.4f}")
            print(f"  • Best Loss: {stats['validation']['best_loss']:.4f}")
            print(f"  • Best Epoch: {stats['validation']['best_epoch']}")
        
        if 'metrics' in stats:
            print(f"\n🎯 Performance Metrics:")
            print(f"  • Final WER: {stats['metrics']['final_wer']:.4f}")
            print(f"  • Final CER: {stats['metrics']['final_cer']:.4f}")
            print(f"  • Best WER: {stats['metrics']['best_wer']:.4f} (Epoch {stats['metrics']['best_wer_epoch']})")
            print(f"  • Best CER: {stats['metrics']['best_cer']:.4f}")
        
        print("="*80 + "\n")


def plot_metrics_from_file(metrics_file: str, save_path: Optional[str] = None) -> plt.Figure:
    """Convenience function to plot learning curves from metrics file.
    
    Args:
        metrics_file: Path to metrics JSONL file
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    visualizer = LearningCurveVisualizer(metrics_file)
    return visualizer.plot_learning_curves(save_path=save_path)

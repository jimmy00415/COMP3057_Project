"""Visualization utilities for training and evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import os


sns.set_style("whitegrid")


class TrainingVisualizer:
    """Visualize training progress."""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_losses(self, train_losses: List[float], val_losses: List[float], 
                   save_name: str = "losses.png"):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics(self, metrics_history: Dict[str, List[float]], 
                    save_name: str = "metrics.png"):
        """Plot multiple metrics over time."""
        fig, axes = plt.subplots(1, len(metrics_history), figsize=(15, 5))
        
        if len(metrics_history) == 1:
            axes = [axes]
        
        for idx, (metric_name, values) in enumerate(metrics_history.items()):
            ax = axes[idx]
            epochs = range(1, len(values) + 1)
            
            ax.plot(epochs, values, 'g-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric_name.upper(), fontsize=10)
            ax.set_title(f'{metric_name.upper()} Over Time', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_learning_rate(self, lr_history: List[float], 
                          save_name: str = "learning_rate.png"):
        """Plot learning rate schedule."""
        plt.figure(figsize=(10, 6))
        
        steps = range(1, len(lr_history) + 1)
        plt.plot(steps, lr_history, 'purple', linewidth=2)
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path


class EvaluationVisualizer:
    """Visualize evaluation results."""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_model_comparison(self, comparison_data: Dict[str, Dict], 
                             save_name: str = "model_comparison.png"):
        """Plot comparison of different models."""
        models = list(comparison_data.keys())
        wer_scores = [data.get('wer_clean', 0) for data in comparison_data.values()]
        latencies = [data.get('latency', 0) for data in comparison_data.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # WER comparison
        ax1.bar(models, wer_scores, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('WER', fontsize=12)
        ax1.set_title('Word Error Rate Comparison', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Latency comparison
        ax2.bar(models, latencies, color='coral', alpha=0.8)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Latency (s)', fontsize=12)
        ax2.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_latency_distribution(self, latencies: List[float], 
                                  save_name: str = "latency_distribution.png"):
        """Plot distribution of inference latencies."""
        plt.figure(figsize=(10, 6))
        
        plt.hist(latencies, bins=30, color='teal', alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(latencies), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(latencies):.3f}s')
        
        plt.xlabel('Latency (s)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Inference Latency Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_accuracy_vs_speed(self, model_data: Dict[str, Dict],
                              save_name: str = "accuracy_vs_speed.png"):
        """Plot accuracy vs speed trade-off."""
        plt.figure(figsize=(10, 8))
        
        for model_name, data in model_data.items():
            wer = data.get('wer_clean', 0)
            rtf = data.get('rtf', 0)
            
            plt.scatter(rtf, wer, s=200, alpha=0.6, label=model_name)
            plt.annotate(model_name, (rtf, wer), fontsize=10, 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Real-Time Factor (RTF)', fontsize=12)
        plt.ylabel('Word Error Rate (WER)', fontsize=12)
        plt.title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add ideal region
        plt.axhline(y=0.1, color='green', linestyle='--', alpha=0.3, label='Target WER')
        plt.axvline(x=0.5, color='blue', linestyle='--', alpha=0.3, label='Real-time threshold')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

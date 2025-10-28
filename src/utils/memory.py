"""Memory management and optimization utilities."""

import torch
import gc
import logging
from typing import Optional
import psutil
import os


logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage memory usage and optimization."""
    
    def __init__(self, threshold_gb: float = 0.9):
        """Initialize memory manager.
        
        Args:
            threshold_gb: Memory usage threshold (0-1) to trigger cleanup
        """
        self.threshold = threshold_gb
        self.process = psutil.Process(os.getpid())
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        info = {
            'ram_used_gb': self.process.memory_info().rss / 1e9,
            'ram_percent': self.process.memory_percent()
        }
        
        if torch.cuda.is_available():
            info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info['gpu_free_gb'] = info['gpu_total_gb'] - info['gpu_reserved_gb']
        
        return info
    
    def log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        info = self.get_memory_info()
        msg = f"{prefix}RAM: {info['ram_used_gb']:.2f}GB ({info['ram_percent']:.1f}%)"
        
        if 'gpu_allocated_gb' in info:
            msg += f", GPU: {info['gpu_allocated_gb']:.2f}/{info['gpu_total_gb']:.2f}GB"
        
        logger.info(msg)
    
    def cleanup(self, aggressive: bool = False):
        """Clean up memory.
        
        Args:
            aggressive: If True, perform more aggressive cleanup
        """
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if aggressive:
                # Force synchronization and clear all caches
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        logger.debug("Memory cleanup completed")
    
    def check_and_cleanup(self) -> bool:
        """Check memory usage and cleanup if needed.
        
        Returns:
            True if cleanup was performed
        """
        info = self.get_memory_info()
        
        # Check RAM usage
        if info['ram_percent'] > self.threshold * 100:
            logger.warning(f"High RAM usage: {info['ram_percent']:.1f}%, performing cleanup")
            self.cleanup(aggressive=True)
            return True
        
        # Check GPU usage
        if 'gpu_free_gb' in info and info['gpu_free_gb'] < 2.0:
            logger.warning(f"Low GPU memory: {info['gpu_free_gb']:.2f}GB free, performing cleanup")
            self.cleanup(aggressive=True)
            return True
        
        return False
    
    def optimize_for_inference(self):
        """Optimize memory for inference."""
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            
            logger.info("Inference optimizations enabled")
    
    def get_optimal_batch_size(self, 
                               model: torch.nn.Module,
                               input_shape: tuple,
                               max_batch_size: int = 32,
                               safety_factor: float = 0.8) -> int:
        """Estimate optimal batch size based on available memory.
        
        Args:
            model: PyTorch model
            input_shape: Shape of single input (excluding batch dimension)
            max_batch_size: Maximum batch size to try
            safety_factor: Safety margin (0-1)
        
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 1
        
        model.eval()
        device = next(model.parameters()).device
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 1
        
        with torch.no_grad():
            while low <= high:
                mid = (low + high) // 2
                
                try:
                    # Try forward pass
                    batch_shape = (mid,) + input_shape
                    dummy_input = torch.randn(batch_shape, device=device)
                    _ = model(dummy_input)
                    
                    # Check memory usage
                    info = self.get_memory_info()
                    if info['gpu_free_gb'] > 1.0:  # Still have headroom
                        optimal = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        high = mid - 1
                        torch.cuda.empty_cache()
                    else:
                        raise
                
                # Cleanup
                del dummy_input
                torch.cuda.empty_cache()
        
        # Apply safety factor
        optimal = max(1, int(optimal * safety_factor))
        logger.info(f"Estimated optimal batch size: {optimal}")
        
        return optimal


def log_tensor_memory(prefix: str = ""):
    """Log memory usage of all tensors."""
    if not torch.cuda.is_available():
        return
    
    total_memory = 0
    tensor_count = 0
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    total_memory += obj.element_size() * obj.nelement()
                    tensor_count += 1
        except:
            pass
    
    logger.debug(f"{prefix}Total GPU tensors: {tensor_count}, Memory: {total_memory / 1e9:.2f}GB")


def detect_memory_leak(threshold_mb: float = 100.0) -> bool:
    """Detect potential memory leaks.
    
    Args:
        threshold_mb: Threshold in MB for memory increase
    
    Returns:
        True if potential leak detected
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        before = torch.cuda.memory_allocated()
        
        # Force garbage collection
        gc.collect()
        
        after = torch.cuda.memory_allocated()
        diff_mb = (after - before) / 1e6
        
        if diff_mb > threshold_mb:
            logger.warning(f"Potential memory leak detected: {diff_mb:.2f}MB not freed")
            return True
    
    return False

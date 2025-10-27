"""Evaluation metrics and benchmarking."""

import torch
from jiwer import wer, cer
import time
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate WER, CER, and other ASR metrics."""
    
    def __init__(self):
        pass
    
    def compute_wer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Word Error Rate."""
        # Filter empty strings
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        
        if not valid_pairs:
            return 0.0
        
        preds, refs = zip(*valid_pairs)
        return wer(refs, preds)
    
    def compute_cer(self, predictions: List[str], references: List[str]) -> float:
        """Compute Character Error Rate."""
        valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        
        if not valid_pairs:
            return 0.0
        
        preds, refs = zip(*valid_pairs)
        return cer(refs, preds)
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'wer': self.compute_wer(predictions, references),
            'cer': self.compute_cer(predictions, references)
        }


class LatencyBenchmark:
    """Benchmark inference latency."""
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.eval()
    
    def benchmark_single(self, audio_tensor: torch.Tensor, sr: int = 16000) -> Tuple[str, float]:
        """Benchmark single audio clip."""
        # Prepare input
        input_features = self.processor(
            audio_tensor.numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate(input_features)
        
        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription, elapsed_time
    
    def benchmark_batch(self, audio_tensors: List[torch.Tensor], 
                       sr: int = 16000) -> Dict[str, float]:
        """Benchmark multiple audio clips."""
        latencies = []
        audio_lengths = []
        
        for audio in tqdm(audio_tensors, desc="Benchmarking"):
            _, latency = self.benchmark_single(audio, sr)
            latencies.append(latency)
            audio_lengths.append(len(audio) / sr)
        
        latencies = np.array(latencies)
        audio_lengths = np.array(audio_lengths)
        
        # Calculate real-time factor (RTF)
        rtf = latencies / audio_lengths
        
        return {
            'mean_latency': float(np.mean(latencies)),
            'std_latency': float(np.std(latencies)),
            'min_latency': float(np.min(latencies)),
            'max_latency': float(np.max(latencies)),
            'mean_rtf': float(np.mean(rtf)),
            'std_rtf': float(np.std(rtf))
        }


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.metrics_calc = MetricsCalculator()
        self.latency_bench = LatencyBenchmark(model, processor, device)
        self.model.eval()
    
    def evaluate_dataset(self, dataloader) -> Dict[str, float]:
        """Evaluate model on dataset."""
        all_predictions = []
        all_references = []
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"]
                
                # Generate predictions
                predicted_ids = self.model.generate(input_features)
                
                # Decode predictions
                predictions = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )
                
                # Decode references (handle -100 padding)
                labels = labels.cpu().numpy()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Compute metrics
        metrics = self.metrics_calc.compute_all_metrics(all_predictions, all_references)
        
        logger.info(f"WER: {metrics['wer']:.3f}")
        logger.info(f"CER: {metrics['cer']:.3f}")
        
        return metrics
    
    def evaluate_with_samples(self, dataloader, num_samples: int = 5) -> Dict:
        """Evaluate and return sample predictions."""
        metrics = self.evaluate_dataset(dataloader)
        
        # Get sample predictions
        samples = []
        count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if count >= num_samples:
                    break
                
                input_features = batch["input_features"][:1].to(self.device)
                labels = batch["labels"][:1]
                
                predicted_ids = self.model.generate(input_features)
                prediction = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                labels = labels.cpu().numpy()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                reference = self.processor.batch_decode(labels, skip_special_tokens=True)[0]
                
                samples.append({
                    'prediction': prediction,
                    'reference': reference
                })
                
                count += 1
        
        return {
            'metrics': metrics,
            'samples': samples
        }


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table comparing models."""
    header = "| Model | Parameters | WER (clean) | WER (accented) | Latency (s) | RTF |\n"
    header += "|-------|-----------|-------------|----------------|-------------|-----|\n"
    
    table = header
    
    for model_name, data in results.items():
        row = f"| {model_name} | {data.get('params', 'N/A')} | "
        row += f"{data.get('wer_clean', 0):.3f} | "
        row += f"{data.get('wer_accented', 0):.3f} | "
        row += f"{data.get('latency', 0):.3f} | "
        row += f"{data.get('rtf', 0):.3f} |\n"
        table += row
    
    return table

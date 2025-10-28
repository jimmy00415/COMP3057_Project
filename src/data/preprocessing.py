"""Audio preprocessing utilities."""

import torchaudio
import torch
import numpy as np
from typing import Tuple, Optional


class AudioPreprocessor:
    """Handles audio loading, resampling, and normalization."""
    
    def __init__(self, target_sr: int = 16000, normalize: bool = True):
        self.target_sr = target_sr
        self.normalize = normalize
    
    def load_and_preprocess(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and preprocess to target format."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize loudness
        if self.normalize:
            waveform = self._normalize_loudness(waveform)
        
        return waveform.squeeze(0), self.target_sr
    
    def _normalize_loudness(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio loudness to -20 dBFS."""
        # Peak normalization
        max_val = torch.abs(waveform).max()
        if max_val > 1e-8:  # Prevent division by near-zero
            waveform = waveform / max_val * 0.95
        return waveform
    
    def trim_silence(self, waveform: torch.Tensor, sr: int, 
                     threshold: float = 0.01) -> torch.Tensor:
        """Trim leading and trailing silence."""
        # Simple energy-based trimming
        energy = waveform.abs()
        mask = energy > threshold
        
        if mask.any():
            start = mask.nonzero()[0].item()
            end = mask.nonzero()[-1].item()
            return waveform[start:end+1]
        
        return waveform
    
    def segment_audio(self, waveform: torch.Tensor, sr: int, 
                     segment_length_sec: float = 30.0,
                     overlap_sec: float = 0.0) -> list:
        """Segment long audio into chunks."""
        segment_samples = int(segment_length_sec * sr)
        overlap_samples = int(overlap_sec * sr)
        stride = segment_samples - overlap_samples
        
        segments = []
        for start in range(0, len(waveform), stride):
            end = min(start + segment_samples, len(waveform))
            segment = waveform[start:end]
            
            # Pad last segment if needed
            if len(segment) < segment_samples:
                padding = segment_samples - len(segment)
                segment = torch.nn.functional.pad(segment, (0, padding))
            
            segments.append(segment)
            
            if end >= len(waveform):
                break
        
        return segments


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD."""
    
    def __init__(self, threshold: float = 0.5, min_silence_duration_ms: int = 300,
                 speech_pad_ms: int = 30):
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD model."""
        try:
            import torch
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
        except Exception as e:
            print(f"Warning: Could not load Silero VAD: {e}")
            self.model = None
    
    def detect_speech(self, waveform: torch.Tensor, sr: int = 16000) -> list:
        """Detect speech segments in audio."""
        if self.model is None:
            # Fallback: return full audio as one segment
            return [{'start': 0, 'end': len(waveform)}]
        
        # Ensure correct sample rate
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            waveform,
            self.model,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            sampling_rate=16000
        )
        
        return speech_timestamps
    
    def extract_speech_segments(self, waveform: torch.Tensor, 
                               sr: int = 16000) -> list:
        """Extract only speech segments from audio."""
        timestamps = self.detect_speech(waveform, sr)
        segments = []
        
        for ts in timestamps:
            start = ts['start']
            end = ts['end']
            segments.append(waveform[start:end])
        
        return segments

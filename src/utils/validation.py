"""Data validation and quality checking utilities."""

import numpy as np
import torch
import logging
from typing import Tuple, Optional, Dict
import librosa


logger = logging.getLogger(__name__)


class AudioValidator:
    """Validate audio data quality."""
    
    def __init__(self, 
                 min_duration_sec: float = 0.1,
                 max_duration_sec: float = 30.0,
                 expected_sr: int = 16000,
                 silence_threshold: float = 0.01,
                 max_silence_ratio: float = 0.9):
        """Initialize audio validator.
        
        Args:
            min_duration_sec: Minimum acceptable audio duration
            max_duration_sec: Maximum acceptable audio duration
            expected_sr: Expected sample rate
            silence_threshold: Threshold for silence detection (RMS)
            max_silence_ratio: Maximum acceptable ratio of silence
        """
        self.min_duration = min_duration_sec
        self.max_duration = max_duration_sec
        self.expected_sr = expected_sr
        self.silence_threshold = silence_threshold
        self.max_silence_ratio = max_silence_ratio
    
    def validate(self, 
                 audio: np.ndarray, 
                 sr: int,
                 text: Optional[str] = None) -> Dict[str, any]:
        """Validate audio data.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            text: Optional transcription text
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check if audio is None or empty
        if audio is None or len(audio) == 0:
            results['valid'] = False
            results['issues'].append("Empty audio")
            return results
        
        # Check for NaN or Inf
        if np.isnan(audio).any():
            results['valid'] = False
            results['issues'].append("Audio contains NaN values")
        
        if np.isinf(audio).any():
            results['valid'] = False
            results['issues'].append("Audio contains Inf values")
        
        # Check duration
        duration_sec = len(audio) / sr
        if duration_sec < self.min_duration:
            results['valid'] = False
            results['issues'].append(f"Audio too short: {duration_sec:.2f}s < {self.min_duration}s")
        elif duration_sec > self.max_duration:
            results['warnings'].append(f"Audio very long: {duration_sec:.2f}s > {self.max_duration}s")
        
        # Check sample rate
        if sr != self.expected_sr:
            results['warnings'].append(f"Sample rate mismatch: {sr} != {self.expected_sr}")
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
        if clipping_ratio > 0.01:
            results['warnings'].append(f"Audio clipping detected: {clipping_ratio*100:.1f}%")
        
        # Check for silence
        try:
            silence_ratio = self._detect_silence_ratio(audio)
            if silence_ratio > self.max_silence_ratio:
                results['valid'] = False
                results['issues'].append(f"Too much silence: {silence_ratio*100:.1f}%")
            elif silence_ratio > 0.5:
                results['warnings'].append(f"High silence ratio: {silence_ratio*100:.1f}%")
        except Exception as e:
            logger.debug(f"Silence detection failed: {e}")
        
        # Check signal-to-noise ratio
        try:
            snr = self._estimate_snr(audio)
            if snr < 10:
                results['warnings'].append(f"Low SNR: {snr:.1f} dB")
        except Exception as e:
            logger.debug(f"SNR estimation failed: {e}")
        
        # Validate transcription if provided
        if text is not None:
            if not text or not text.strip():
                results['valid'] = False
                results['issues'].append("Empty transcription text")
            elif len(text) > 500:
                results['warnings'].append(f"Very long transcription: {len(text)} characters")
        
        return results
    
    def _detect_silence_ratio(self, audio: np.ndarray) -> float:
        """Detect ratio of silence in audio.
        
        Args:
            audio: Audio waveform
        
        Returns:
            Ratio of silence (0-1)
        """
        # Calculate RMS energy
        frame_length = 2048
        hop_length = 512
        
        # Pad audio if too short
        if len(audio) < frame_length:
            audio = np.pad(audio, (0, frame_length - len(audio)))
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Count silent frames
        silent_frames = np.sum(rms < self.silence_threshold)
        silence_ratio = silent_frames / len(rms)
        
        return silence_ratio
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio.
        
        Args:
            audio: Audio waveform
        
        Returns:
            SNR in dB
        """
        # Simple SNR estimation
        # Assume first and last 10% are noise
        noise_samples = int(len(audio) * 0.1)
        
        if noise_samples < 100:
            # Too short for reliable SNR
            return 20.0
        
        noise = np.concatenate([audio[:noise_samples], audio[-noise_samples:]])
        noise_power = np.mean(noise ** 2)
        
        signal_power = np.mean(audio ** 2)
        
        if noise_power == 0:
            return 100.0  # Perfect signal
        
        snr = 10 * np.log10(signal_power / noise_power)
        return max(0, snr)  # Clip to non-negative
    
    def check_corruption(self, audio: np.ndarray) -> bool:
        """Check if audio might be corrupted.
        
        Args:
            audio: Audio waveform
        
        Returns:
            True if corruption detected
        """
        if audio is None or len(audio) == 0:
            return True
        
        # Check for repeated patterns (could indicate corruption)
        if len(audio) > 1000:
            chunk = audio[:1000]
            if np.std(chunk) < 1e-6:
                return True  # Constant values
        
        # Check for extreme values
        if np.max(np.abs(audio)) > 100:
            return True
        
        # Check for sudden jumps
        if len(audio) > 1:
            diffs = np.abs(np.diff(audio))
            if np.max(diffs) > 10:
                return True
        
        return False


def validate_dataset_batch(audio_list: list, 
                           sr_list: list, 
                           text_list: list = None) -> Tuple[list, list]:
    """Validate a batch of audio samples.
    
    Args:
        audio_list: List of audio waveforms
        sr_list: List of sample rates
        text_list: Optional list of transcriptions
    
    Returns:
        Tuple of (valid_indices, validation_results)
    """
    validator = AudioValidator()
    valid_indices = []
    results = []
    
    for i, (audio, sr) in enumerate(zip(audio_list, sr_list)):
        text = text_list[i] if text_list else None
        result = validator.validate(audio, sr, text)
        results.append(result)
        
        if result['valid']:
            valid_indices.append(i)
        else:
            logger.warning(f"Sample {i} invalid: {result['issues']}")
    
    logger.info(f"Validated {len(audio_list)} samples, {len(valid_indices)} valid")
    
    return valid_indices, results

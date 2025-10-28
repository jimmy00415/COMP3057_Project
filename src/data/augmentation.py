"""Audio augmentation for training robustness."""

import torch
import torchaudio
import random
import numpy as np
from typing import Optional


class AudioAugmenter:
    """Applies various augmentations to audio data."""
    
    def __init__(self, 
                 speed_perturbation: list = [0.9, 1.0, 1.1],
                 pitch_shift_semitones: list = [-2, 0, 2],
                 background_noise_prob: float = 0.3,
                 noise_snr_db: tuple = (10, 20)):
        self.speed_factors = speed_perturbation
        self.pitch_semitones = pitch_shift_semitones
        self.noise_prob = background_noise_prob
        self.noise_snr_db = noise_snr_db
    
    def apply_speed_perturbation(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply random speed perturbation."""
        speed_factor = random.choice(self.speed_factors)
        
        if speed_factor == 1.0:
            return waveform
        
        try:
            # Resample to change speed
            new_sr = int(sr * speed_factor)
            
            # Prevent extreme speeds
            if new_sr < 8000 or new_sr > 32000:
                return waveform
            
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            perturbed = resampler(waveform.unsqueeze(0))
            
            # Resample back to original rate
            resampler_back = torchaudio.transforms.Resample(new_sr, sr)
            perturbed = resampler_back(perturbed)
            
            result = perturbed.squeeze(0)
            
            # Check for NaN/Inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                return waveform
            
            return result
        except Exception:
            return waveform
    
    def apply_pitch_shift(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply random pitch shift."""
        semitones = random.choice(self.pitch_semitones)
        
        if semitones == 0:
            return waveform
        
        try:
            # Pitch shift using resampling
            n_steps = semitones
            rate = 2 ** (n_steps / 12)
            
            new_sr = int(sr * rate)
            
            # Prevent extreme resampling rates
            if new_sr < 8000 or new_sr > 32000:
                return waveform
            
            resampler = torchaudio.transforms.Resample(sr, new_sr)
            shifted = resampler(waveform.unsqueeze(0))
            
            # Resample back without changing speed
            resampler_back = torchaudio.transforms.Resample(new_sr, sr)
            shifted = resampler_back(shifted)
            
            result = shifted.squeeze(0)
            
            # Check for NaN/Inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                return waveform
            
            return result
        except Exception:
            return waveform
    
    def add_background_noise(self, waveform: torch.Tensor, 
                            noise_waveform: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add background noise at random SNR."""
        if random.random() > self.noise_prob:
            return waveform
        
        # Generate white noise if no noise provided
        if noise_waveform is None:
            noise_waveform = torch.randn_like(waveform) * 0.1  # Limit initial noise amplitude
        else:
            # Ensure noise matches signal length
            if len(noise_waveform) < len(waveform):
                # Repeat noise
                repeats = (len(waveform) // len(noise_waveform)) + 1
                noise_waveform = noise_waveform.repeat(repeats)
            
            # Random crop noise to match signal
            if len(noise_waveform) > len(waveform):
                start = random.randint(0, len(noise_waveform) - len(waveform))
                noise_waveform = noise_waveform[start:start + len(waveform)]
        
        # Calculate SNR
        snr_db = random.uniform(*self.noise_snr_db)
        
        # Calculate signal and noise power
        signal_power = waveform.norm(p=2)
        noise_power = noise_waveform.norm(p=2)
        
        # Prevent division by zero
        if signal_power < 1e-8 or noise_power < 1e-8:
            return waveform
        
        # Scale noise to achieve target SNR
        snr = 10 ** (snr_db / 10)
        scale = signal_power / (noise_power * np.sqrt(snr))
        
        # Clamp scale to prevent extreme values
        scale = min(scale, 10.0)
        
        noisy_waveform = waveform + scale * noise_waveform
        
        # Check for NaN/Inf
        if torch.isnan(noisy_waveform).any() or torch.isinf(noisy_waveform).any():
            return waveform
        
        # Normalize to prevent clipping
        max_val = noisy_waveform.abs().max()
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val * 0.95
        
        return noisy_waveform
    
    def apply_augmentations(self, waveform: torch.Tensor, sr: int,
                           apply_speed: bool = True,
                           apply_pitch: bool = True,
                           apply_noise: bool = True) -> torch.Tensor:
        """Apply random combination of augmentations."""
        augmented = waveform.clone()
        
        if apply_speed:
            augmented = self.apply_speed_perturbation(augmented, sr)
        
        if apply_pitch:
            augmented = self.apply_pitch_shift(augmented, sr)
        
        if apply_noise:
            augmented = self.add_background_noise(augmented)
        
        return augmented


class SpecAugment:
    """SpecAugment for spectrogram-based augmentation."""
    
    def __init__(self, freq_mask_param: int = 27, time_mask_param: int = 100,
                 n_freq_masks: int = 1, n_time_masks: int = 1):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram."""
        augmented = spectrogram.clone()
        
        for _ in range(self.n_freq_masks):
            augmented = self.freq_mask(augmented)
        
        for _ in range(self.n_time_masks):
            augmented = self.time_mask(augmented)
        
        return augmented

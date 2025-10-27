"""Real-time streaming inference engine."""

import torch
import torchaudio
import numpy as np
from collections import deque
from typing import Optional, Callable, List
import time
import logging


logger = logging.getLogger(__name__)


class AudioBuffer:
    """Circular buffer for streaming audio."""
    
    def __init__(self, max_length_sec: float = 30.0, sr: int = 16000):
        self.max_length = int(max_length_sec * sr)
        self.sr = sr
        self.buffer = deque(maxlen=self.max_length)
    
    def append(self, audio_chunk: np.ndarray):
        """Add audio samples to buffer."""
        self.buffer.extend(audio_chunk.flatten())
    
    def get_audio(self, length_sec: Optional[float] = None) -> np.ndarray:
        """Get audio from buffer."""
        if length_sec is None:
            return np.array(list(self.buffer))
        
        length_samples = int(length_sec * self.sr)
        length_samples = min(length_samples, len(self.buffer))
        
        # Get last N samples
        return np.array(list(self.buffer)[-length_samples:])
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class StreamingASR:
    """Real-time streaming ASR system."""
    
    def __init__(self, 
                 model,
                 processor,
                 vad=None,
                 chunk_length_sec: float = 2.0,
                 overlap_sec: float = 0.2,
                 device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.vad = vad
        self.chunk_length_sec = chunk_length_sec
        self.overlap_sec = overlap_sec
        self.sr = 16000
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Streaming state
        self.buffer = AudioBuffer(max_length_sec=30.0, sr=self.sr)
        self.last_chunk_offset = 0
        self.transcription_history = []
        
        self.model.eval()
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Process a single audio chunk."""
        # Add to buffer
        self.buffer.append(audio_chunk)
        
        # Get audio for transcription
        chunk_samples = int(self.chunk_length_sec * self.sr)
        
        if len(self.buffer) < chunk_samples:
            return None  # Not enough audio yet
        
        # Extract chunk with overlap
        audio = self.buffer.get_audio(self.chunk_length_sec)
        
        # Apply VAD if available
        if self.vad is not None:
            audio_tensor = torch.from_numpy(audio).float()
            speech_timestamps = self.vad.detect_speech(audio_tensor, self.sr)
            
            if not speech_timestamps:
                return None  # No speech detected
        
        # Transcribe
        transcription = self._transcribe(audio)
        
        if transcription:
            self.transcription_history.append(transcription)
        
        return transcription
    
    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio chunk."""
        # Prepare input
        input_features = self.processor(
            audio,
            sampling_rate=self.sr,
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=128,
                num_beams=1,  # Greedy decoding for speed
                temperature=0.0
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return transcription
    
    def stream_from_microphone(self, 
                               duration_sec: Optional[float] = None,
                               callback: Optional[Callable] = None):
        """Stream audio from microphone and transcribe in real-time."""
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice")
            return
        
        chunk_samples = int(self.chunk_length_sec * self.sr)
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Process chunk
            audio_chunk = indata[:, 0] if indata.ndim > 1 else indata
            transcription = self.process_chunk(audio_chunk)
            
            if transcription and callback:
                callback(transcription)
        
        # Start streaming
        logger.info("Starting microphone stream...")
        
        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            callback=audio_callback,
            blocksize=chunk_samples
        ):
            if duration_sec:
                sd.sleep(int(duration_sec * 1000))
            else:
                logger.info("Press Ctrl+C to stop streaming")
                try:
                    while True:
                        sd.sleep(1000)
                except KeyboardInterrupt:
                    logger.info("Streaming stopped")
    
    def stream_from_file(self, audio_path: str, 
                        chunk_duration_sec: float = 0.5,
                        callback: Optional[Callable] = None) -> List[str]:
        """Simulate streaming from audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)
        
        # Process in chunks
        chunk_size = int(chunk_duration_sec * self.sr)
        transcriptions = []
        
        for i in range(0, len(waveform), chunk_size):
            chunk = waveform[i:i + chunk_size].numpy()
            
            # Simulate real-time delay
            time.sleep(chunk_duration_sec * 0.1)  # 10% of real-time for simulation
            
            transcription = self.process_chunk(chunk)
            
            if transcription:
                transcriptions.append(transcription)
                if callback:
                    callback(transcription)
        
        return transcriptions
    
    def get_full_transcription(self, merge: bool = True) -> str:
        """Get complete transcription from history."""
        if merge:
            # Use local agreement to merge overlapping transcriptions
            return self._merge_transcriptions(self.transcription_history)
        else:
            return " ".join(self.transcription_history)
    
    def _merge_transcriptions(self, transcriptions: List[str]) -> str:
        """Merge overlapping transcriptions using simple deduplication."""
        if not transcriptions:
            return ""
        
        # Simple approach: join with space and remove obvious duplicates
        merged = transcriptions[0]
        
        for i in range(1, len(transcriptions)):
            current = transcriptions[i]
            
            # Find overlap by checking if end of merged matches start of current
            overlap_found = False
            for overlap_len in range(min(50, len(merged.split())), 0, -1):
                merged_end = " ".join(merged.split()[-overlap_len:])
                current_start = " ".join(current.split()[:overlap_len])
                
                if merged_end.lower() == current_start.lower():
                    # Found overlap, merge
                    remaining = " ".join(current.split()[overlap_len:])
                    if remaining:
                        merged = merged + " " + remaining
                    overlap_found = True
                    break
            
            if not overlap_found:
                # No overlap, just append
                merged = merged + " " + current
        
        return merged.strip()
    
    def reset(self):
        """Reset streaming state."""
        self.buffer.clear()
        self.transcription_history.clear()
        self.last_chunk_offset = 0


class BatchInference:
    """Optimized batch inference for multiple files."""
    
    def __init__(self, model, processor, device: str = "cuda", batch_size: int = 8):
        self.model = model
        self.processor = processor
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model.eval()
    
    def transcribe_batch(self, audio_paths: List[str]) -> List[str]:
        """Transcribe multiple audio files in batches."""
        transcriptions = []
        
        for i in range(0, len(audio_paths), self.batch_size):
            batch_paths = audio_paths[i:i + self.batch_size]
            batch_transcriptions = self._process_batch(batch_paths)
            transcriptions.extend(batch_transcriptions)
        
        return transcriptions
    
    def _process_batch(self, audio_paths: List[str]) -> List[str]:
        """Process a batch of audio files."""
        # Load and preprocess audio
        audio_arrays = []
        
        for path in audio_paths:
            waveform, sr = torchaudio.load(path)
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            else:
                waveform = waveform.squeeze(0)
            
            audio_arrays.append(waveform.numpy())
        
        # Prepare batch - pad to Whisper's expected mel feature length
        input_features = self.processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest"
        ).input_features.to(self.device)
        
        # Generate transcriptions
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        
        # Decode
        transcriptions = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )
        
        return transcriptions

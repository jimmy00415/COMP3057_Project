"""Dataset classes for Whisper training."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional
import logging

from src.utils.validation import AudioValidator


logger = logging.getLogger(__name__)


class WhisperDataset(Dataset):
    """PyTorch Dataset for Whisper fine-tuning."""
    
    def __init__(self, 
                 dataset,
                 processor,
                 audio_column: str = "audio",
                 text_column: str = "sentence",
                 max_audio_length_sec: float = 30.0,
                 augmenter=None,
                 validate_audio: bool = True):
        self.dataset = dataset
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_audio_length = int(max_audio_length_sec * 16000)
        self.augmenter = augmenter
        self.validate_audio = validate_audio
        
        # Initialize audio validator
        if validate_audio:
            self.validator = AudioValidator(
                min_duration_sec=0.1,
                max_duration_sec=max_audio_length_sec,
                expected_sr=16000
            )
        else:
            self.validator = None
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
        except Exception as e:
            logger.error(f"Failed to load dataset item {idx}: {e}")
            raise RuntimeError(f"Dataset access error at index {idx}") from e
        
        # Extract audio - handles both decoded and path-based audio
        audio = item.get(self.audio_column)
        if audio is None:
            raise ValueError(f"Audio column '{self.audio_column}' not found in dataset item {idx}")
        
        waveform = None
        sr = 16000
        
        try:
            if isinstance(audio, dict):
                if "array" in audio:
                    # Already decoded - could be numpy array or Python list
                    array_data = audio["array"]
                    if array_data is None or (hasattr(array_data, '__len__') and len(array_data) == 0):
                        logger.warning(f"Empty audio array at index {idx}, using silence")
                        waveform = torch.zeros(16000, dtype=torch.float32)  # 1 second of silence
                    elif isinstance(array_data, list):
                        waveform = torch.tensor(array_data, dtype=torch.float32)
                    else:
                        waveform = torch.tensor(array_data, dtype=torch.float32)
                    sr = audio.get("sampling_rate", 16000)
                elif "path" in audio and audio["path"]:
                    # Load from path
                    import soundfile as sf
                    try:
                        waveform, sr = sf.read(audio["path"])
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                    except Exception as e:
                        logger.error(f"Failed to load audio from {audio['path']}: {e}")
                        waveform = torch.zeros(16000, dtype=torch.float32)
                elif "bytes" in audio and audio["bytes"]:
                    # Load from bytes
                    import soundfile as sf
                    import io
                    try:
                        waveform, sr = sf.read(io.BytesIO(audio["bytes"]))
                        waveform = torch.tensor(waveform, dtype=torch.float32)
                    except Exception as e:
                        logger.error(f"Failed to load audio from bytes at index {idx}: {e}")
                        waveform = torch.zeros(16000, dtype=torch.float32)
                else:
                    raise ValueError(f"Audio dict at index {idx} missing valid data. Keys: {audio.keys()}")
            else:
                waveform = torch.tensor(audio, dtype=torch.float32)
                sr = 16000
                
        except Exception as e:
            logger.error(f"Audio loading error at index {idx}: {e}")
            # Return silence as fallback
            waveform = torch.zeros(16000, dtype=torch.float32)
            sr = 16000
        
        # Validate waveform
        if waveform is None:
            raise RuntimeError(f"Failed to load audio at index {idx}")
        
        # Validate audio quality if enabled
        if self.validator:
            text = item.get(self.text_column, "")
            validation_result = self.validator.validate(
                waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform,
                sr,
                text
            )
            
            if not validation_result['valid']:
                logger.warning(f"Audio validation failed at index {idx}: {validation_result['issues']}")
                # Continue with fallback instead of failing
            
            if validation_result['warnings']:
                logger.debug(f"Audio warnings at index {idx}: {validation_result['warnings']}")
            
        # Check for NaN or Inf
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            logger.warning(f"NaN/Inf detected in audio at index {idx}, replacing with zeros")
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=-1)
        
        # Ensure 1D tensor
        if len(waveform.shape) == 0:
            logger.warning(f"Scalar audio at index {idx}, converting to 1D")
            waveform = waveform.unsqueeze(0)
        
        # Check if audio is too short (< 0.1 seconds)
        if len(waveform) < 1600:  # 0.1 seconds at 16kHz
            logger.warning(f"Very short audio at index {idx} ({len(waveform)} samples), padding")
            waveform = torch.nn.functional.pad(waveform, (0, 1600 - len(waveform)))
        
        # Apply augmentation during training
        if self.augmenter is not None:
            try:
                waveform = self.augmenter.apply_augmentations(waveform, sr)
            except Exception as e:
                logger.error(f"Augmentation failed at index {idx}: {e}, using original audio")
        
        # Truncate if too long
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Process audio to features
        try:
            input_features = self.processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.squeeze(0)
            
            # Validate features
            if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                logger.error(f"NaN/Inf in features at index {idx}, using zeros")
                input_features = torch.zeros_like(input_features)
                
        except Exception as e:
            logger.error(f"Feature extraction failed at index {idx}: {e}")
            raise RuntimeError(f"Processor failed for audio at index {idx}") from e
        
        # Process text to labels
        text = item.get(self.text_column, "")
        if not text or not text.strip():
            logger.warning(f"Empty text at index {idx}, using placeholder")
            text = "<empty>"
        
        try:
            labels = self.processor.tokenizer(text).input_ids
        except Exception as e:
            logger.error(f"Tokenization failed at index {idx} for text: '{text}': {e}")
            raise RuntimeError(f"Tokenizer failed at index {idx}") from e
        
        return {
            "input_features": input_features,
            "labels": labels
        }


class DataCollatorWithPadding:
    """Custom data collator for Whisper batching."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract features and labels
        input_features = [f["input_features"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Stack input features (already same size from mel spectrogram)
        batch_input_features = torch.stack(input_features)
        
        # Pad labels to same length
        max_label_length = max(len(l) for l in labels)
        padded_labels = []
        
        for label in labels:
            padding_length = max_label_length - len(label)
            padded_label = label + [-100] * padding_length  # -100 is ignore index
            padded_labels.append(padded_label)
        
        batch_labels = torch.tensor(padded_labels)
        
        return {
            "input_features": batch_input_features,
            "labels": batch_labels
        }


def prepare_datasets(config: Dict, processor, augmenter=None):
    """Load and prepare datasets based on config."""
    datasets_list = []
    
    for ds_config in config['data']['datasets']:
        dataset_name = ds_config['name']
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            if dataset_name == "common_voice":
                ds = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    ds_config.get('language', 'en'),
                    split=ds_config.get('split', 'train'),
                    trust_remote_code=True
                )
                # Rename columns if needed
                if 'sentence' in ds.column_names:
                    pass  # Already correct
                else:
                    ds = ds.rename_column("text", "sentence")
                    
            elif dataset_name == "peoples_speech":
                # Placeholder - would need actual implementation
                logger.warning(f"Dataset {dataset_name} not fully implemented, skipping")
                continue
                
            elif dataset_name == "speechocean762":
                ds = load_dataset(
                    "mispeech/speechocean762",
                    split=ds_config.get('split', 'train'),
                    trust_remote_code=True
                )
                # Adapt column names
                if 'text' in ds.column_names:
                    ds = ds.rename_column("text", "sentence")
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            datasets_list.append(ds)
            logger.info(f"Loaded {len(ds)} samples from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue
    
    # Concatenate all datasets
    if datasets_list:
        combined_dataset = concatenate_datasets(datasets_list)
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
    else:
        logger.warning("No datasets loaded!")
        combined_dataset = None
    
    return combined_dataset


def create_dataloaders(train_dataset, val_dataset, processor, config: Dict):
    """Create training and validation dataloaders."""
    collator = DataCollatorWithPadding(processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Set to 0 for Colab compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

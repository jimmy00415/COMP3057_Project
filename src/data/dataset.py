"""Dataset classes for Whisper training."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


class WhisperDataset(Dataset):
    """PyTorch Dataset for Whisper fine-tuning."""
    
    def __init__(self, 
                 dataset,
                 processor,
                 audio_column: str = "audio",
                 text_column: str = "sentence",
                 max_audio_length_sec: float = 30.0,
                 augmenter=None):
        self.dataset = dataset
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.max_audio_length = int(max_audio_length_sec * 16000)
        self.augmenter = augmenter
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract audio - handles both decoded and path-based audio
        audio = item[self.audio_column]
        
        if isinstance(audio, dict):
            if "array" in audio:
                # Already decoded - could be numpy array or Python list
                array_data = audio["array"]
                if isinstance(array_data, list):
                    # Convert from list to tensor
                    waveform = torch.tensor(array_data, dtype=torch.float32)
                else:
                    # Already numpy array or similar
                    waveform = torch.tensor(array_data, dtype=torch.float32)
                sr = audio.get("sampling_rate", 16000)
            elif "path" in audio:
                # Load from path
                import soundfile as sf
                waveform, sr = sf.read(audio["path"])
                waveform = torch.tensor(waveform, dtype=torch.float32)
            elif "bytes" in audio:
                # Load from bytes
                import soundfile as sf
                import io
                waveform, sr = sf.read(io.BytesIO(audio["bytes"]))
                waveform = torch.tensor(waveform, dtype=torch.float32)
            else:
                raise ValueError(f"Audio format not supported: {audio.keys()}")
        else:
            waveform = torch.tensor(audio, dtype=torch.float32)
            sr = 16000
        
        # Ensure mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=-1)
        
        # Apply augmentation during training
        if self.augmenter is not None:
            waveform = self.augmenter.apply_augmentations(waveform, sr)
        
        # Truncate if too long
        if len(waveform) > self.max_audio_length:
            waveform = waveform[:self.max_audio_length]
        
        # Process audio to features
        input_features = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # Process text to labels
        text = item[self.text_column]
        labels = self.processor.tokenizer(text).input_ids
        
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

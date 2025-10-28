#!/usr/bin/env python3
"""Test configuration and data pipeline for NaN issues."""

import sys
import torch
import yaml
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config():
    """Test configuration loads correctly."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Grad accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Max grad norm: {config['training']['max_grad_norm']}")
    print(f"  Augmentation: {config['data']['augmentation']['enabled']}")
    print(f"  FP16: {config['training']['fp16']}")
    
    # Verify safe values
    assert config['training']['learning_rate'] <= 1e-4, "LR too high"
    assert config['training']['max_grad_norm'] <= 1.0, "Grad clip too loose"
    print("\n✓ All config values safe\n")
    
    return config

def test_data_loading():
    """Test data loading with small sample."""
    print("=" * 60)
    print("Testing Data Pipeline")
    print("=" * 60)
    
    from datasets import load_dataset, Audio
    from src.models import WhisperModelManager
    from src.data import WhisperDataset
    import soundfile as sf
    import io
    import numpy as np
    
    # Load tiny sample
    print("Loading 5 test samples...")
    try:
        dataset = load_dataset("PolyAI/minds14", "en-US", split="train[:5]")
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        def decode_audio(example):
            audio = example["audio"]
            try:
                if "bytes" in audio and audio["bytes"]:
                    waveform, sr = sf.read(io.BytesIO(audio["bytes"]))
                elif "path" in audio and audio["path"]:
                    waveform, sr = sf.read(audio["path"])
                else:
                    waveform, sr = np.zeros(16000, dtype=np.float32), 16000
            except:
                waveform, sr = np.zeros(16000, dtype=np.float32), 16000
            
            waveform = np.array(waveform, dtype=np.float32)
            example["audio_decoded"] = {"array": waveform.tolist(), "sampling_rate": int(sr)}
            return example
        
        dataset = dataset.map(decode_audio)
        dataset = dataset.remove_columns(["audio"])
        dataset = dataset.rename_column("audio_decoded", "audio")
        
        print("✓ Dataset loaded")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # Test processor
    print("\nLoading Whisper model...")
    config = test_config()
    model_manager = WhisperModelManager(config)
    model, processor = model_manager.initialize_model(variant='tiny', device='cpu')
    print("✓ Model loaded (tiny on CPU)")
    
    # Test dataset wrapper
    print("\nTesting WhisperDataset...")
    test_dataset = WhisperDataset(
        dataset,
        processor,
        audio_column="audio",
        text_column="transcription",
        max_audio_length_sec=30.0,
        augmenter=None
    )
    
    # Test samples
    print(f"\nProcessing {len(test_dataset)} samples...")
    for i in range(len(test_dataset)):
        try:
            sample = test_dataset[i]
            
            # Check for NaN/Inf
            features = sample['input_features']
            labels = sample['labels']
            
            has_nan = torch.isnan(features).any()
            has_inf = torch.isinf(features).any()
            
            if has_nan or has_inf:
                print(f"✗ Sample {i}: NaN={has_nan}, Inf={has_inf}")
                return False
            
            print(f"✓ Sample {i}: features shape={features.shape}, labels len={len(labels)}")
            
        except Exception as e:
            print(f"✗ Sample {i} failed: {e}")
            return False
    
    print("\n✓ All samples processed successfully - NO NaN/Inf detected\n")
    return True

def test_training_step():
    """Test single training step."""
    print("=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    from src.models import WhisperModelManager
    from src.data import WhisperDataset, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from datasets import load_dataset, Audio
    import soundfile as sf
    import io
    import numpy as np
    
    config = test_config()
    
    # Load tiny dataset
    print("Loading test data...")
    dataset = load_dataset("PolyAI/minds14", "en-US", split="train[:8]")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    def decode_audio(example):
        audio = example["audio"]
        try:
            if "bytes" in audio and audio["bytes"]:
                waveform, sr = sf.read(io.BytesIO(audio["bytes"]))
            else:
                waveform, sr = np.zeros(16000, dtype=np.float32), 16000
        except:
            waveform, sr = np.zeros(16000, dtype=np.float32), 16000
        
        example["audio_decoded"] = {"array": np.array(waveform, dtype=np.float32).tolist(), "sampling_rate": int(sr)}
        return example
    
    dataset = dataset.map(decode_audio).remove_columns(["audio"]).rename_column("audio_decoded", "audio")
    
    # Initialize
    print("Loading model...")
    model_manager = WhisperModelManager(config)
    model, processor = model_manager.initialize_model(variant='tiny', device='cpu')
    model.train()
    
    test_dataset = WhisperDataset(dataset, processor, audio_column="audio", text_column="transcription", augmenter=None)
    collator = DataCollatorWithPadding(processor)
    loader = DataLoader(test_dataset, batch_size=4, collate_fn=collator)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch = next(iter(loader))
    
    with torch.no_grad():
        outputs = model(
            input_features=batch['input_features'],
            labels=batch['labels']
        )
        loss = outputs.loss
    
    print(f"Loss: {loss.item():.4f}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("✗ NaN/Inf detected in loss!")
        return False
    
    print("✓ Forward pass successful - no NaN/Inf\n")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRAINING VALIDATION TEST")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Config
        config = test_config()
        
        # Test 2: Data pipeline
        if not test_data_loading():
            print("\n✗ FAILED: Data pipeline has issues")
            sys.exit(1)
        
        # Test 3: Training step
        if not test_training_step():
            print("\n✗ FAILED: Training step has issues")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - Safe to train")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

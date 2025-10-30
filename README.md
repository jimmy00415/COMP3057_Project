# Real-Time English ASR with Whisper

**Author:** CHEN YIWEI (22256024)  
**Course:** COMP3057 Advanced Topics in AI  
**Institution:** HKBU

Production-ready real-time speech recognition system using OpenAI's Whisper models with complete training pipeline, evaluation metrics, and learning curve visualization.


<img width="2090" height="1578" alt="image" src="https://github.com/user-attachments/assets/521ed570-3288-43e3-8bf7-e06e323589ff" />


## 🎯 Features

- **Multi-variant Whisper Models**: Tiny (39M), Base (74M), Small (244M) variants
- **Complete Training Pipeline**: Fine-tuning with WER/CER tracking during training
- **Learning Curve Visualization**: Automatic plotting of loss, WER, CER progression
- **Real-time Streaming Inference**: Chunked audio processing with VAD
- **Comprehensive Evaluation**: WER, CER, latency benchmarking with multiple methods
- **Colab-Ready**: Optimized for Google Colab (A100/T4 GPU)

## 📁 Project Structure

```
COMP3057_Project/
├── whisper_asr_colab .ipynb   # Main notebook (all-in-one)
├── config.yaml                # Training configuration
├── requirements.txt           # Dependencies
├── src/
│   ├── data/                  # Data processing
│   │   ├── preprocessing.py   # Audio preprocessing & VAD
│   │   ├── augmentation.py    # Audio augmentation
│   │   └── dataset.py         # Dataset wrapper
│   ├── models/
│   │   └── whisper_model.py   # Model initialization
│   ├── training/
│   │   └── trainer.py         # Training loop with WER/CER tracking
│   ├── evaluation/
│   │   ├── metrics.py         # WER/CER computation
│   │   ├── visualization.py   # Performance plots
│   │   └── learning_curves.py # Learning curve visualization
│   ├── inference/
│   │   └── streaming.py       # Real-time streaming
│   └── utils/
│       ├── config.py          # Config & logging
│       ├── memory.py          # Memory management
│       └── versioning.py      # Experiment tracking
└── plots/                     # Generated visualizations
```

## 🚀 Quick Start (Google Colab)

### One-Click Setup
1. Open `whisper_asr_colab .ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4/A100)
3. Run cells sequentially from top to bottom

### Notebook Workflow
The notebook follows this complete pipeline:

1. **Setup** - Install dependencies, mount Google Drive
2. **Configuration** - Load config, setup logging, experiment tracking
3. **Data Preparation** - Load minds14/LibriSpeech dataset (2000 train / 400 val)
4. **Model Initialization** - Load Whisper-Small (244M params)
5. **Training** - Fine-tune for 5 epochs with WER/CER tracking
6. **Learning Curves** - Automatic visualization of training progress
7. **Evaluation** - WER/CER metrics on validation set
8. **Inference Testing** - Batch/streaming/single-pass comparison
9. **Visualization** - 7-panel comprehensive analysis plot

### Training Configuration (Default)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | whisper-small | 244M params, 2GB VRAM |
| Dataset | minds14 | 2000 train / 400 val |
| Epochs | 5 | ~40-50 min on A100 |
| Batch Size | 8 | Effective: 32 (grad accum 4x) |
| Learning Rate | 1e-5 | With 500-step warmup |
| WER/CER Tracking | Every epoch | Adds ~2-3 min/epoch |

## 🔧 Local Setup

```bash
# Clone repository
git clone https://github.com/jimmy00415/COMP3057_Project.git
cd COMP3057_Project

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
jupyter notebook whisper_asr_colab.ipynb
```

## 📊 Model Variants

| Model | Parameters | Memory | Speed | WER (Expected) |
|-------|-----------|--------|-------|----------------|
| tiny | 39M | ~500MB | Fastest | ~10-15% |
| base | 74M | ~1GB | Fast | ~8-12% |
| small | 244M | ~2GB | Moderate | ~3-5% |
| medium | 769M | ~6GB | Slow | ~2-4% |

*WER values on minds14 dataset after 5 epochs fine-tuning*

## 🎓 Key Implementation Details

### Training Pipeline (`src/training/trainer.py`)

```python
# Training with WER/CER tracking
trainer = WhisperTrainer(model, processor, train_loader, val_loader, config)
best_loss = trainer.train(
    num_epochs=5,
    compute_metrics_every=1  # Compute WER/CER every epoch
)
```

**Features:**
- Mixed precision training (FP16)
- Gradient accumulation (4x)
- Early stopping (patience=3)
- Automatic checkpoint saving
- WER/CER computation with text normalization
- Metrics logging to JSONL

### Learning Curve Visualization

Built-in visualization after training shows:
1. **Training Loss** - Raw + smoothed (moving average)
2. **Validation Loss** - Per epoch with best marker
3. **WER & CER** - Error rates over epochs
4. **Learning Rate** - Schedule progression
5. **Train vs Val** - Overfitting detection

```python
# Automatically generated in notebook after training
# Saved to plots/learning_curves.png
```

### Evaluation Methods

Three inference methods compared:
- **Batch Processing** - Standard batch inference
- **Streaming** - Chunked processing (2s chunks, 0.5s overlap)
- **Single-Pass** - Complete audio in one pass

Metrics computed:
- WER (Word Error Rate)
- CER (Character Error Rate)
- Latency (seconds)
- RTF (Real-Time Factor)

## 🔬 Technical Stack

- **Deep Learning**: PyTorch 2.0+, Transformers 4.35+
- **Audio**: torchaudio, librosa, soundfile, silero-vad
- **Datasets**: HuggingFace Datasets (minds14, LibriSpeech)
- **Metrics**: jiwer (WER/CER computation)
- **Visualization**: matplotlib, seaborn, numpy
- **Logging**: Python logging, JSONL metrics files
- **Platform**: Google Colab (T4/A100 GPU)

## 📈 Expected Results

### Training Performance (Whisper-Small, 5 epochs)

```
Training: 2000 samples, ~1250 steps
Validation: 400 samples

Final Metrics:
├─ Validation Loss: ~0.23-0.28
├─ WER: 1-5% (minds14 dataset)
├─ CER: 0.5-2%
└─ Training Time: ~40-50 min (A100 GPU)
```

### Inference Performance

| Method | WER | CER | Latency | RTF |
|--------|-----|-----|---------|-----|
| Batch | 0.03-0.05 | 0.01-0.02 | 0.3-0.5s | 0.10-0.15x |
| Streaming | 0.04-0.08 | 0.02-0.03 | 0.5-0.8s | 0.20-0.30x |
| Single | 0.03-0.05 | 0.01-0.02 | 0.2-0.4s | 0.07-0.12x |

*RTF < 1.0 = Real-time capable*

### Visualization Outputs

1. **Learning Curves** (`plots/learning_curves.png`)
   - 5-panel visualization showing training progression
   - Automatically generated after training

2. **Comprehensive Analysis** (`plots/comprehensive_analysis.png`)
   - 7-panel performance analysis
   - Includes WER/CER distributions, latency, streaming comparison
   - Model architecture diagram

## 🛠️ Configuration (`config.yaml`)

Key parameters used in the project:

```yaml
project:
  seed: 42
  device: cuda

model:
  default_variant: small  # 244M params

training:
  num_epochs: 5
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size: 32
  learning_rate: 1.0e-5
  warmup_steps: 500
  fp16: true
  early_stopping_patience: 3

data:
  sampling_rate: 16000
  audio_max_length_sec: 30
  augmentation:
    enabled: false  # Disabled for training stability

inference:
  streaming:
    chunk_length_sec: 2.0
    overlap_sec: 0.5
```

## � Local Setup (Optional)

```bash
# Clone repository
git clone https://github.com/jimmy00415/COMP3057_Project.git
cd COMP3057_Project

# Install dependencies
pip install -r requirements.txt

# Run notebook locally
jupyter notebook "whisper_asr_colab .ipynb"
```

**Note:** Google Colab is recommended for GPU access.

## � Output Files

After running the notebook, the following files are generated:

### Checkpoints
- `checkpoints/best_model.pt` - Best model checkpoint (PyTorch)
- `checkpoints/best_model_hf/` - HuggingFace format model
- `final_model/` - Exported model for deployment

### Logs
- `logs/training.log` - Training log file
- `logs/metrics_YYYYMMDD_HHMMSS.jsonl` - Metrics (loss, WER, CER) by step/epoch

### Visualizations
- `plots/learning_curves.png` - 5-panel training progression visualization
- `plots/comprehensive_analysis.png` - 7-panel performance analysis

### Google Drive (if mounted)
- Checkpoints and logs are also saved to Google Drive for persistence across sessions

## 📚 References

1. [Whisper Paper](https://arxiv.org/abs/2212.04356) - Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
2. [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Whisper model implementation
3. [minds14 Dataset](https://huggingface.co/datasets/PolyAI/minds14) - E-banking intent classification dataset
4. [LibriSpeech](https://www.openslr.org/12/) - Speech recognition corpus

## 📄 License

MIT License

## 👥 Author

**CHEN YIWEI (22256024)**  
COMP3057 Advanced Topics in AI  
University of Macau

## 🙏 Acknowledgments

- OpenAI for the Whisper model architecture and pre-trained weights
- HuggingFace for the Transformers library and Datasets
- Google Colab for providing free GPU resources
- University of Macau COMP3057 course staff

---

**Project completed for COMP3057 - October 2025** 🎓

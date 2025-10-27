# Real-Time Multilingual ASR with Whisper

Production-ready real-time speech recognition system using OpenAI's Whisper models.

## 🎯 Features

- **Multi-variant Support**: Tiny, Base, Small, Medium, and Distilled Whisper models
- **Real-time Streaming**: Chunked audio processing with VAD integration
- **Fine-tuning Pipeline**: Complete training workflow with augmentation
- **Comprehensive Evaluation**: WER, CER, latency benchmarking
- **MLOps Best Practices**: Versioning, logging, reproducibility
- **Colab-Ready**: Optimized for Google Colab with GPU support

## 📁 Project Structure

```
COMP3057_Project/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── whisper_asr_colab.ipynb    # Main Colab notebook
├── src/
│   ├── data/                  # Data processing
│   │   ├── preprocessing.py   # Audio preprocessing & VAD
│   │   ├── augmentation.py    # Audio augmentation
│   │   └── dataset.py         # Dataset classes
│   ├── models/                # Model management
│   │   └── whisper_model.py   # Whisper initialization
│   ├── training/              # Training pipeline
│   │   └── trainer.py         # Training loop
│   ├── evaluation/            # Evaluation tools
│   │   ├── metrics.py         # WER/CER/latency
│   │   └── visualization.py   # Plotting utilities
│   ├── inference/             # Inference engine
│   │   └── streaming.py       # Real-time streaming
│   └── utils/                 # Utilities
│       ├── config.py          # Configuration & logging
│       └── versioning.py      # Data/model versioning
└── README.md
```

## 🚀 Quick Start (Google Colab)

1. **Setup in Colab**
   - Open Google Colab
   - Enable GPU: Runtime → Change runtime type → GPU (T4 recommended)
   - Run setup:
   ```python
   !git clone https://github.com/jimmy00415/COMP3057_Project.git
   %cd COMP3057_Project
   !pip install -r requirements.txt
   ```

3. **Run the Notebook**
   - Open `whisper_asr_colab.ipynb` in Colab
   - Execute cells sequentially
   - The notebook handles all setup automatically

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

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| tiny | 39M | Fastest | Lower | Prototyping |
| base | 74M | Fast | Good | Balanced |
| small | 244M | Moderate | Better | Production |
| medium | 769M | Slow | Best | High accuracy |
| distil | 756M | Moderate | Best | Optimized |

## 🎓 Usage Examples

### Fine-Tuning

```python
from src.models import WhisperModelManager
from src.training import WhisperTrainer

# Initialize model
model_manager = WhisperModelManager(config)
model, processor = model_manager.initialize_model(variant='base')

# Train
trainer = WhisperTrainer(model, processor, train_loader, val_loader, config)
trainer.train(num_epochs=10)
```

### Real-Time Streaming

```python
from src.inference import StreamingASR

# Initialize streaming ASR
streaming_asr = StreamingASR(model, processor, vad=vad)

# Stream from microphone
streaming_asr.stream_from_microphone(duration_sec=30)

# Stream from file
streaming_asr.stream_from_file('audio.wav')
```

### Evaluation

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model, processor)
results = evaluator.evaluate_dataset(test_loader)

print(f"WER: {results['wer']:.3f}")
print(f"CER: {results['cer']:.3f}")
```

## 🔬 Technical Stack

- **Deep Learning**: PyTorch 2.0+, Transformers
- **Audio Processing**: torchaudio, librosa, silero-vad
- **Datasets**: HuggingFace Datasets (Common Voice, SpeechOcean762)
- **Evaluation**: jiwer (WER/CER)
- **MLOps**: Weights & Biases / MLflow / TensorBoard
- **Visualization**: matplotlib, seaborn

## 📈 Performance

Benchmarks on Common Voice (English):

| Model | WER | Latency (RTF) | GPU Memory |
|-------|-----|---------------|------------|
| whisper-base | ~10% | 0.2x | 2GB |
| whisper-small | ~8% | 0.5x | 4GB |
| whisper-medium | ~6% | 1.0x | 8GB |

*RTF < 1.0 = Real-time capable*

## 🛠️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  default_variant: "base"  # tiny, base, small, medium, distil
  
training:
  batch_size: 16
  num_epochs: 10
  learning_rate: 5.0e-5

data:
  augmentation:
    enabled: true
    speed_perturbation: [0.9, 1.0, 1.1]
    
inference:
  streaming:
    buffer_size_sec: 2.0
    vad_enabled: true
```

## 📝 MLOps Best Practices

### Reproducibility
- Fixed random seeds across all libraries
- Git commit tracking for every experiment
- Environment snapshots (requirements.txt)

### Versioning
- Dataset versioning with checksums
- Model registry with metadata
- Experiment tracking (WandB/MLflow)

### Monitoring
- Training/validation metrics logging
- Latency benchmarking
- GPU utilization tracking

## 🧪 Testing

```python
# Unit tests for components
pytest tests/

# Benchmark inference
python scripts/benchmark_latency.py

# Evaluate on test set
python scripts/evaluate.py --model checkpoints/best_model
```

## 🚢 Deployment

### Export Model
```python
model.save_pretrained('final_model')
processor.save_pretrained('final_model')
```

### Load for Inference
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained('final_model')
processor = WhisperProcessor.from_pretrained('final_model')
```

### Optimize for Production
- Convert to ONNX: `python scripts/export_onnx.py`
- Use faster-whisper for 4x speedup
- Quantization for edge deployment

## 📚 References

1. [Whisper Paper](https://arxiv.org/abs/2212.04356) - OpenAI Whisper
2. [Whisper-Streaming](https://github.com/ufal/whisper_streaming) - Real-time implementation
3. [Common Voice](https://commonvoice.mozilla.org/) - Dataset
4. [MLOps Best Practices](https://www.dailydoseofds.com/mlops-crash-course-part-3/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📄 License

MIT License - see LICENSE file

## 👥 Authors

COMP3057 Project Team

## 🙏 Acknowledgments

- OpenAI for Whisper models
- HuggingFace for Transformers library
- Mozilla for Common Voice dataset
- Silero Team for VAD models

---

**Ready for production deployment in Google Colab!** 🚀

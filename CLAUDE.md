# CLAUDE.md — AI Assistant Guide for sneezing_detection

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

---

## Project Overview

**sneezing_detection** is a real-time audio-based sneeze detection system built for embedded devices (Raspberry Pi, Jetson Nano). It combines a lightweight CNN trained on MFCC features with a modular Python pipeline for live microphone inference.

The project has two main components:
1. **Training** — Jupyter notebooks in `notebooks/` for experimenting with models and feature extraction
2. **Real-time Inference** — Production system in `realtime_detection/` that runs on-device

---

## Repository Structure

```
sneezing_detection/
├── CLAUDE.md                          # This file
├── README.md                          # Project overview and quick start
├── .gitignore                         # Excludes models/, audio files, datasets
├── legacy_code/                       # Old implementations (do not use)
│   ├── requirements.txt
│   ├── sneeze_synth_engine.py
│   ├── test.py / test_filtered.py
│   ├── train.ipynb / train_flitered.ipynb
│   └── sneeze_model_filtered.keras
├── notebooks/                         # Jupyter training/research notebooks
│   ├── sneeze_detection_lightweight.ipynb   # PRIMARY training notebook
│   ├── sneeze_detection_feature_extraction.ipynb
│   ├── Net_spectogram.ipynb
│   ├── U-net.ipynb
│   ├── YAMnet_fine_tuning.ipynb
│   └── YAMnet_from_none.ipynb
├── scripts/
│   └── save_random_pickles.py         # Data utility for loading pickled datasets
└── realtime_detection/                # PRODUCTION system
    ├── README.md                      # Detailed deployment guide
    ├── main.py                        # Entry point — RealtimeSneezeDetector
    ├── requirements.txt               # Production dependencies
    ├── detected_sneezes/              # Output dir (auto-created, tracked by .gitkeep)
    ├── modules/
    │   ├── __init__.py
    │   ├── audio_capture.py           # Real-time mic input with circular buffer
    │   ├── preprocessing.py           # RMS norm, pre-emphasis, silence trim
    │   ├── mfcc_extractor.py          # MFCC + delta + delta-delta via librosa
    │   ├── model_inference.py         # PyTorch model loading and inference
    │   └── output_handler.py          # Save audio clips, log CSV, cooldown logic
    └── utils/
        ├── __init__.py
        ├── config.py                  # ALL configuration parameters (edit here)
        └── model_definition.py        # LightweightSneezeCNN architecture
```

---

## Audio Processing Pipeline

```
Microphone (16 kHz)
    → [1024-sample chunks via PyAudio callback]
Circular Buffer (4 s / 64000 samples)
    → [2 s sliding window / 32000 samples]
Preprocessing (RMS norm=0.1, pre-emphasis=0.97, silence trim at 20 dB)
    → [~10 ms]
MFCC Extraction (20 coefs + Δ + ΔΔ = 60 features, FFT=2048, hop=512)
    → [~60–80 ms]
LightweightSneezeCNN inference
    → [~15–25 ms]
Output Handler (threshold=0.95, cooldown=1.0 s, save WAV + CSV log)
```

Total latency: ~2.1 s (2 s window + ~100 ms processing)

---

## Model Architecture

**`LightweightSneezeCNN`** (defined in `utils/model_definition.py`):

- Input shape: `(batch, 1, 60, 63)` — MFCC with deltas, 2-second window
- 3× Depthwise Separable Conv blocks (MobileNet-inspired): 1→32→64→128 channels
- Global Average Pooling → FC(128) → FC(64) → FC(2)
- Output: `[not_sneeze_prob, sneeze_prob]`
- ~20,751 parameters, ~0.2 MB (FP32)

The trained model is saved at `models/best_model.pth` (excluded from git via `.gitignore`).

---

## Key Configuration — `realtime_detection/utils/config.py`

All tunable parameters live here. Do not hard-code values in modules.

| Section | Key Parameters |
|---|---|
| Audio | `SAMPLE_RATE=16000`, `CHUNK_SIZE=1024`, `WINDOW_SAMPLES=32000`, `BUFFER_SAMPLES=64000` |
| Preprocessing | `TARGET_RMS=0.1`, `PRE_EMPHASIS=0.97`, `TRIM_DB=20` |
| MFCC | `N_MFCC=20`, `N_FFT=2048`, `HOP_LENGTH=512`, `WINDOW='hann'`, include deltas |
| Model | `MODEL_PATH='../models/best_model.pth'`, `DEVICE='cpu'`, `THRESHOLD=0.95` |
| Output | `OUTPUT_DIR='detected_sneezes/'`, `LOG_CSV=True`, `COOLDOWN_SECONDS=1.0` |
| Performance | PyTorch threads limited to 2 (embedded-friendly) |

---

## Development Workflows

### Running Real-time Detection

```bash
cd realtime_detection
pip install -r requirements.txt
python main.py                          # Default config
python main.py --threshold 0.85         # Lower sensitivity
python main.py --model /path/to/model.pth --verbose
python main.py --quiet
```

### Testing Individual Modules

Each module is self-testable — run as a module from the `realtime_detection/` directory:

```bash
cd realtime_detection
python -m modules.audio_capture
python -m modules.preprocessing
python -m modules.mfcc_extractor
python -m modules.model_inference
python -m modules.output_handler
python -m utils.model_definition
```

### Training a New Model

Open and run `notebooks/sneeze_detection_lightweight.ipynb` (the primary training notebook). It handles:
- Dataset loading (ESC-50 for negatives, custom sneeze recordings for positives)
- Data augmentation (time stretch, pitch shift, noise addition)
- Training the `LightweightSneezeCNN`
- Saving `best_model.pth`

### Raspberry Pi Deployment

See `realtime_detection/README.md` for full instructions including systemd service setup.

---

## Code Conventions

### Module Pattern

Every module in `modules/` follows this structure:

```python
class XModule:
    def __init__(self, config):
        # Initialize from config
        pass

    def method(self, input):
        # Single responsibility
        pass

if __name__ == "__main__":
    # Self-test with dummy data
    pass
```

### Style Guidelines

- **Docstrings**: Google-style on all public classes and methods
- **Type hints**: Use where they aid clarity
- **Error handling**: `try/except` with descriptive messages; never silently swallow errors
- **Logging**: Print-based with emoji status indicators (`🎤 ✓ ❌ 🤧`) — no external logging framework
- **Thread safety**: Use `threading.Lock()` when accessing the shared audio buffer
- **Configuration**: All parameters must come from `config.py`, never hard-coded in modules

### Naming Conventions

- Variables: `mfcc_features`, `sneeze_prob`, `audio_chunk` (descriptive snake_case)
- Classes: `PascalCase` (`AudioCaptureModule`, `LightweightSneezeCNN`)
- Constants in `config.py`: `UPPER_SNAKE_CASE`

---

## Dependencies

### Production (`realtime_detection/requirements.txt`)

| Library | Version | Purpose |
|---|---|---|
| `torch` | ≥2.0.0 | Model inference |
| `librosa` | ≥0.10.0 | MFCC extraction |
| `pyaudio` | ≥0.2.13 | Real-time mic capture |
| `numpy` | ≥1.24.0 | Array operations |
| `scipy` | ≥1.10.0 | Signal processing |
| `soundfile` | ≥0.12.1 | WAV file I/O |

System dependency: `portaudio19-dev` (required by PyAudio on Linux/RPi).

### Training (notebooks)

Additional: `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`/`keras` (legacy notebooks), `jupyter`

---

## What to Avoid

- **Do not edit files in `legacy_code/`** — kept only for historical reference
- **Do not hard-code audio/model parameters** — all changes go through `config.py`
- **Do not commit model weights** (`*.pth`, `*.keras`, `*.h5`) — excluded by `.gitignore`
- **Do not commit audio datasets or recordings** — also excluded by `.gitignore`
- **Do not add GPU-specific code** without a CPU fallback — target hardware is CPU-only embedded devices
- **Do not increase PyTorch thread count** beyond what is in `config.py` on embedded targets

---

## Output Files

Detected sneezes are saved in `realtime_detection/detected_sneezes/`:

```
detected_sneezes/
├── detection_log.csv          # Timestamp, filename, probability for every detection
├── sneeze_20260221_143022.wav
├── sneeze_20260221_143155.wav
└── ...
```

The directory is tracked in git via `.gitkeep` but its contents are gitignored.

---

## Performance Targets

| Platform | CPU Usage | Memory | Latency |
|---|---|---|---|
| Desktop (x86) | 15–25% | ~60–80 MB | ~2.1 s |
| Raspberry Pi 4 | 25–40% | ~60–80 MB | ~2.1 s |

Inference threshold of 0.95 is deliberately high to minimize false positives (life noise from ESC-50 is used as the negative class).

---

## Project Metadata

- **Primary author**: Bahk Insung
- **AI collaborator**: Claude (Anthropic) via Claude Code CLI
- **Last updated**: February 2026
- **License**: Research/Educational

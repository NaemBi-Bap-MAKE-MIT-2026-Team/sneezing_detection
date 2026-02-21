# 🤧 Sneeze Detection System

A comprehensive machine learning project for real-time sneeze detection using log-mel spectrogram features and lightweight CNN models, optimized for embedded systems like Raspberry Pi 4.

## 🎯 Project Overview

This project consists of two main components:

1. **Model Training** (notebooks): Training pipeline for a lightweight CNN model using log-mel spectrogram features
2. **Real-time Detection** ([`src/`](src/)): Production-ready real-time inference system with Hybrid Burst detection

### Key Features

- ✅ **Lightweight Architecture**: ~20K parameters TFLite model for efficient inference on RPi 4
- ✅ **Log-Mel Spectrogram**: 64-band mel spectrogram features matching v4 training settings exactly
- ✅ **Hybrid Burst Detection**: RMS guard gate + burst sliding inference — minimal CPU when quiet
- ✅ **Modular Design**: Clean separation of audio, model, and output concerns
- ✅ **Dual Input Modes**: Local microphone or UDP network audio stream
- ✅ **GIF Animation Output**: ST7789 240×240 LCD with per-frame GIF playback on detection
- ✅ **Raspberry Pi Ready**: Optimized for RPi 4 deployment

## 📁 Project Structure

```
sneezing_detection/
├── README.md
├── src/                                        # RPi deployment (v4, modular)
│   ├── main.py                                 # Entry point — Hybrid Burst detector
│   ├── v4_model.tflite                         # TFLite model weights
│   ├── v4_norm_stats.npz                       # Z-score normalisation statistics
│   ├── communication/                          # Unified audio-input layer
│   │   ├── send.py                             # AudioSender  — mic → UDP stream
│   │   ├── recv.py                             # NetworkMicStream — UDP → mic API
│   │   ├── README.md                           # Communication module docs
│   │   ├── example_local.py                    # Local mic example
│   │   └── example_network.py                  # Network sender/receiver example
│   ├── microphone/
│   │   └── mic_input.py                        # MicrophoneStream (sounddevice)
│   ├── ml_model/
│   │   ├── config.py                           # All constants (SR, thresholds, …)
│   │   ├── preprocessing.py                    # rms, logmel, preproc, load_stats
│   │   └── model.py                            # LiteModel (TFLite wrapper)
│   ├── output_feature/
│   │   ├── images/bless_you.gif                # Detection animation (GIF)
│   │   ├── sounds/bless_you.wav                # Detection sound
│   │   ├── speaker_output.py                   # SpeakerOutput (alert / beep)
│   │   └── lcd_output.py                       # LCD + GifAnimator (ST7789 240×240)
│   └── requirements_rpi4.txt                   # RPi 4 production dependencies
├── raspi/sneeze-detection/
│   └── real_time_detection.py                  # Original standalone RPi script (reference)
├── realtime_detection/                         # Legacy MFCC-based detection (reference)
│   └── main.py
├── legacy_code/                                # v1–v4 reference implementations
│   └── output/v4/
│       ├── test_no_saving_v4.py
│       └── test_saving_v4.py
└── notebooks/                                  # Training notebooks
    ├── sneeze_detection_lightweight.ipynb
    └── YAMnet_fine_tuning.ipynb
```

## 🚀 Quick Start

### 1. Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/sneeze_detection_lightweight.ipynb
```

### 2. Real-time Detection (`src/`)

#### Asset layout

Place model weights and assets under `src/` before running:

```
src/
├── v4_model.tflite
├── v4_norm_stats.npz
└── output_feature/
    ├── images/  bless_you.gif
    └── sounds/  bless_you.wav
```

---

#### Mode A — Local microphone (single device)

The simplest setup: microphone and inference run on the same Raspberry Pi.

```bash
cd src

# With LCD (ST7789 attached)
python main.py

# Without LCD
python main.py --no-lcd
```

---

#### Mode B — Network streaming (two devices)

Use this when the **microphone is on one device** (e.g. a PC or second RPi) and
**inference runs on another** (e.g. the main Raspberry Pi).

```
[Device A — Sender]                [Device B — Receiver / RPi]
  has microphone                     runs inference + LCD + speaker

  python send.py                     python main.py --network
    --host <Device-B-IP>               --recv-host 0.0.0.0
    --port 12345                       --recv-port 12345
                                       --no-lcd          ← optional
```

**Device A — Sender** (machine with microphone):

```bash
cd src
python communication/send.py --host 192.168.1.42 --port 12345
```

**Device B — Receiver** (Raspberry Pi running inference):

```bash
cd src
python main.py --network --recv-host 0.0.0.0 --recv-port 12345
```

Loopback test on a single machine (both terminals in `src/`):

```bash
# Terminal 1
python main.py --network --no-lcd

# Terminal 2
python communication/send.py --host 127.0.0.1 --port 12345
```

---

#### `main.py` — All arguments

| Argument | Default | Description |
| --- | --- | --- |
| *(none)* | — | Local microphone, LCD enabled |
| `--network` | off | Receive audio from `send.py` over UDP |
| `--recv-host` | `0.0.0.0` | `[--network]` UDP bind address |
| `--recv-port` | `12345` | `[--network]` UDP port |
| `--no-lcd` | off | Disable the ST7789 LCD driver |

#### `send.py` — All arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Destination IP (receiver device) |
| `--port` | `12345` | Destination UDP port |
| `--capture-sr` | `48000` | Microphone sample rate (Hz) |
| `--block-ms` | `10` | Packet size in ms (lower = less latency) |
| `--device` | system default | sounddevice input device index or name |

---

## 📊 Model Performance

### Training Results
- **Architecture**: LightweightSneezeCNN (Depthwise Separable Convolutions)
- **Parameters**: ~20,751
- **Model Size**: ~0.2 MB (FP32 TFLite)
- **Test Accuracy**: >90%
- **F1 Score**: ~0.87–0.90

### Real-time Performance (Hybrid Burst mode)
- **IDLE CPU**: near-zero (only RMS gate check per frame)
- **BURST inference**: ~80–120 ms per 2-second window on RPi 4
  - Resample (48k→16k): ~5 ms
  - Log-mel spectrogram: ~10 ms
  - TFLite inference: ~15–25 ms
- **Worst-case detection latency**: trigger delay + 0.5 s hop ≈ 0.6–1.1 s
- **Memory**: ~60 MB

---

## 🧠 Technical Details

### Detection Pipeline (Hybrid Burst)

```
Mic (48 kHz)
  │
  ▼
Frame (100 ms)  ──►  RMS < threshold?  ──► [IDLE: discard, no inference]
                              │
                         RMS ≥ threshold
                              │
                              ▼
                        [BURST mode, 3 s]
                              │
                        every 0.5 s hop
                              │
                              ▼
              Last 2 s of ring buffer (48 kHz)
                              │
                       Resample → 16 kHz
                              │
                       RMS normalisation
                              │
                    Log-mel spectrogram (64 mels)
                              │
                       Z-score normalisation
                              │
                     TFLite inference (1, 199, 64, 1)
                              │
                      p ≥ 0.90?  ──► Bless you! + LCD GIF + WAV
```

### Audio Feature Extraction

| Parameter | Value | Note |
|---|---|---|
| Capture SR | 48 000 Hz | stable across most RPi drivers |
| Model SR | 16 000 Hz | resample before feature extraction |
| Clip length | 2.0 s | window fed into model |
| n_mels | 64 | mel filter banks |
| n_fft | 400 | FFT window |
| hop_length | 160 | frame stride |
| center | False | **must match v4 training** |
| Target RMS | 0.1 | per-clip normalisation |

### Detection Thresholds (`src/ml_model/config.py`)

| Parameter | Value | Description |
|---|---|---|
| `RMS_TRIGGER_TH` | `0.008` | RMS level that enters BURST mode |
| `PROB_TH` | `0.90` | Sneeze probability to trigger detection |
| `COOLDOWN_SEC` | `1.5` | Silence period after a detection |
| `FRAME_SEC` | `0.10` | RMS guard frame duration |

### Model Architecture

```
Input: (1, 199, 64, 1)   — (batch, time_frames, n_mels, channel)
  → Conv2D blocks (Depthwise Separable)
  → GlobalAveragePooling
  → FC + Dropout
Output: (1, 2)            — [non-sneeze prob, sneeze prob]
```

---

## 🔧 Configuration

All tunable parameters are in [`src/ml_model/config.py`](src/ml_model/config.py):

```python
# Audio capture
CAPTURE_SR   = 48000   # microphone sample rate (Hz)
MODEL_SR     = 16000   # model input sample rate (Hz)

# Analysis window
CLIP_SECONDS = 2.0     # inference window length
FRAME_SEC    = 0.10    # RMS guard frame duration

# Detection thresholds
RMS_TRIGGER_TH = 0.008   # enter BURST when RMS exceeds this
PROB_TH        = 0.90    # sneeze probability threshold
COOLDOWN_SEC   = 1.5     # post-detection cooldown (seconds)

# Feature extraction — MUST match v4 training
N_MELS = 64
N_FFT  = 400
HOP    = 160
CENTER = False           # IMPORTANT: must be False
```

---

## 📦 Dependencies

### RPi 4 Runtime (`src/requirements_rpi4.txt`)
- `numpy` — numerical computing
- `librosa` — audio resampling + log-mel spectrogram
- `sounddevice` — microphone capture
- `pillow` — image loading for LCD
- `ai-edge-litert` / `tflite-runtime` — TFLite inference
- `st7789` — ST7789 LCD SPI driver (RPi only)

### For Training
- `torch>=2.0.0` — deep learning framework
- `librosa>=0.10.0` — audio feature extraction
- `polars` — fast dataframe operations
- `scikit-learn` — ML utilities
- `matplotlib` — visualization

---

## 🎓 Dataset

### Training Data
- **Positive Samples**: 968 sneeze recordings
- **Negative Samples**: 968 life noise samples from ESC-50
  - Categories: coughing, breathing, laughing, footsteps, etc.
- **Augmentation**: 2× data (1 936 positive + 1 936 negative)

### ESC-50 Dataset
The project uses [ESC-50](https://github.com/karolpiczak/ESC-50) (Environmental Sound Classification) dataset for negative samples.

```bash
# Download ESC-50 (if needed)
git clone https://github.com/karolpiczak/ESC-50.git esc-50
```

---

## 📈 Future Improvements

### Short-term
- [ ] INT8 quantization for lower inference latency on RPi 4
- [ ] Tune `BURST_SECONDS` and `HOP_SEC` per deployment environment
- [ ] Web interface for monitoring

### Long-term
- [ ] Edge TPU support (Google Coral)
- [ ] Multi-class detection (sneeze, cough, speech)
- [ ] Mobile app with on-device inference
- [ ] Cloud integration for logging

---

## 🤖 Built with AI

This project was developed collaboratively with **[Claude](https://claude.ai)** (Anthropic's AI assistant) using the **[Claude Code](https://claude.ai/claude-code)** CLI tool.

## 👨‍💻 Author

**Bahk Insung**
- GitHub: [@bahk_insung](https://github.com/bahk_insung)
- Developed in collaboration with Claude (Anthropic)
- Date: February 2026

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- **Training Data**: [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sound dataset
- **Model Architecture**: Inspired by MobileNet's Depthwise Separable Convolutions
- **Feature Extraction**: [Librosa](https://librosa.org/) library
- **AI Development**: [Claude](https://claude.ai) by Anthropic
- **Development Tool**: [Claude Code](https://claude.ai/claude-code) CLI

## 📚 References

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [ESC-50 Dataset Paper](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)
- [Depthwise Separable Convolutions](https://paperswithcode.com/method/depthwise-separable-convolution)

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**🎯 Built with Human-AI Collaboration · 🤧 Powered by Machine Learning · 🚀 Ready for Production**

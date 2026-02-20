# Codebase Documentation — `src/` (Raspberry Pi Branch)

## Overview

The `src/` directory is the **production source tree for the Raspberry Pi deployment** of the sneezing detection system. It provides a fully modular, hardware-aware implementation that:

- Captures audio from a **local USB / I2S microphone** or from a **remote device over UDP**.
- Runs **TFLite V4 inference** on the captured clip.
- Outputs alerts via a **ST7789 240×240 LCD** (animated frames) and a **speaker** (sine-wave beep).

The code is designed to run on a Raspberry Pi 4 (2 GB RAM) with `tflite_runtime` installed; every module also works on a desktop with full TensorFlow.

---

## Directory Tree

```
src/
├── __init__.py
├── main.py                          # Entry point — wires all modules together
│
├── microphone/
│   ├── __init__.py
│   └── mic_input.py                 # MicrophoneStream (local mic via sounddevice)
│
├── communication/
│   ├── __init__.py
│   ├── send.py                      # AudioSender — capture mic, stream over UDP
│   ├── recv.py                      # NetworkMicStream — receive UDP audio frames
│   ├── example_local.py             # Runnable example: local mic usage
│   └── example_network.py           # Runnable example: sender / receiver roles
│
├── ml_model/
│   ├── __init__.py
│   ├── config.py                    # All tunable constants (paths, rates, thresholds)
│   ├── model.py                     # LiteModel — TFLite wrapper
│   └── preprocessing.py             # Audio processing pipeline (rms, logmel, preproc …)
│
├── output_feature/
│   ├── __init__.py
│   ├── lcd_output.py                # LCD, LCDAnimator — ST7789 display driver
│   ├── speaker_output.py            # SpeakerOutput — beep / WAV playback
│   └── images/
│       ├── bless_you.gif            # Animated GIF played on detection (optional)
│       ├── detect1.jpg              # Animation frame 1
│       ├── detect2.jpg              # Animation frame 2
│       ├── detect3.jpg              # Animation frame 3
│       └── detect4.jpg              # Animation frame 4
│
└── tests/
    ├── __init__.py
    ├── test_communication.py        # UDP loopback tests (no real mic required)
    ├── test_ml_model.py             # Preprocessing + TFLite inference tests
    └── test_lcd.py                  # LCD slideshow / GIF test script
```

---

## Module Reference

### `main.py` — Entry Point

**Purpose:** Orchestrates all subsystems and runs the real-time detection loop.

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--network` | off | Receive audio via UDP instead of a local mic |
| `--recv-host` | `0.0.0.0` | UDP bind address (used with `--network`) |
| `--recv-port` | `12345` | UDP port (used with `--network`) |
| `--no-lcd` | off | Disable the ST7789 LCD driver |

**Detection loop logic:**

```
Start
  │
  ├─ load model (LiteModel) + normalisation stats (load_stats)
  ├─ open MicrophoneStream or NetworkMicStream
  ├─ initialise SpeakerOutput + LCDAnimator
  │
  └─ loop
       │
       ├─ read frame from mic
       ├─ skip if within cooldown window
       │
       ├─ [NOT capturing] push frame to pre_buffer
       │      if rms(frame) >= RMS_TRIGGER_TH → start capture
       │
       └─ [capturing] accumulate frames until clip_samples collected
              │
              ├─ resample 48 kHz → 16 kHz
              ├─ preproc (pad/trim → RMS norm → logmel → z-score)
              ├─ LiteModel.predict_proba(x_in)
              │
              └─ if prob >= PROB_TH:
                     print "Bless you!"
                     SpeakerOutput.alert()
                     LCDAnimator.trigger()      ← daemon thread, non-blocking
                     ignore_until = now + COOLDOWN_SEC
```

**Required asset files** (placed alongside the script or in `~/Documents/sneeze-detection/`):

| File | Description |
|---|---|
| `v4_model.tflite` | Quantised TFLite model |
| `v4_norm_stats.npz` | Per-mel-band µ and σ statistics |
| `images/idle.png` | LCD idle frame |
| `images/detect1–3.png` | LCD detection animation frames |

---

### `microphone/mic_input.py` — `MicrophoneStream`

**Purpose:** Continuous microphone capture with a rolling pre-roll buffer.

**Class:** `MicrophoneStream`

| Parameter | Default | Description |
|---|---|---|
| `capture_sr` | `48000` | Microphone sample rate (Hz) |
| `frame_sec` | `0.10` | Duration of each chunk (s) |
| `pre_seconds` | `0.5` | Pre-roll buffer length (s) |
| `device` | `None` | sounddevice device index (None = system default) |

**Key attributes / methods:**

| Member | Description |
|---|---|
| `pre_buffer` | `collections.deque` of recent frames (auto-sized from `pre_seconds / frame_sec`) |
| `open()` | Start the sounddevice InputStream |
| `close()` | Stop and close the stream |
| `read()` | Blocking call — returns one `(frame_samples,) float32` numpy array |
| Context manager | `with MicrophoneStream(...) as mic:` calls open/close automatically |

**Internal flow:** A sounddevice callback pushes each chunk into a thread-safe `queue.Queue`; `read()` blocks on `queue.get()`.

---

### `communication/` — Audio Input Abstraction

This package provides two interchangeable audio sources with an **identical public API** (`read()` + `pre_buffer`), so `main.py` switches between local and network modes with zero logic changes.

---

#### `send.py` — `AudioSender`

**Purpose:** Capture a local microphone and stream float32 audio frames over UDP to a `NetworkMicStream` receiver.

**Class:** `AudioSender`

| Parameter | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | Destination IP address |
| `port` | `12345` | Destination UDP port |
| `capture_sr` | `48000` | Microphone sample rate (Hz) |
| `block_ms` | `10.0` | Packet size (ms) — lower = less latency |
| `device` | `None` | sounddevice device index |

**Method:** `run()` — opens the mic and sends each frame as raw `float32` bytes over UDP until `KeyboardInterrupt`.

**CLI usage:**
```bash
# Stream to a Raspberry Pi at 192.168.1.42:
python send.py --host 192.168.1.42 --port 12345

# Custom block size (5 ms ≈ 200 packets/s):
python send.py --host 192.168.1.42 --block-ms 5
```

---

#### `recv.py` — `NetworkMicStream`

**Purpose:** Receive UDP audio packets from `AudioSender` and expose the exact same interface as `MicrophoneStream`.

**Class:** `NetworkMicStream`

| Parameter | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Bind address |
| `port` | `12345` | UDP port to listen on |
| `capture_sr` | `48000` | Expected sample rate (Hz) |
| `frame_sec` | `0.10` | Output frame duration (s) |
| `pre_seconds` | `0.5` | Pre-roll buffer length (s) |

**Internal flow:**
1. A background daemon thread (`_recv_loop`) calls `recvfrom()` with a 0.5 s timeout (so `_running` is checked regularly).
2. Incoming raw `float32` bytes are appended to an accumulation buffer.
3. When the buffer holds at least `frame_samples` samples, a complete frame is sliced off and put into a `queue.Queue`.
4. The main thread's `read()` call blocks on `queue.get()`, identical to `MicrophoneStream`.

**Public API:** identical to `MicrophoneStream` — `open()`, `close()`, `read()`, `pre_buffer`, context manager.

---

#### `example_local.py` and `example_network.py`

Runnable demonstration scripts:

```bash
# Local mic (no network):
python communication/example_local.py

# Two-terminal network demo (same machine):
# Terminal A:
python communication/example_network.py --role sender --host 127.0.0.1
# Terminal B:
python communication/example_network.py --role receiver
```

Both examples print per-frame shape, dtype, and RMS — confirming that `NetworkMicStream` output is indistinguishable from `MicrophoneStream`.

---

### `ml_model/` — Model & Preprocessing

#### `config.py` — Configuration Constants

All tunable parameters live here. **Never hard-code these values elsewhere.**

| Constant | Value | Description |
|---|---|---|
| `TFLITE_PATH` | `v4_model.tflite` | TFLite model file |
| `STATS_PATH` | `v4_norm_stats.npz` | Normalisation statistics file |
| `INPUT_DEVICE` | `None` | sounddevice input device (None = default) |
| `CAPTURE_SR` | `48000` | Microphone capture rate (Hz) |
| `MODEL_SR` | `16000` | Model input rate (Hz) |
| `CLIP_SECONDS` | `2.0` | Analysis window duration (s) |
| `PRE_SECONDS` | `0.5` | Pre-roll buffer duration (s) |
| `FRAME_SEC` | `0.10` | Audio chunk size (s) |
| `RMS_TRIGGER_TH` | `0.008` | RMS level that starts capture |
| `PROB_TH` | `0.90` | Sneeze probability threshold |
| `COOLDOWN_SEC` | `1.5` | Post-detection silence window (s) |
| `N_MELS` | `64` | Mel filter bank count |
| `N_FFT` | `400` | FFT window size |
| `HOP` | `160` | Hop length |
| `CENTER` | `False` | Librosa centering — **must match training** |
| `TARGET_RMS` | `0.1` | RMS normalisation target |

---

#### `model.py` — `LiteModel`

**Purpose:** Thin TFLite wrapper for single-output (probability) classification.

**Constructor:** `LiteModel(path: Path)` — loads the `.tflite` file and calls `allocate_tensors()`. Falls back from `tflite_runtime` to `tensorflow.lite` automatically.

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `predict_proba(x)` | `float` | Set input tensor → invoke → return scalar output |
| `expected_input_shape()` | `np.ndarray \| None` | Expose model's declared input shape |

**Tensor shapes:**

| Tensor | Shape | dtype |
|---|---|---|
| Input | `(1, frames, 64, 1)` | `float32` |
| Output | `(1, 1)` | `float32` |

---

#### `preprocessing.py` — Audio Pipeline

Five standalone functions plus a full pipeline function:

| Function | Signature | Description |
|---|---|---|
| `rms(x)` | `(ndarray) → float` | Compute RMS amplitude |
| `normalize_rms(x, target)` | `(ndarray, float) → ndarray` | Scale to target RMS, clip to `[-1, 1]` |
| `logmel(y_16k_2s)` | `(ndarray) → ndarray (frames, 64)` | Log-mel spectrogram (matches V4 training) |
| `load_stats(stats_path)` | `(Path) → (mu, sdv)` | Load per-mel µ/σ from `.npz`; handles shape variants `(64,)`, `(1,1,64)`, `(1,64)` |
| `preproc(y16, mu, sdv)` | `(ndarray, ndarray, ndarray) → ndarray (1, frames, 64, 1)` | Full pipeline (see below) |
| `resample_to_model_sr(y48, capture_sr)` | `(ndarray, int) → ndarray` | Resample 48 kHz → 16 kHz via librosa |

**`preproc()` pipeline:**

```
y16 (16 kHz, any length)
  │
  ├─ pad to 32 000 samples OR trim to 32 000 samples   (2 s @ 16 kHz)
  ├─ normalize_rms(target=0.1), clip to [-1, 1]
  ├─ logmel → shape (frames, 64)
  ├─ z-score: (f - mu) / (sdv + 1e-6)   per mel band
  └─ reshape → (1, frames, 64, 1)        TFLite-ready
```

---

### `output_feature/` — Hardware Output

#### `lcd_output.py` — LCD Display

**Class `LCD`**

Wraps the `st7789` SPI driver.

| Parameter (hardware) | Value |
|---|---|
| Display | ST7789 240×240 |
| Rotation | 90° |
| SPI port / CS | 0 / 1 |
| DC GPIO | 9 |
| Backlight GPIO | 13 |
| SPI speed | 80 MHz |

**Method:** `show(img: PIL.Image)` — display a 240×240 RGB PIL Image.

**Helper:** `load_frame(path: Path) → Image` — open any image file, convert to RGB, resize to 240×240.

---

**Class `LCDAnimator`**

Plays a detection animation (N frames) then returns to the idle frame. Animation runs in a **daemon thread** so `main.py` is never blocked.

| Parameter | Description |
|---|---|
| `lcd` | `LCD` instance |
| `idle_path` | Path to idle image (displayed when no sneeze detected) |
| `frame_paths` | List of animation frame paths (played in order) |
| `fps` | Animation speed (default 12.0) |

**Methods:**

| Method | Description |
|---|---|
| `start()` | Display the idle frame immediately |
| `trigger()` | Spawn daemon thread: play all frames at `fps`, then restore idle. Duplicate triggers while playing are silently ignored (mutex guard) |

---

#### `speaker_output.py` — `SpeakerOutput`

**Purpose:** Audible alert on detection — console print + sine-wave beep.

| Parameter | Default | Description |
|---|---|---|
| `playback_sr` | `16000` | Speaker sample rate (Hz) |
| `device` | `None` | sounddevice output device |
| `volume` | `1.0` | Scale factor clipped to `[0.0, 1.0]` |

**Methods:**

| Method | Description |
|---|---|
| `alert(message)` | Print message + play 200 ms 880 Hz beep |
| `play_wav(audio)` | Play a float32 numpy array through the speaker |
| `_beep(duration_ms, freq_hz)` | Internal: generate + play a sine tone |

Replace `_beep` with a GPIO buzzer, LED flash, or TTS call as needed.

---

#### `output_feature/images/`

Static assets bundled with the source:

| File | Used by |
|---|---|
| `bless_you.gif` | Optional — can be used with `test_lcd.py` |
| `detect1.jpg` – `detect4.jpg` | Animation frames for `LCDAnimator` |

Production idle / detect PNGs are stored in `~/Documents/sneeze-detection/images/` on the device.

---

### `tests/` — Test Scripts

All tests are runnable directly from `src/` with no test framework required.

#### `test_communication.py`

Tests the UDP loopback pipeline **without a real microphone or running `send.py`**. A background thread acts as a fake sender by pushing raw `float32` zero packets.

| Test | What it checks |
|---|---|
| `test_loopback_sample_count` | Transmitted samples == received samples |
| `test_pre_buffer_updates` | `pre_buffer` deque grows up to `maxlen` |
| `test_context_manager_clean_exit` | `open()` / `close()` via `with` statement do not raise |
| `test_frame_dtype_and_shape` | Each `read()` frame is `float32`, 1-D, exactly `frame_samples` long |

```bash
python tests/test_communication.py
```

---

#### `test_ml_model.py`

Tests the preprocessing pipeline and TFLite inference. Tests that require missing model files are **automatically skipped** (SKIP rather than FAIL).

| Test | What it checks |
|---|---|
| `test_load_stats` | `mu` and `sdv` have shape `(64,)`, dtype `float32` |
| `test_preproc_shape` | `preproc()` output is `(1, frames, 64, 1)`, dtype `float32` |
| `test_model_inference` | `predict_proba()` returns a `float` in `[0.0, 1.0]` |
| `test_model_input_shape` | Model's declared input shape matches `preproc()` output |

```bash
python tests/test_ml_model.py
```

---

#### `test_lcd.py`

Interactive slideshow for validating LCD wiring and image assets. Supports static images (`.jpg`, `.png`) and animated GIFs with per-frame timing.

```bash
# Default: auto-discover images in output_feature/images/, 0.3 s per frame
python tests/test_lcd.py

# Single GIF with native timing:
python tests/test_lcd.py --images output_feature/images/bless_you.gif

# Override timing to 0.2 s, run 3 loops:
python tests/test_lcd.py --interval 0.2 --loops 3

# Show each image once and exit:
python tests/test_lcd.py --once
```

---

## Data Flow Diagram

```
[Microphone / UDP]
        │
        │  float32 frames @ 48 kHz
        ▼
  MicrophoneStream  ──or──  NetworkMicStream
        │
        │  pre_buffer (rolling 500 ms)
        │  rms trigger check
        ▼
  Accumulate 2-second clip (48 kHz)
        │
        ▼
  resample_to_model_sr  →  16 kHz
        │
        ▼
  preproc(y16, mu, sdv)
    ├─ pad / trim to 32 000 samples
    ├─ RMS normalise
    ├─ log-mel spectrogram (64 bands)
    └─ z-score normalise
        │
        │  (1, frames, 64, 1)
        ▼
  LiteModel.predict_proba
        │
        │  float in [0, 1]
        ▼
  prob >= 0.90?
     YES ──► SpeakerOutput.alert()  +  LCDAnimator.trigger()
     NO  ──► discard, continue
```

---

## Running the System

### Local microphone (default)

```bash
cd src
python main.py
```

### Receive audio from another device

```bash
# Device A (laptop / phone with mic):
cd src
python communication/send.py --host <RPi_IP> --port 12345

# Raspberry Pi:
cd src
python main.py --network --recv-host 0.0.0.0 --recv-port 12345
```

### Without LCD hardware

```bash
python main.py --no-lcd
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Array operations throughout |
| `librosa` | Resampling, mel spectrogram |
| `sounddevice` | Microphone capture + speaker playback |
| `tflite_runtime` | Lightweight TFLite inference on RPi |
| `tensorflow` | Fallback TFLite on desktop |
| `Pillow` | Image loading and resizing for LCD |
| `st7789` | ST7789 SPI LCD driver (RPi only) |

Install on Raspberry Pi:
```bash
pip install numpy librosa sounddevice Pillow
pip install tflite-runtime
pip install st7789
```

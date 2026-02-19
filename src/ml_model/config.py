from pathlib import Path

# Model / stats file paths (override via environment or at runtime)
TFLITE_PATH = Path("v4_model.tflite")
STATS_PATH  = Path("v4_norm_stats.npz")

# Audio capture
INPUT_DEVICE = None   # None = system default
CAPTURE_SR   = 48000  # capture sample rate (48k is stable across most drivers)
MODEL_SR     = 16000  # model input sample rate

# Analysis window
CLIP_SECONDS = 2.0
PRE_SECONDS  = 0.5
FRAME_SEC    = 0.10

# Detection thresholds
RMS_TRIGGER_TH = 0.008  # RMS level that starts capture
PROB_TH        = 0.90   # sneeze probability threshold
COOLDOWN_SEC   = 1.5    # seconds to ignore after a detection

# Feature extraction — must match v4 training settings exactly
N_MELS     = 64
N_FFT      = 400
HOP        = 160
CENTER     = False   # IMPORTANT: must be False to match v4 training
TARGET_RMS = 0.1

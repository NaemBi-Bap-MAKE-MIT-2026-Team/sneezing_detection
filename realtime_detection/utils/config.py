"""
Configuration file for Real-time Sneeze Detection System

This file contains all parameters used during training and must be kept
consistent with the training pipeline to ensure accuracy.
"""

# ============================================================================
# Audio Capture Parameters
# ============================================================================
SAMPLE_RATE = 16000      # Hz - All audio resampled to this rate
CHUNK_SIZE = 1024        # PyAudio buffer size (64ms at 16kHz)
WINDOW_SIZE = 32000      # Analysis window size (2 seconds = 32000 samples)
BUFFER_SIZE = 64000      # Circular buffer size (4 seconds)
OVERLAP = 0.5            # Overlap ratio for sliding window (50% = 1s stride)

# ============================================================================
# Preprocessing Parameters (MUST match training)
# ============================================================================
TARGET_RMS = 0.1         # Target RMS level after normalization
PRE_EMPHASIS = 0.97      # Pre-emphasis filter coefficient
TRIM_DB = 20             # dB threshold for silence trimming

# ============================================================================
# MFCC Parameters (MUST match training)
# ============================================================================
N_MFCC = 20              # Number of MFCC coefficients
N_FFT = 2048             # FFT window size
HOP_LENGTH = 512         # Number of samples between frames
WINDOW_TYPE = 'hann'     # Window function
INCLUDE_DELTAS = True    # Include Delta and Delta-Delta features
                         # Total features: 20 + 20 + 20 = 60

# Expected MFCC output shape for 2-second audio at 16kHz:
# (60, 63) = (features, time_frames)

# ============================================================================
# Model Parameters
# ============================================================================
MODEL_PATH = "../models/best_model.pth"  # Path to trained model weights
DEVICE = "cpu"           # Device for inference: "cpu" or "cuda"
THRESHOLD = 0.8          # Detection threshold (probability > 0.8 = sneeze)

# Model input shape
MODEL_INPUT_HEIGHT = 60  # MFCC features (20 + 20 deltas + 20 delta-deltas)
MODEL_INPUT_WIDTH = 63   # Time frames for 2-second audio

# ============================================================================
# Output Parameters
# ============================================================================
SAVE_DIR = "detected_sneezes"  # Directory for saving detected audio clips
COOLDOWN_SECONDS = 1.0         # Cooldown period to prevent duplicate detections
ENABLE_LOGGING = True          # Enable CSV logging of detections
LOG_FILE = "detection_log.csv" # CSV log file name

# ============================================================================
# Performance Parameters
# ============================================================================
# Threading
USE_THREADING = False    # Enable multi-threading (not implemented yet)

# PyTorch optimization
TORCH_NUM_THREADS = 2    # Limit PyTorch CPU threads (for Raspberry Pi)

# Verbosity
VERBOSE = True           # Print detection probabilities
PRINT_INTERVAL = 10      # Print probability every N detections (if not sneeze)

"""
v4 Pipeline Functions
=====================
Preprocessing functions extracted from ``legacy_code/output/v4/test_saving_v4.py``
for use in the RPi4 simulation environment.

This module is **self-contained** — it does not import from the original v4
scripts (which depend on ``sounddevice`` and other hardware-specific packages).

Also provides ``create_mock_stats()`` for generating dummy normalisation
statistics when the real ``.npz`` file is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import librosa


# ===================================================================== #
#  v4 Constants (must match training configuration)
# ===================================================================== #

CAPTURE_SR: int = 48_000       # Microphone capture sample rate
MODEL_SR: int = 16_000         # Model input sample rate
CLIP_SECONDS: float = 2.0      # Audio clip length in seconds
TARGET_RMS: float = 0.1        # RMS normalisation target amplitude

N_MELS: int = 64               # Number of mel filter banks
N_FFT: int = 400                # FFT window size
HOP: int = 160                  # Hop length between frames
CENTER: bool = False            # Critical: must be False for v4

# Derived
TARGET_SAMPLES_16K: int = int(MODEL_SR * CLIP_SECONDS)   # 32 000
TARGET_SAMPLES_48K: int = int(CAPTURE_SR * CLIP_SECONDS)  # 96 000

# Trigger / detection thresholds (for reference, not used in pipeline)
RMS_TRIGGER_TH: float = 0.008
PROB_TH: float = 0.90
COOLDOWN_SEC: float = 1.5


# ===================================================================== #
#  Preprocessing functions  (verbatim from test_saving_v4.py)
# ===================================================================== #

def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    """Root-mean-square of an audio array."""
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    """Scale *x* so that its RMS equals *target*, then clip to [-1, 1]."""
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def logmel(y_16k_2s: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching v4 training parameters.

    Returns shape ``(frames, N_MELS)`` — typically ``(200, 64)``.
    """
    S = librosa.feature.melspectrogram(
        y=y_16k_2s,
        sr=MODEL_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
        center=CENTER,
    )
    return np.log(S + 1e-6).T.astype(np.float32)


def preproc(
    y16: np.ndarray,
    mu: np.ndarray,
    sdv: np.ndarray,
) -> np.ndarray:
    """Full v4 preprocessing: pad/trim -> RMS normalise -> logmel -> z-score.

    Parameters
    ----------
    y16 : ndarray, shape (samples,)
        16 kHz mono audio (float32).
    mu : ndarray, shape (N_MELS,)
        Per-band mean from training statistics.
    sdv : ndarray, shape (N_MELS,)
        Per-band standard deviation from training statistics.

    Returns
    -------
    ndarray, shape (1, frames, N_MELS, 1) — ready for TFLite input.
    """
    target = TARGET_SAMPLES_16K
    y16 = np.asarray(y16, dtype=np.float32)

    # Pad or trim to exactly 2 seconds
    if len(y16) > target:
        y16 = y16[:target]
    elif len(y16) < target:
        y16 = np.pad(y16, (0, target - len(y16)))

    y16 = normalize_rms(y16)
    f = logmel(y16)                                   # (frames, mels)
    fn = (f - mu[None, :]) / (sdv[None, :] + 1e-6)   # z-score normalise
    return fn[None, :, :, None].astype(np.float32)    # (1, frames, mels, 1)


# ===================================================================== #
#  Resample helper
# ===================================================================== #

def resample_48k_to_16k(y48: np.ndarray) -> np.ndarray:
    """Resample 48 kHz audio to 16 kHz (matching v4 pipeline)."""
    return librosa.resample(
        y48.astype(np.float32),
        orig_sr=CAPTURE_SR,
        target_sr=MODEL_SR,
    ).astype(np.float32)


# ===================================================================== #
#  Mock normalisation statistics
# ===================================================================== #

def create_mock_stats() -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu, sdv) arrays that make z-score normalisation a no-op.

    ``mu = 0``, ``sdv = 1`` for each of the 64 mel bands, so
    ``(f - mu) / sdv == f``.
    """
    mu = np.zeros(N_MELS, dtype=np.float32)
    sdv = np.ones(N_MELS, dtype=np.float32)
    return mu, sdv


def load_stats(stats_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalisation statistics from ``.npz`` file.

    Handles the various shapes that may be stored (identical to the
    original v4 implementation).  Falls back to ``create_mock_stats()``
    if the file does not exist.
    """
    path = Path(stats_path)
    if not path.exists():
        print(f"[v4_pipeline] Stats file not found: {path}  -> using mock stats")
        return create_mock_stats()

    st = np.load(str(path), allow_pickle=True)
    mu = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    # Flatten from (1,1,64) or (1,64) to (64,)
    if mu.ndim == 3 and mu.shape[0] == 1 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if sdv.ndim == 3 and sdv.shape[0] == 1 and sdv.shape[1] == 1:
        sdv = sdv.reshape(-1)

    if mu.ndim == 3 and mu.shape[0] == 1:
        mu = mu[0]
    if sdv.ndim == 3 and sdv.shape[0] == 1:
        sdv = sdv[0]

    mu = mu.reshape(-1).astype(np.float32)
    sdv = sdv.reshape(-1).astype(np.float32)
    return mu, sdv


# ===================================================================== #
#  MockLiteModel  (uses MockInterpreter under the hood)
# ===================================================================== #

class MockLiteModel:
    """Drop-in replacement for v4's ``LiteModel`` backed by ``MockInterpreter``.

    Parameters
    ----------
    fake_output : float or list[float]
        Probability value(s) returned by ``predict_proba``.
    inference_delay_ms : float
        Artificial delay per ``invoke()`` call.
    """

    def __init__(
        self,
        fake_output: float = 0.95,
        inference_delay_ms: float = 10.0,
    ) -> None:
        from .mock_tflite_interpreter import MockInterpreter

        self.interp = MockInterpreter(
            fake_output=fake_output,
            inference_delay_ms=inference_delay_ms,
        )
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

    def expected_input_shape(self) -> np.ndarray:
        return self.in_det.get("shape", None)

    def predict_proba(self, x: np.ndarray) -> float:
        """Run mock inference and return the fake probability."""
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)

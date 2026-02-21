from pathlib import Path

import numpy as np
import librosa

from .config import MODEL_SR, CLIP_SECONDS, N_MELS, N_FFT, HOP, CENTER, TARGET_RMS


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def logmel(y_16k_2s: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram. Returns shape (frames, n_mels)."""
    S = librosa.feature.melspectrogram(
        y=y_16k_2s,
        sr=MODEL_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
        center=CENTER,  # IMPORTANT: must match training
    )
    return np.log(S + 1e-6).T.astype(np.float32)


def load_stats(stats_path: Path):
    """Load µ and σ from an .npz file. Returns (mu, sdv) each of shape (n_mels,)."""
    st = np.load(str(stats_path), allow_pickle=True)
    mu  = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    # Handle multiple stored shapes: (64,), (1,1,64), (1,frames,64)
    for arr in [mu, sdv]:
        pass  # reshape applied below

    def _flatten(a: np.ndarray) -> np.ndarray:
        if a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == 1:
            a = a.reshape(-1)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        return a.reshape(-1).astype(np.float32)

    return _flatten(mu), _flatten(sdv)


def preproc(y16: np.ndarray, mu: np.ndarray, sdv: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline.

    Args:
        y16:  16 kHz mono audio array.
        mu:   Per-mel mean, shape (n_mels,).
        sdv:  Per-mel std-dev, shape (n_mels,).

    Returns:
        Float32 array of shape (1, frames, n_mels, 1) ready for TFLite input.
    """
    target = int(MODEL_SR * CLIP_SECONDS)
    y16 = np.asarray(y16, dtype=np.float32)

    if len(y16) > target:
        y16 = y16[:target]
    elif len(y16) < target:
        y16 = np.pad(y16, (0, target - len(y16)))

    y16 = normalize_rms(y16)
    f   = logmel(y16)                              # (frames, n_mels)
    fn  = (f - mu[None, :]) / (sdv[None, :] + 1e-6)
    return fn[None, :, :, None].astype(np.float32)  # (1, frames, n_mels, 1)


def resample_to_model_sr(y48: np.ndarray, capture_sr: int) -> np.ndarray:
    """Resample captured audio to MODEL_SR (16 kHz)."""
    return librosa.resample(y48, orig_sr=capture_sr, target_sr=MODEL_SR).astype(np.float32)

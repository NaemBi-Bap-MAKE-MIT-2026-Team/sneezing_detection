"""
Audio Preprocessing Module

This module provides audio preprocessing functions that MUST match
the training pipeline to ensure model accuracy.

Preprocessing steps:
    1. RMS normalization (target = 0.1)
    2. Pre-emphasis filter (α = 0.97)
    3. Silence trimming (top_db = 20)
    4. Length preservation (pad to original length)
"""

import numpy as np
import librosa
from utils.config import TARGET_RMS, PRE_EMPHASIS, TRIM_DB


class PreprocessingModule:
    """
    Audio preprocessing module for real-time sneeze detection

    This class wraps preprocessing functions to ensure consistency
    with the training pipeline.
    """

    def __init__(self, target_rms=TARGET_RMS, pre_emphasis=PRE_EMPHASIS, trim_db=TRIM_DB):
        """
        Initialize preprocessing module

        Args:
            target_rms (float): Target RMS level after normalization (default: 0.1)
            pre_emphasis (float): Pre-emphasis filter coefficient (default: 0.97)
            trim_db (int): dB threshold for silence trimming (default: 20)
        """
        self.target_rms = target_rms
        self.pre_emphasis = pre_emphasis
        self.trim_db = trim_db

    def process(self, audio, sr=16000):
        """
        Apply full preprocessing pipeline

        Args:
            audio (np.ndarray): Raw audio array (1D)
            sr (int): Sample rate (default: 16000)

        Returns:
            np.ndarray: Preprocessed audio array (same length as input)
        """
        # 1. RMS normalization
        audio = self.normalize_rms(audio)

        # 2. Pre-emphasis filter
        audio = self.apply_pre_emphasis(audio)

        # 3. Trim silence and maintain original length
        audio = self.trim_silence(audio, sr)

        return audio

    def normalize_rms(self, audio):
        """
        Normalize audio to target RMS level

        Args:
            audio (np.ndarray): Input audio

        Returns:
            np.ndarray: RMS-normalized audio
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * self.target_rms
        return audio

    def apply_pre_emphasis(self, audio):
        """
        Apply pre-emphasis filter to boost high frequencies

        Formula: y[n] = x[n] - α * x[n-1]

        Args:
            audio (np.ndarray): Input audio

        Returns:
            np.ndarray: Pre-emphasized audio
        """
        emphasized = np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])
        return emphasized

    def trim_silence(self, audio, sr):
        """
        Trim silence from audio and maintain original length

        Args:
            audio (np.ndarray): Input audio
            sr (int): Sample rate

        Returns:
            np.ndarray: Trimmed audio (same length as input, padded if necessary)
        """
        original_length = len(audio)

        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_db)

        # Maintain original length
        if len(audio_trimmed) < original_length:
            # Pad with zeros
            pad_length = original_length - len(audio_trimmed)
            audio_trimmed = np.pad(audio_trimmed, (0, pad_length), mode='constant')
        elif len(audio_trimmed) > original_length:
            # Truncate
            audio_trimmed = audio_trimmed[:original_length]

        return audio_trimmed


# Standalone function for backward compatibility
def preprocess_audio(audio, sr=16000, target_rms=TARGET_RMS,
                    pre_emphasis=PRE_EMPHASIS, trim_db=TRIM_DB):
    """
    Standalone preprocessing function (matches training pipeline)

    This function provides the exact preprocessing used during training.

    Args:
        audio (np.ndarray): Raw audio array
        sr (int): Sample rate
        target_rms (float): Target RMS level
        pre_emphasis (float): Pre-emphasis coefficient
        trim_db (int): Silence trimming threshold

    Returns:
        np.ndarray: Preprocessed audio
    """
    # 1. RMS normalization
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio / rms * target_rms

    # 2. Pre-emphasis filter
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # 3. Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=trim_db)

    # 4. Maintain original length
    if len(audio_trimmed) < len(audio):
        pad_length = len(audio) - len(audio_trimmed)
        audio_trimmed = np.pad(audio_trimmed, (0, pad_length), mode='constant')
    elif len(audio_trimmed) > len(audio):
        audio_trimmed = audio_trimmed[:len(audio)]

    return audio_trimmed


if __name__ == "__main__":
    # Test preprocessing module
    preprocessor = PreprocessingModule()

    # Generate dummy audio
    dummy_audio = np.random.randn(32000).astype(np.float32)

    # Preprocess
    preprocessed = preprocessor.process(dummy_audio)

    print(f"Input shape: {dummy_audio.shape}")
    print(f"Output shape: {preprocessed.shape}")
    print(f"Input RMS: {np.sqrt(np.mean(dummy_audio**2)):.6f}")
    print(f"Output RMS: {np.sqrt(np.mean(preprocessed**2)):.6f}")
    print(f"Target RMS: {TARGET_RMS}")

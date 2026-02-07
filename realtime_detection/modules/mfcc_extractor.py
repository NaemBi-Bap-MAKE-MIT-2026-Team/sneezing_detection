"""
MFCC Feature Extraction Module

This module extracts MFCC (Mel-Frequency Cepstral Coefficients) features
with Delta and Delta-Delta features for audio classification.

MUST match training parameters for accurate inference.
"""

import numpy as np
import librosa
from utils.config import N_MFCC, N_FFT, HOP_LENGTH, WINDOW_TYPE, INCLUDE_DELTAS


class MFCCExtractorModule:
    """
    MFCC feature extractor for real-time sneeze detection

    Extracts MFCC features along with first-order (Delta) and
    second-order (Delta-Delta) derivatives for temporal information.
    """

    def __init__(self, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
                 window=WINDOW_TYPE, include_deltas=INCLUDE_DELTAS):
        """
        Initialize MFCC extractor

        Args:
            n_mfcc (int): Number of MFCC coefficients (default: 20)
            n_fft (int): FFT window size (default: 2048)
            hop_length (int): Number of samples between frames (default: 512)
            window (str): Window function type (default: 'hann')
            include_deltas (bool): Include Delta and Delta-Delta features (default: True)
        """
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.include_deltas = include_deltas

    def extract(self, audio, sr=16000):
        """
        Extract MFCC features from audio

        Args:
            audio (np.ndarray): Preprocessed audio array (1D)
            sr (int): Sample rate (default: 16000)

        Returns:
            np.ndarray: MFCC features
                - Shape (20, time_frames) if include_deltas=False
                - Shape (60, time_frames) if include_deltas=True
                  (20 MFCC + 20 Delta + 20 Delta-Delta)

        Expected output shape for 2-second audio at 16kHz: (60, 63)
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window
        )

        # Mean normalization (normalize each coefficient independently)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        if self.include_deltas:
            # Delta features (first-order derivative - rate of change)
            delta_mfcc = librosa.feature.delta(mfcc)

            # Delta-Delta features (second-order derivative - acceleration)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # Stack all features vertically
            # Shape: (60, time_frames) = (20+20+20, time_frames)
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        else:
            features = mfcc

        return features

    def get_expected_shape(self, audio_length, sr=16000):
        """
        Calculate expected MFCC output shape

        Args:
            audio_length (int): Audio length in samples
            sr (int): Sample rate

        Returns:
            tuple: Expected (features, time_frames) shape
        """
        # Calculate number of time frames
        time_frames = 1 + (audio_length - self.n_fft) // self.hop_length

        # Calculate number of features
        if self.include_deltas:
            num_features = self.n_mfcc * 3  # MFCC + Delta + Delta-Delta
        else:
            num_features = self.n_mfcc

        return (num_features, time_frames)


# Standalone function for backward compatibility
def extract_mfcc_features(audio, sr=16000, n_mfcc=N_MFCC, n_fft=N_FFT,
                         hop_length=HOP_LENGTH, include_deltas=INCLUDE_DELTAS):
    """
    Standalone MFCC extraction function (matches training pipeline)

    Args:
        audio (np.ndarray): Preprocessed audio array
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        n_fft (int): FFT window size
        hop_length (int): Frame stride
        include_deltas (bool): Include Delta and Delta-Delta

    Returns:
        np.ndarray: MFCC features with shape (60, time_frames) if include_deltas=True
    """
    # MFCC extraction
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hann'
    )

    # Mean normalization
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    if include_deltas:
        # Delta features (1st derivative)
        delta_mfcc = librosa.feature.delta(mfcc)

        # Delta-Delta features (2nd derivative)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # Stack: (20 + 20 + 20, time_frames) = (60, time_frames)
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    else:
        features = mfcc

    return features


if __name__ == "__main__":
    # Test MFCC extractor
    extractor = MFCCExtractorModule()

    # Generate dummy audio (2 seconds at 16kHz)
    dummy_audio = np.random.randn(32000).astype(np.float32)

    # Extract MFCC
    mfcc_features = extractor.extract(dummy_audio)

    # Expected shape
    expected_shape = extractor.get_expected_shape(len(dummy_audio))

    print(f"Input audio shape: {dummy_audio.shape}")
    print(f"MFCC features shape: {mfcc_features.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Shape matches: {mfcc_features.shape == expected_shape}")

    # Feature breakdown
    print(f"\nFeature breakdown:")
    print(f"  MFCC coefficients: {extractor.n_mfcc}")
    print(f"  Delta features: {extractor.n_mfcc if extractor.include_deltas else 0}")
    print(f"  Delta-Delta features: {extractor.n_mfcc if extractor.include_deltas else 0}")
    print(f"  Total features: {mfcc_features.shape[0]}")
    print(f"  Time frames: {mfcc_features.shape[1]}")

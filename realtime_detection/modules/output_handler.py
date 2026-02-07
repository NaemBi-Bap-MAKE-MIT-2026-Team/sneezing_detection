"""
Output Handler Module

This module handles detection results including:
- Saving detected audio clips
- Logging detections to CSV
- Managing cooldown periods
- User callbacks
"""

import numpy as np
import soundfile as sf
import csv
import time
from pathlib import Path
from datetime import datetime

from utils.config import SAVE_DIR, COOLDOWN_SECONDS, ENABLE_LOGGING, LOG_FILE, THRESHOLD


class OutputHandlerModule:
    """
    Output handler for sneeze detection results

    Manages saving of detected audio clips, logging, and cooldown periods
    to prevent duplicate detections.
    """

    def __init__(self, save_dir=SAVE_DIR, threshold=THRESHOLD,
                 cooldown=COOLDOWN_SECONDS, enable_logging=ENABLE_LOGGING):
        """
        Initialize output handler

        Args:
            save_dir (str): Directory for saving detected audio clips
            threshold (float): Detection threshold (default: 0.8)
            cooldown (float): Cooldown period in seconds (default: 1.0)
            enable_logging (bool): Enable CSV logging (default: True)
        """
        self.save_dir = Path(save_dir)
        self.threshold = threshold
        self.cooldown = cooldown
        self.enable_logging = enable_logging

        # Create save directory
        self.save_dir.mkdir(exist_ok=True)

        # CSV log file path
        self.log_file = self.save_dir / LOG_FILE

        # Initialize CSV log if enabled
        if self.enable_logging:
            self._initialize_log()

        # Cooldown tracking
        self.last_detection_time = 0

        # Statistics
        self.detection_count = 0
        self.total_processed = 0

        # User callbacks
        self.callbacks = []

        print(f"📁 Output handler initialized")
        print(f"  Save directory: {self.save_dir}")
        print(f"  Logging: {'enabled' if self.enable_logging else 'disabled'}")
        print(f"  Cooldown: {self.cooldown}s")

    def _initialize_log(self):
        """
        Initialize CSV log file with headers
        """
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'probability',
                    'is_sneeze',
                    'filename',
                    'detection_number'
                ])
            print(f"  Log file created: {self.log_file}")

    def handle_detection(self, is_sneeze, probability, audio_chunk, sr=16000):
        """
        Handle detection result

        Args:
            is_sneeze (bool): Whether sneeze was detected
            probability (float): Sneeze probability
            audio_chunk (np.ndarray): Audio chunk
            sr (int): Sample rate (default: 16000)

        Returns:
            bool: True if detection was handled (not in cooldown)
        """
        self.total_processed += 1
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_detection_time < self.cooldown:
            return False

        # Handle sneeze detection
        if is_sneeze:
            self.detection_count += 1

            # Save audio
            filename = self._save_audio(audio_chunk, probability, sr)

            # Log detection
            if self.enable_logging:
                self._log_detection(probability, is_sneeze, filename)

            # Update last detection time
            self.last_detection_time = current_time

            # Print notification
            print(f"🤧 SNEEZE DETECTED! Probability: {probability:.4f}")
            print(f"   Saved: {filename}")
            print(f"   Detection #{self.detection_count}")

            # Call user callbacks
            for callback in self.callbacks:
                try:
                    callback(is_sneeze, probability, audio_chunk, filename)
                except Exception as e:
                    print(f"⚠️ Callback error: {e}")

            return True

        return False

    def _save_audio(self, audio_chunk, probability, sr):
        """
        Save audio chunk to file

        Args:
            audio_chunk (np.ndarray): Audio data
            probability (float): Sneeze probability
            sr (int): Sample rate

        Returns:
            str: Saved filename
        """
        # Generate filename with timestamp and probability
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_sneeze_prob{probability:.2f}.wav"
        filepath = self.save_dir / filename

        # Save audio
        sf.write(filepath, audio_chunk, sr)

        return filename

    def _log_detection(self, probability, is_sneeze, filename):
        """
        Log detection to CSV file

        Args:
            probability (float): Sneeze probability
            is_sneeze (bool): Whether sneeze was detected
            filename (str): Saved audio filename
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                f"{probability:.4f}",
                'Yes' if is_sneeze else 'No',
                filename,
                self.detection_count
            ])

    def register_callback(self, callback_fn):
        """
        Register a user callback function

        The callback will be called with: callback(is_sneeze, probability, audio_chunk, filename)

        Args:
            callback_fn: Callback function
        """
        self.callbacks.append(callback_fn)
        print(f"✓ Callback registered: {callback_fn.__name__}")

    def get_statistics(self):
        """
        Get detection statistics

        Returns:
            dict: Statistics dictionary
        """
        return {
            'total_processed': self.total_processed,
            'detections': self.detection_count,
            'detection_rate': self.detection_count / self.total_processed if self.total_processed > 0 else 0,
            'save_dir': str(self.save_dir),
            'log_file': str(self.log_file) if self.enable_logging else None
        }

    def print_statistics(self):
        """
        Print detection statistics
        """
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        print("Detection Statistics")
        print("=" * 70)
        print(f"Total windows processed: {stats['total_processed']}")
        print(f"Sneezes detected: {stats['detections']}")
        print(f"Detection rate: {stats['detection_rate']*100:.2f}%")
        print(f"Save directory: {stats['save_dir']}")
        if stats['log_file']:
            print(f"Log file: {stats['log_file']}")
        print("=" * 70)


if __name__ == "__main__":
    print("Testing OutputHandlerModule...")
    print("=" * 70)

    # Create output handler
    handler = OutputHandlerModule()

    # Test callback
    def my_callback(is_sneeze, probability, audio_chunk, filename):
        print(f"  Callback called: {filename}")

    handler.register_callback(my_callback)

    # Generate dummy audio
    dummy_audio = np.random.randn(32000).astype(np.float32)

    # Test detection handling
    print("\n" + "=" * 70)
    print("Test 1: Sneeze detection (above threshold)")
    print("=" * 70)
    handler.handle_detection(True, 0.95, dummy_audio)

    # Test cooldown
    print("\n" + "=" * 70)
    print("Test 2: Detection during cooldown (should be ignored)")
    print("=" * 70)
    result = handler.handle_detection(True, 0.92, dummy_audio)
    print(f"  Handled: {result}")

    # Wait for cooldown
    print(f"\nWaiting {COOLDOWN_SECONDS}s for cooldown...")
    time.sleep(COOLDOWN_SECONDS + 0.1)

    # Test after cooldown
    print("\n" + "=" * 70)
    print("Test 3: Detection after cooldown")
    print("=" * 70)
    handler.handle_detection(True, 0.88, dummy_audio)

    # Test non-sneeze
    print("\n" + "=" * 70)
    print("Test 4: Non-sneeze (below threshold)")
    print("=" * 70)
    handler.handle_detection(False, 0.45, dummy_audio)

    # Print statistics
    handler.print_statistics()

    print("\n✓ Output handler test passed!")

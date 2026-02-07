#!/usr/bin/env python3
"""
Real-time Sneeze Detection System

This is the main entry point for the real-time sneeze detection system.
It orchestrates all modules to capture audio, process it, and detect sneezes.

Usage:
    python main.py [--verbose] [--threshold 0.8]

Author: Claude & Bahk Insung
Date: 2026-02-07
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from modules.audio_capture import AudioCaptureModule
from modules.preprocessing import PreprocessingModule
from modules.mfcc_extractor import MFCCExtractorModule
from modules.model_inference import ModelInferenceModule
from modules.output_handler import OutputHandlerModule

from utils.config import (
    SAMPLE_RATE, WINDOW_SIZE, THRESHOLD, VERBOSE, PRINT_INTERVAL
)


class RealtimeSneezeDetector:
    """
    Real-time Sneeze Detection System

    Orchestrates all modules for end-to-end real-time sneeze detection.
    """

    def __init__(self, model_path=None, threshold=None, verbose=None):
        """
        Initialize real-time sneeze detector

        Args:
            model_path (str): Path to trained model (default: from config)
            threshold (float): Detection threshold (default: from config)
            verbose (bool): Verbose output (default: from config)
        """
        self.threshold = threshold if threshold is not None else THRESHOLD
        self.verbose = verbose if verbose is not None else VERBOSE

        print("=" * 70)
        print("🎤 Real-time Sneeze Detection System")
        print("=" * 70)

        # Initialize modules
        print("\nInitializing modules...")

        try:
            # 1. Audio Capture
            self.audio_capture = AudioCaptureModule()

            # 2. Preprocessing
            self.preprocessor = PreprocessingModule()
            print("✓ Preprocessing module loaded")

            # 3. MFCC Extractor
            self.mfcc_extractor = MFCCExtractorModule()
            print("✓ MFCC extractor loaded")

            # 4. Model Inference
            if model_path:
                from utils.config import MODEL_PATH
                import utils.config as config
                config.MODEL_PATH = model_path
            self.model_inference = ModelInferenceModule(threshold=self.threshold)

            # 5. Output Handler
            self.output_handler = OutputHandlerModule(threshold=self.threshold)

            print("\n✓ All modules initialized successfully!")

        except Exception as e:
            print(f"\n❌ Initialization failed: {e}")
            raise

        # Statistics
        self.window_count = 0
        self.start_time = None

    def start(self):
        """
        Start real-time detection

        Main detection loop that:
        1. Captures audio from microphone
        2. Preprocesses audio
        3. Extracts MFCC features
        4. Performs inference
        5. Handles detection output
        """
        print("\n" + "=" * 70)
        print("Starting detection...")
        print("=" * 70)
        print(f"🎯 Detection threshold: {self.threshold}")
        print(f"⏱️  Window size: {WINDOW_SIZE / SAMPLE_RATE:.1f} seconds")
        print("\nPress Ctrl+C to stop\n")

        try:
            # Start audio capture
            self.audio_capture.start_capture()
            self.start_time = time.time()

            # Wait for buffer to fill
            print("Waiting for audio buffer to fill...")
            while not self.audio_capture.is_buffer_ready():
                time.sleep(0.1)

            print("✓ Buffer ready. Starting detection...\n")
            print("=" * 70)

            # Main detection loop
            while True:
                # Extract audio chunk
                audio_chunk = self.audio_capture.get_audio_chunk()

                if audio_chunk is None:
                    time.sleep(0.1)
                    continue

                # Process audio
                try:
                    # 1. Preprocess
                    start_time = time.time()
                    preprocessed = self.preprocessor.process(audio_chunk)
                    preprocess_time = time.time() - start_time

                    # 2. Extract MFCC
                    start_time = time.time()
                    mfcc_features = self.mfcc_extractor.extract(preprocessed)
                    mfcc_time = time.time() - start_time

                    # 3. Model inference
                    start_time = time.time()
                    is_sneeze, probability = self.model_inference.predict(mfcc_features)
                    inference_time = time.time() - start_time

                    # 4. Handle output
                    self.output_handler.handle_detection(
                        is_sneeze, probability, audio_chunk, SAMPLE_RATE
                    )

                    # Update statistics
                    self.window_count += 1

                    # Print status
                    if self.verbose:
                        self._print_status(probability, is_sneeze, preprocess_time,
                                         mfcc_time, inference_time)

                except Exception as e:
                    print(f"⚠️ Processing error: {e}")
                    continue

                # Sleep to reduce CPU usage
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping detection...")

        except Exception as e:
            print(f"\n❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._cleanup()

    def _print_status(self, probability, is_sneeze, preprocess_time, mfcc_time, inference_time):
        """
        Print detection status

        Args:
            probability (float): Sneeze probability
            is_sneeze (bool): Whether sneeze was detected
            preprocess_time (float): Preprocessing time in seconds
            mfcc_time (float): MFCC extraction time in seconds
            inference_time (float): Inference time in seconds
        """
        if is_sneeze:
            # Don't print here, OutputHandler already printed
            pass
        elif self.window_count % PRINT_INTERVAL == 0 or probability > 0.5:
            total_time = preprocess_time + mfcc_time + inference_time
            print(f"[{self.window_count:5d}] "
                  f"⚪ Probability: {probability:.4f} "
                  f"| Processing: {total_time*1000:.1f}ms "
                  f"(prep: {preprocess_time*1000:.1f}ms, "
                  f"mfcc: {mfcc_time*1000:.1f}ms, "
                  f"infer: {inference_time*1000:.1f}ms)")

    def _cleanup(self):
        """
        Cleanup resources
        """
        # Stop audio capture
        if hasattr(self, 'audio_capture'):
            self.audio_capture.stop_capture()

        # Print final statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            print("\n" + "=" * 70)
            print("Session Statistics")
            print("=" * 70)
            print(f"Total time: {elapsed:.1f}s")
            print(f"Windows processed: {self.window_count}")
            print(f"Processing rate: {self.window_count / elapsed:.2f} windows/s")

        # Print output handler statistics
        if hasattr(self, 'output_handler'):
            self.output_handler.print_statistics()

        print("\n✓ Cleanup complete")


def main():
    """
    Main entry point
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Real-time Sneeze Detection System'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model file (.pth)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help=f'Detection threshold (default: {THRESHOLD})'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable verbose output'
    )

    args = parser.parse_args()

    # Determine verbosity
    if args.quiet:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        verbose = VERBOSE

    # Create detector
    detector = RealtimeSneezeDetector(
        model_path=args.model,
        threshold=args.threshold,
        verbose=verbose
    )

    # Start detection
    detector.start()


if __name__ == "__main__":
    main()

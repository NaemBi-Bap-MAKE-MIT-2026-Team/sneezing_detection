"""
Audio Capture Module

This module captures live audio from the microphone using PyAudio
and manages a circular buffer for real-time processing.
"""

import numpy as np
import pyaudio
from collections import deque
import threading

from utils.config import SAMPLE_RATE, CHUNK_SIZE, WINDOW_SIZE, BUFFER_SIZE


class AudioCaptureModule:
    """
    Audio capture module for real-time microphone input

    Uses PyAudio callback mode for non-blocking audio capture
    with a circular buffer for efficient windowing.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE,
                 window_size=WINDOW_SIZE, buffer_size=BUFFER_SIZE):
        """
        Initialize audio capture module

        Args:
            sample_rate (int): Audio sample rate in Hz (default: 16000)
            chunk_size (int): PyAudio buffer size in samples (default: 1024)
            window_size (int): Analysis window size in samples (default: 32000)
            buffer_size (int): Circular buffer size in samples (default: 64000)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.buffer_size = buffer_size

        # Circular buffer for audio samples
        # maxlen ensures automatic overflow handling
        self.audio_buffer = deque(maxlen=buffer_size)

        # PyAudio objects
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Threading lock for buffer access
        self.buffer_lock = threading.Lock()

        # Status flags
        self.is_capturing = False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio input

        This function is called by PyAudio in a separate thread
        whenever new audio data is available.

        Args:
            in_data: Audio data from microphone
            frame_count: Number of frames
            time_info: Time information
            status: Stream status flags

        Returns:
            tuple: (None, pyaudio.paContinue)
        """
        if status:
            print(f"⚠️ Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        # Add to buffer (thread-safe)
        with self.buffer_lock:
            self.audio_buffer.extend(audio_data)

        return (None, pyaudio.paContinue)

    def start_capture(self):
        """
        Start capturing audio from microphone

        Raises:
            RuntimeError: If capture is already running
            OSError: If microphone device not found
        """
        if self.is_capturing:
            raise RuntimeError("Audio capture is already running")

        try:
            # Open audio stream with callback
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,  # Mono
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback,
                start=False  # Don't start immediately
            )

            # Start stream
            self.stream.start_stream()
            self.is_capturing = True

            print("🎤 Audio capture started")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sample_rate*1000:.1f} ms)")
            print(f"  Window size: {self.window_size} samples ({self.window_size/self.sample_rate:.1f} s)")
            print(f"  Buffer size: {self.buffer_size} samples ({self.buffer_size/self.sample_rate:.1f} s)")

        except OSError as e:
            print(f"❌ Error opening audio device: {e}")
            print("\nAvailable audio devices:")
            self.list_audio_devices()
            raise

    def stop_capture(self):
        """
        Stop capturing audio from microphone
        """
        if not self.is_capturing:
            return

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.is_capturing = False
        print("🛑 Audio capture stopped")

    def get_audio_chunk(self, window_size=None):
        """
        Extract audio chunk from buffer

        Args:
            window_size (int): Size of chunk to extract (default: self.window_size)

        Returns:
            np.ndarray: Audio chunk or None if buffer not ready
        """
        if window_size is None:
            window_size = self.window_size

        with self.buffer_lock:
            if len(self.audio_buffer) < window_size:
                return None

            # Extract last window_size samples
            chunk = np.array(list(self.audio_buffer)[-window_size:])

        return chunk

    def is_buffer_ready(self, window_size=None):
        """
        Check if buffer has enough samples for extraction

        Args:
            window_size (int): Required window size (default: self.window_size)

        Returns:
            bool: True if buffer is ready
        """
        if window_size is None:
            window_size = self.window_size

        with self.buffer_lock:
            return len(self.audio_buffer) >= window_size

    def clear_buffer(self):
        """
        Clear the audio buffer
        """
        with self.buffer_lock:
            self.audio_buffer.clear()
        print("🗑️ Audio buffer cleared")

    def get_buffer_status(self):
        """
        Get current buffer status

        Returns:
            dict: Buffer status information
        """
        with self.buffer_lock:
            buffer_length = len(self.audio_buffer)

        return {
            "buffer_samples": buffer_length,
            "buffer_seconds": buffer_length / self.sample_rate,
            "buffer_fill_percent": (buffer_length / self.buffer_size) * 100,
            "is_ready": buffer_length >= self.window_size,
            "is_capturing": self.is_capturing
        }

    def list_audio_devices(self):
        """
        List all available audio devices
        """
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')

        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device_info.get('maxInputChannels') > 0:
                print(f"  [{i}] {device_info.get('name')} "
                      f"(Channels: {device_info.get('maxInputChannels')}, "
                      f"Sample Rate: {int(device_info.get('defaultSampleRate'))} Hz)")

    def __del__(self):
        """
        Cleanup on deletion
        """
        self.stop_capture()
        if hasattr(self, 'audio'):
            self.audio.terminate()


if __name__ == "__main__":
    import time

    print("Testing AudioCaptureModule...")
    print("=" * 70)

    # Create audio capture module
    capture = AudioCaptureModule()

    print("\nAvailable audio devices:")
    capture.list_audio_devices()

    try:
        # Start capture
        print("\n" + "=" * 70)
        capture.start_capture()

        # Wait for buffer to fill
        print("\nWaiting for buffer to fill...")
        while not capture.is_buffer_ready():
            status = capture.get_buffer_status()
            print(f"  Buffer: {status['buffer_fill_percent']:.1f}% "
                  f"({status['buffer_seconds']:.2f}s / {WINDOW_SIZE/SAMPLE_RATE:.2f}s)")
            time.sleep(0.5)

        print("\n✓ Buffer ready!")

        # Extract a few chunks
        print("\nExtracting audio chunks...")
        for i in range(3):
            chunk = capture.get_audio_chunk()
            if chunk is not None:
                print(f"  Chunk {i+1}: {chunk.shape}, "
                      f"RMS: {np.sqrt(np.mean(chunk**2)):.6f}")
            time.sleep(1)

        print("\n✓ Audio capture test passed!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop capture
        print("\n" + "=" * 70)
        capture.stop_capture()

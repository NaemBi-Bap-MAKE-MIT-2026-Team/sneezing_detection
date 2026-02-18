#!/usr/bin/env python3
"""
RPi4 Sneeze-Detection Simulation Runner
========================================
Runs the v4 TFLite sneeze-detection pipeline against real ``.wav`` files
while simulating RPi4 resource constraints (CPU throttling, memory
monitoring, timing profiling).

No real TFLite model is required — a ``MockInterpreter`` returns
configurable fake probabilities.

Usage
-----
::

    # Run with default settings
    python -m legacy_code.simulation.run_simulation --wav-dir /path/to/wavs/

    # Customise
    python -m legacy_code.simulation.run_simulation \\
        --wav-dir /path/to/wavs/ \\
        --fake-output 0.95 \\
        --rpi4-ram 2048 \\
        --budget-ms 1000 \\
        --repeat 5 \\
        --no-throttle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import librosa

from .v4_pipeline import (
    CAPTURE_SR,
    MODEL_SR,
    CLIP_SECONDS,
    TARGET_SAMPLES_16K,
    TARGET_SAMPLES_48K,
    PROB_TH,
    rms,
    resample_48k_to_16k,
    preproc,
    create_mock_stats,
    load_stats,
    MockLiteModel,
)
from .rpi4_resource_simulator import CPUThrottler, MemoryMonitor, TimingProfiler


# ===================================================================== #
#  WAV file loading
# ===================================================================== #

def load_wav(path: Path) -> np.ndarray:
    """Load a ``.wav`` file and return 48 kHz mono float32 audio.

    If the file has a different sample rate it is resampled to 48 kHz.
    Multi-channel files are mixed down to mono.
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != CAPTURE_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CAPTURE_SR)
    y = y.astype(np.float32)

    # Pad or trim to exactly CLIP_SECONDS at 48 kHz
    if len(y) > TARGET_SAMPLES_48K:
        y = y[:TARGET_SAMPLES_48K]
    elif len(y) < TARGET_SAMPLES_48K:
        y = np.pad(y, (0, TARGET_SAMPLES_48K - len(y)))

    return y


def discover_wavs(wav_dir: Path) -> List[Path]:
    """Return sorted list of ``.wav`` files in *wav_dir*."""
    if not wav_dir.is_dir():
        print(f"[ERROR] --wav-dir is not a directory: {wav_dir}", file=sys.stderr)
        sys.exit(1)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        print(f"[ERROR] No .wav files found in: {wav_dir}", file=sys.stderr)
        sys.exit(1)
    return wavs


# ===================================================================== #
#  Single-file simulation
# ===================================================================== #

def simulate_one(
    wav_path: Path,
    model: MockLiteModel,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    mem: MemoryMonitor,
) -> dict:
    """Run the full v4 pipeline on one wav file and return results."""

    profiler.reset()
    result = {"file": wav_path.name}

    # 1. Load wav file
    with profiler.measure("file_io"):
        with throttler.throttle("file_io"):
            y48 = load_wav(wav_path)
    mem.snapshot("after_load")

    result["rms_48k"] = rms(y48)

    # 2. Resample 48 kHz -> 16 kHz
    with profiler.measure("resample"):
        with throttler.throttle("resample"):
            y16 = resample_48k_to_16k(y48)
    mem.snapshot("after_resample")

    # 3. Pad/trim to exact length
    with profiler.measure("pad_trim"):
        with throttler.throttle("pad_trim"):
            if len(y16) > TARGET_SAMPLES_16K:
                y16 = y16[:TARGET_SAMPLES_16K]
            elif len(y16) < TARGET_SAMPLES_16K:
                y16 = np.pad(y16, (0, TARGET_SAMPLES_16K - len(y16)))
    mem.snapshot("after_pad_trim")

    # 4. RMS normalise
    with profiler.measure("normalize"):
        with throttler.throttle("normalize"):
            from .v4_pipeline import normalize_rms
            y16_norm = normalize_rms(y16)
    mem.snapshot("after_normalize")

    # 5. Log-mel spectrogram
    with profiler.measure("melspectrogram"):
        with throttler.throttle("melspectrogram"):
            from .v4_pipeline import logmel
            f = logmel(y16_norm)
    mem.snapshot("after_logmel")

    result["logmel_shape"] = f.shape

    # 6. Z-score normalise + reshape
    with profiler.measure("normalize"):
        with throttler.throttle("normalize"):
            fn = (f - mu[None, :]) / (sdv[None, :] + 1e-6)
            x_in = fn[None, :, :, None].astype(np.float32)
    mem.snapshot("after_zscore")

    result["input_shape"] = x_in.shape

    # 7. Mock TFLite inference
    with profiler.measure("tflite_invoke"):
        with throttler.throttle("tflite_invoke"):
            prob = model.predict_proba(x_in)
    mem.snapshot("after_inference")

    result["probability"] = prob
    result["detected"] = prob >= PROB_TH
    result["profiler_summary"] = profiler.summary_table()
    result["total_desktop_ms"] = profiler.total_desktop_ms()
    result["total_rpi4_ms"] = profiler.total_rpi4_ms()
    result["within_budget"] = profiler.within_budget()

    return result


# ===================================================================== #
#  Report formatting
# ===================================================================== #

def print_file_report(result: dict, idx: int, total: int) -> None:
    """Print a formatted report for one simulation run."""
    header = f"  [{idx}/{total}] {result['file']}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  RMS (48 kHz input) : {result['rms_48k']:.6f}")
    print(f"  Log-mel shape      : {result['logmel_shape']}")
    print(f"  Model input shape  : {result['input_shape']}")
    print(f"  Probability        : {result['probability']:.4f}")
    print(f"  Detected (>={PROB_TH})  : {'YES' if result['detected'] else 'NO'}")
    print()

    # Indent the profiler summary
    for line in result["profiler_summary"].splitlines():
        print(f"  {line}")
    print()


def print_aggregate_report(
    all_results: List[dict],
    mem: MemoryMonitor,
    budget_ms: float,
) -> None:
    """Print the final aggregate summary."""
    n = len(all_results)
    desk_times = [r["total_desktop_ms"] for r in all_results]
    rpi4_times = [r["total_rpi4_ms"] for r in all_results]
    passes = sum(1 for r in all_results if r["within_budget"])

    print("=" * 70)
    print("  AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"  Files processed    : {n}")
    print(f"  Budget             : {budget_ms:.0f} ms")
    print(f"  Budget pass rate   : {passes}/{n} ({passes/n*100:.0f}%)")
    print()
    print(f"  Desktop time (avg) : {np.mean(desk_times):.2f} ms")
    print(f"  Desktop time (max) : {np.max(desk_times):.2f} ms")
    print(f"  RPi4 est.    (avg) : {np.mean(rpi4_times):.2f} ms")
    print(f"  RPi4 est.    (max) : {np.max(rpi4_times):.2f} ms")
    print()

    # Detection summary
    detections = sum(1 for r in all_results if r["detected"])
    print(f"  Detections         : {detections}/{n}")
    print()

    # Memory summary
    print("  --- Memory ---")
    print(f"  {mem.summary_table()}")
    print()

    # Final viability verdict
    max_rpi4 = np.max(rpi4_times)
    if max_rpi4 <= budget_ms:
        verdict = "VIABLE"
        detail = f"worst-case {max_rpi4:.0f} ms <= {budget_ms:.0f} ms budget"
    else:
        verdict = "NOT VIABLE"
        detail = f"worst-case {max_rpi4:.0f} ms > {budget_ms:.0f} ms budget"

    print(f"  VERDICT: {verdict}  ({detail})")
    print("=" * 70)


# ===================================================================== #
#  CLI entry point
# ===================================================================== #

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RPi4 sneeze-detection simulation (v4 TFLite pipeline)",
    )
    p.add_argument(
        "--wav-dir", type=Path, required=True,
        help="Directory containing .wav test files",
    )
    p.add_argument(
        "--stats-path", type=Path, default=None,
        help="Path to v4_norm_stats.npz (omit to use mock stats: mu=0, sd=1)",
    )
    p.add_argument(
        "--fake-output", type=float, default=0.95,
        help="Mock model probability output (default: 0.95)",
    )
    p.add_argument(
        "--inference-delay-ms", type=float, default=10.0,
        help="Simulated TFLite invoke() delay in ms (default: 10.0)",
    )
    p.add_argument(
        "--rpi4-ram", type=int, default=2048,
        help="RPi4 total RAM in MB (default: 2048)",
    )
    p.add_argument(
        "--budget-ms", type=float, default=1000.0,
        help="Per-inference time budget in ms (default: 1000.0)",
    )
    p.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to repeat each file (default: 1)",
    )
    p.add_argument(
        "--no-throttle", action="store_true",
        help="Disable CPU throttling (desktop-speed only, still records timing)",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Setup ---
    print("=" * 70)
    print("  RPi4 Sneeze Detection Simulation  (v4 pipeline)")
    print("=" * 70)
    print(f"  wav-dir          : {args.wav_dir}")
    print(f"  fake-output      : {args.fake_output}")
    print(f"  inference-delay  : {args.inference_delay_ms} ms")
    print(f"  rpi4-ram         : {args.rpi4_ram} MB")
    print(f"  budget           : {args.budget_ms} ms")
    print(f"  repeat           : {args.repeat}")
    print(f"  throttle         : {'OFF' if args.no_throttle else 'ON'}")
    print()

    # Load or create normalisation stats
    if args.stats_path:
        mu, sdv = load_stats(args.stats_path)
        print(f"  stats            : {args.stats_path}")
    else:
        mu, sdv = create_mock_stats()
        print("  stats            : mock (mu=0, sd=1)")
    print()

    # Discover wav files
    wavs = discover_wavs(args.wav_dir)
    print(f"  Found {len(wavs)} wav file(s):")
    for w in wavs:
        print(f"    - {w.name}")
    print()

    # Build simulation components
    throttler = CPUThrottler(enabled=not args.no_throttle)
    profiler = TimingProfiler(budget_ms=args.budget_ms, throttler=throttler)
    mem = MemoryMonitor(rpi4_total_mb=args.rpi4_ram)
    mem.start()
    mem.snapshot("init")

    model = MockLiteModel(
        fake_output=args.fake_output,
        inference_delay_ms=args.inference_delay_ms,
    )
    mem.snapshot("model_loaded")

    # --- Run simulations ---
    all_results: List[dict] = []

    total_runs = len(wavs) * args.repeat
    run_idx = 0

    for rep in range(args.repeat):
        if args.repeat > 1:
            print(f"--- Repeat {rep + 1}/{args.repeat} ---")

        for wav_path in wavs:
            run_idx += 1
            result = simulate_one(
                wav_path=wav_path,
                model=model,
                mu=mu,
                sdv=sdv,
                throttler=throttler,
                profiler=profiler,
                mem=mem,
            )
            all_results.append(result)
            print_file_report(result, run_idx, total_runs)

    # --- Final report ---
    mem.snapshot("final")
    print_aggregate_report(all_results, mem, args.budget_ms)
    mem.stop()


if __name__ == "__main__":
    main()

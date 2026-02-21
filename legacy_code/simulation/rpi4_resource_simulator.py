"""
RPi4 Resource Simulator
=======================
Three components that simulate the resource constraints of a Raspberry Pi 4
(2 GB RAM, ARM Cortex-A72 @ 1.5 GHz) on a desktop/laptop.

1. **CPUThrottler** — adds artificial sleep after each pipeline stage to
   approximate the wall-clock time on an RPi4.
2. **MemoryMonitor** — tracks peak RSS / tracemalloc snapshots and warns
   when usage would exceed the RPi4 memory budget.
3. **TimingProfiler** — records per-stage desktop-actual and RPi4-simulated
   durations, then reports a formatted summary.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional


# ===================================================================== #
#  1. CPU Throttler
# ===================================================================== #

# Empirical slowdown factors: RPi4 ARM Cortex-A72 vs. typical x86 laptop.
# Based on benchmarks of librosa, numpy, and TFLite operations.
DEFAULT_SLOWDOWN_FACTORS: Dict[str, float] = {
    "resample":        7.0,   # librosa.resample 48k -> 16k
    "melspectrogram":  5.0,   # librosa.feature.melspectrogram
    "normalize":       3.0,   # RMS normalisation / z-score
    "tflite_invoke":   4.0,   # TFLite Interpreter.invoke()
    "file_io":         2.0,   # wav file read/write
    "pad_trim":        2.0,   # librosa.util.fix_length / np slicing
    "preproc_other":   3.0,   # catch-all for remaining pre-processing
}


class CPUThrottler:
    """Injects artificial delay to simulate RPi4 CPU speed.

    After a pipeline stage runs on the desktop, the throttler sleeps for
    ``(desktop_elapsed * (factor - 1))`` so that the *total* wall-clock
    time approximates what RPi4 would take.

    Parameters
    ----------
    slowdown_factors : dict, optional
        Mapping of ``stage_name -> float`` multiplier.
        Missing stages fall back to ``default_factor``.
    default_factor : float
        Fallback slowdown multiplier for unlisted stages.
    enabled : bool
        If False, no throttling is applied (useful for quick desktop runs).
    """

    def __init__(
        self,
        slowdown_factors: Optional[Dict[str, float]] = None,
        default_factor: float = 3.0,
        enabled: bool = True,
    ) -> None:
        self.factors = dict(DEFAULT_SLOWDOWN_FACTORS)
        if slowdown_factors:
            self.factors.update(slowdown_factors)
        self.default_factor = default_factor
        self.enabled = enabled

    def get_factor(self, stage: str) -> float:
        return self.factors.get(stage, self.default_factor)

    @contextmanager
    def throttle(self, stage: str) -> Generator[None, None, None]:
        """Context manager that sleeps after the block to simulate RPi4.

        Yields control to the caller, measures elapsed desktop time, then
        sleeps for the remaining simulated duration.

        Usage::

            with throttler.throttle("resample"):
                y16 = librosa.resample(y48, ...)
        """
        factor = self.get_factor(stage)
        t0 = time.perf_counter()
        yield
        desktop_elapsed = time.perf_counter() - t0

        if self.enabled and factor > 1.0:
            extra = desktop_elapsed * (factor - 1.0)
            time.sleep(extra)

    def simulated_time(self, stage: str, desktop_elapsed: float) -> float:
        """Return what the desktop elapsed time would be on RPi4."""
        return desktop_elapsed * self.get_factor(stage)


# ===================================================================== #
#  2. Memory Monitor
# ===================================================================== #

@dataclass
class MemorySnapshot:
    """A single memory measurement."""
    stage: str
    current_mb: float
    peak_mb: float


class MemoryMonitor:
    """Tracks Python memory via ``tracemalloc`` and warns on RPi4 limits.

    Parameters
    ----------
    rpi4_total_mb : int
        Total RAM on the target RPi4 (default 2048 MB).
    os_reserved_mb : int
        RAM reserved for the OS and desktop environment.
    """

    def __init__(
        self,
        rpi4_total_mb: int = 2048,
        os_reserved_mb: int = 512,
    ) -> None:
        self.rpi4_total_mb = rpi4_total_mb
        self.os_reserved_mb = os_reserved_mb
        self.app_limit_mb = rpi4_total_mb - os_reserved_mb
        self.snapshots: List[MemorySnapshot] = []
        self._active = False

    def start(self) -> None:
        """Begin tracemalloc tracking."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._active = True

    def snapshot(self, stage: str) -> MemorySnapshot:
        """Take a memory snapshot and store it.

        Returns the snapshot for immediate inspection.
        """
        if not self._active:
            self.start()

        current, peak = tracemalloc.get_traced_memory()
        snap = MemorySnapshot(
            stage=stage,
            current_mb=current / (1024 * 1024),
            peak_mb=peak / (1024 * 1024),
        )
        self.snapshots.append(snap)
        return snap

    def check_limit(self, snap: Optional[MemorySnapshot] = None) -> bool:
        """Return True if peak memory is within RPi4 budget.

        Prints a warning to stderr if the limit is exceeded.
        """
        if snap is None:
            snap = self.snapshots[-1] if self.snapshots else None
        if snap is None:
            return True

        within = snap.peak_mb <= self.app_limit_mb
        if not within:
            import sys
            print(
                f"[MemoryMonitor] WARNING: peak {snap.peak_mb:.1f} MB "
                f"exceeds RPi4 app limit {self.app_limit_mb} MB "
                f"(stage: {snap.stage})",
                file=sys.stderr,
            )
        return within

    def stop(self) -> None:
        """Stop tracemalloc."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        self._active = False

    def summary_table(self) -> str:
        """Return a formatted text table of all snapshots."""
        lines = [
            f"{'Stage':<25} {'Current (MB)':>14} {'Peak (MB)':>14} {'Status':>10}",
            "-" * 67,
        ]
        for s in self.snapshots:
            status = "OK" if s.peak_mb <= self.app_limit_mb else "OVER"
            lines.append(
                f"{s.stage:<25} {s.current_mb:>14.2f} {s.peak_mb:>14.2f} {status:>10}"
            )
        return "\n".join(lines)


# ===================================================================== #
#  3. Timing Profiler
# ===================================================================== #

@dataclass
class TimingRecord:
    """Timing measurement for one pipeline stage."""
    stage: str
    desktop_ms: float
    rpi4_estimated_ms: float
    slowdown_factor: float


class TimingProfiler:
    """Measures per-stage timing and compares against an RPi4 budget.

    Parameters
    ----------
    budget_ms : float
        Total processing-time budget for one inference cycle on RPi4.
        Default 1000 ms (v4 event-triggered architecture with 1 s budget).
    throttler : CPUThrottler or None
        If provided, used to compute RPi4 estimated times.
    """

    def __init__(
        self,
        budget_ms: float = 1000.0,
        throttler: Optional[CPUThrottler] = None,
    ) -> None:
        self.budget_ms = budget_ms
        self.throttler = throttler or CPUThrottler(enabled=False)
        self.records: List[TimingRecord] = []

    @contextmanager
    def measure(self, stage: str) -> Generator[None, None, None]:
        """Context manager that records desktop and simulated RPi4 time.

        Usage::

            with profiler.measure("resample"):
                y16 = librosa.resample(y48, ...)
        """
        t0 = time.perf_counter()
        yield
        desktop_elapsed = time.perf_counter() - t0
        desktop_ms = desktop_elapsed * 1000.0
        factor = self.throttler.get_factor(stage)
        rpi4_ms = desktop_ms * factor

        self.records.append(TimingRecord(
            stage=stage,
            desktop_ms=desktop_ms,
            rpi4_estimated_ms=rpi4_ms,
            slowdown_factor=factor,
        ))

    def total_desktop_ms(self) -> float:
        return sum(r.desktop_ms for r in self.records)

    def total_rpi4_ms(self) -> float:
        return sum(r.rpi4_estimated_ms for r in self.records)

    def within_budget(self) -> bool:
        return self.total_rpi4_ms() <= self.budget_ms

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self.records.clear()

    def summary_table(self) -> str:
        """Return a formatted text table of timing measurements."""
        lines = [
            f"{'Stage':<25} {'Desktop (ms)':>14} {'RPi4 Est (ms)':>15} {'Factor':>8}",
            "-" * 66,
        ]
        for r in self.records:
            lines.append(
                f"{r.stage:<25} {r.desktop_ms:>14.2f} {r.rpi4_estimated_ms:>15.2f} {r.slowdown_factor:>8.1f}x"
            )
        lines.append("-" * 66)

        total_desk = self.total_desktop_ms()
        total_rpi4 = self.total_rpi4_ms()
        verdict = "PASS" if self.within_budget() else "FAIL"
        lines.append(
            f"{'TOTAL':<25} {total_desk:>14.2f} {total_rpi4:>15.2f} "
            f"{'budget':>8}"
        )
        lines.append(
            f"{'Budget':<25} {'':>14} {self.budget_ms:>15.1f} {verdict:>8}"
        )
        return "\n".join(lines)

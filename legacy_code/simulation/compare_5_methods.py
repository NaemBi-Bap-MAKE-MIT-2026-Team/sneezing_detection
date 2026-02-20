from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import librosa

from legacy_code.simulation.v4_pipeline import (
    CAPTURE_SR,
    PROB_TH,
    rms,
    resample_48k_to_16k,
    preproc,
    create_mock_stats,
    load_stats,
)
from legacy_code.simulation.rpi4_resource_simulator import CPUThrottler, TimingProfiler


# -----------------------------
# WAV load (full length, 48 kHz mono)
# -----------------------------
def load_wav_48k_full(path: Path) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != CAPTURE_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CAPTURE_SR)
    return y.astype(np.float32)


def safe_pct(numer: float, denom: float) -> float:
    if denom <= 1e-9:
        return float("nan")
    return 100.0 * (numer / denom)


# -----------------------------
# "Inference" that profiles real preprocessing cost,
# and simulates model invoke cost using sleep.
# Output probability is f(time) = high if infer_time >= onset else low.
# -----------------------------
def infer_once_profiled(
    y48_2s: np.ndarray,
    infer_time_sec: float,
    onset_sec: float,
    fake_low: float,
    fake_high: float,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    inference_delay_ms: float,
) -> float:
    profiler.reset()

    # 1) resample
    with profiler.measure("resample"):
        with throttler.throttle("resample"):
            y16 = resample_48k_to_16k(y48_2s)

    # 2) preproc (pad/trim + rms norm + logmel + zscore)
    with profiler.measure("preproc_other"):
        with throttler.throttle("preproc_other"):
            _ = preproc(y16, mu, sdv)

    # 3) "tflite invoke" simulated
    with profiler.measure("tflite_invoke"):
        with throttler.throttle("tflite_invoke"):
            if inference_delay_ms > 0:
                time.sleep(inference_delay_ms / 1000.0)
            p = float(fake_high if infer_time_sec >= onset_sec else fake_low)

    return p


# -----------------------------
# Result container
# -----------------------------
@dataclass
class MethodResult:
    name: str
    first_detect_time: Optional[float]
    first_prob: Optional[float]
    last_prob: Optional[float]
    infer_calls: int
    rpi4_total_ms: float
    rpi4_ms_per_infer: float
    cpu_util_est_pct: float
    latency_sec: Optional[float]


# -----------------------------
# Common stream helpers
# -----------------------------
def make_frames(y48: np.ndarray, frame_samples: int) -> List[np.ndarray]:
    frames = []
    i = 0
    while i < len(y48):
        frames.append(y48[i:i + frame_samples])
        i += frame_samples
    return frames


def pad_or_trim_to_2s(y: np.ndarray) -> np.ndarray:
    target = int(CAPTURE_SR * 2.0)
    if len(y) > target:
        return y[:target]
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y


# -----------------------------
# 1) Pure Trigger: after RMS trigger, collect future 2s then infer once
# -----------------------------
def run_pure_trigger(
    frames: List[np.ndarray],
    frame_sec: float,
    rms_trigger_th: float,
    cooldown_sec: float,
    onset_sec: float,
    fake_low: float,
    fake_high: float,
    prob_th: float,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    inference_delay_ms: float,
) -> MethodResult:
    frame_samples = int(CAPTURE_SR * frame_sec)
    target_samples = int(CAPTURE_SR * 2.0)

    collecting = False
    collected: List[np.ndarray] = []
    collected_len = 0
    ignore_until = -1e9

    first_detect_time = None
    first_prob = None
    last_prob = None

    infer_calls = 0
    total_ms = 0.0

    now_sec = 0.0

    for x in frames:
        if len(x) == 0:
            break

        if now_sec < ignore_until:
            now_sec += frame_sec
            continue

        if not collecting:
            if rms(x) >= rms_trigger_th:
                collecting = True
                collected = [x]  # no pre-roll
                collected_len = len(x)
        else:
            collected.append(x)
            collected_len += len(x)

        if collecting and collected_len >= target_samples:
            y48 = pad_or_trim_to_2s(np.concatenate(collected, axis=0))
            p = infer_once_profiled(
                y48_2s=y48,
                infer_time_sec=now_sec,
                onset_sec=onset_sec,
                fake_low=fake_low,
                fake_high=fake_high,
                mu=mu,
                sdv=sdv,
                throttler=throttler,
                profiler=profiler,
                inference_delay_ms=inference_delay_ms,
            )
            infer_calls += 1
            total_ms += profiler.total_rpi4_ms()
            last_prob = p

            if (p >= prob_th) and (first_detect_time is None):
                first_detect_time = now_sec
                first_prob = p
                ignore_until = now_sec + cooldown_sec

            collecting = False
            collected = []
            collected_len = 0

        now_sec += frame_sec

    ms_per = (total_ms / infer_calls) if infer_calls > 0 else float("nan")
    return MethodResult(
        name="pure_trigger",
        first_detect_time=first_detect_time,
        first_prob=first_prob,
        last_prob=last_prob,
        infer_calls=infer_calls,
        rpi4_total_ms=total_ms,
        rpi4_ms_per_infer=ms_per,
        cpu_util_est_pct=0.0,  # filled by caller
        latency_sec=None,      # filled by caller
    )


# -----------------------------
# 2/3) Always Sliding: 2s ring buffer + hop inference until end
# -----------------------------
def run_sliding_always(
    frames: List[np.ndarray],
    chunk_sec: float,
    hop_sec: float,
    cooldown_sec: float,
    onset_sec: float,
    fake_low: float,
    fake_high: float,
    prob_th: float,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    inference_delay_ms: float,
    name: str,
) -> MethodResult:
    chunk_samples = int(CAPTURE_SR * chunk_sec)
    hop_samples = int(CAPTURE_SR * hop_sec)
    win_samples = int(CAPTURE_SR * 2.0)

    buf = np.zeros(win_samples, dtype=np.float32)
    filled = 0
    since_last = 0

    ignore_until = -1e9
    first_detect_time = None
    first_prob = None
    last_prob = None

    infer_calls = 0
    total_ms = 0.0

    now_sec = 0.0

    for x in frames:
        if len(x) == 0:
            break

        # push ring
        n = len(x)
        if n >= win_samples:
            buf[:] = x[-win_samples:]
            filled = win_samples
        else:
            buf = np.roll(buf, -n)
            buf[-n:] = x
            filled = min(win_samples, filled + n)

        since_last += n

        # infer every hop after window is filled
        if filled >= win_samples and since_last >= hop_samples:
            since_last = 0
            y48 = buf.copy()

            p = infer_once_profiled(
                y48_2s=y48,
                infer_time_sec=now_sec,
                onset_sec=onset_sec,
                fake_low=fake_low,
                fake_high=fake_high,
                mu=mu,
                sdv=sdv,
                throttler=throttler,
                profiler=profiler,
                inference_delay_ms=inference_delay_ms,
            )
            infer_calls += 1
            total_ms += profiler.total_rpi4_ms()
            last_prob = p

            detected = (p >= prob_th) and (now_sec >= ignore_until)
            if detected and first_detect_time is None:
                first_detect_time = now_sec
                first_prob = p
                ignore_until = now_sec + cooldown_sec

        now_sec += chunk_sec

    ms_per = (total_ms / infer_calls) if infer_calls > 0 else float("nan")
    return MethodResult(
        name=name,
        first_detect_time=first_detect_time,
        first_prob=first_prob,
        last_prob=last_prob,
        infer_calls=infer_calls,
        rpi4_total_ms=total_ms,
        rpi4_ms_per_infer=ms_per,
        cpu_util_est_pct=0.0,
        latency_sec=None,
    )


# -----------------------------
# 4) Capture-type (your code): 0.5s pre-buffer + trigger + collect to 2s then infer once
# -----------------------------
def run_capture_type(
    frames: List[np.ndarray],
    frame_sec: float,
    pre_seconds: float,
    rms_trigger_th: float,
    cooldown_sec: float,
    onset_sec: float,
    fake_low: float,
    fake_high: float,
    prob_th: float,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    inference_delay_ms: float,
) -> MethodResult:
    frame_samples = int(CAPTURE_SR * frame_sec)
    target_samples = int(CAPTURE_SR * 2.0)

    pre_frames = max(1, int(pre_seconds / frame_sec))
    pre_buf: List[np.ndarray] = []

    capturing = False
    captured: List[np.ndarray] = []
    captured_len = 0
    ignore_until = -1e9

    first_detect_time = None
    first_prob = None
    last_prob = None

    infer_calls = 0
    total_ms = 0.0

    now_sec = 0.0

    def push_pre(x: np.ndarray):
        pre_buf.append(x)
        if len(pre_buf) > pre_frames:
            pre_buf.pop(0)

    for x in frames:
        if len(x) == 0:
            break

        if now_sec < ignore_until:
            push_pre(x)
            now_sec += frame_sec
            continue

        if not capturing:
            push_pre(x)
            if rms(x) >= rms_trigger_th:
                capturing = True
                captured = list(pre_buf)
                captured.append(x)
                captured_len = int(sum(len(c) for c in captured))
        else:
            captured.append(x)
            captured_len += len(x)

        if capturing and captured_len >= target_samples:
            y48 = pad_or_trim_to_2s(np.concatenate(captured, axis=0))
            p = infer_once_profiled(
                y48_2s=y48,
                infer_time_sec=now_sec,
                onset_sec=onset_sec,
                fake_low=fake_low,
                fake_high=fake_high,
                mu=mu,
                sdv=sdv,
                throttler=throttler,
                profiler=profiler,
                inference_delay_ms=inference_delay_ms,
            )
            infer_calls += 1
            total_ms += profiler.total_rpi4_ms()
            last_prob = p

            if (p >= prob_th) and (first_detect_time is None):
                first_detect_time = now_sec
                first_prob = p
                ignore_until = now_sec + cooldown_sec

            capturing = False
            captured = []
            captured_len = 0
            pre_buf.clear()

        now_sec += frame_sec

    ms_per = (total_ms / infer_calls) if infer_calls > 0 else float("nan")
    return MethodResult(
        name="capture_type",
        first_detect_time=first_detect_time,
        first_prob=first_prob,
        last_prob=last_prob,
        infer_calls=infer_calls,
        rpi4_total_ms=total_ms,
        rpi4_ms_per_infer=ms_per,
        cpu_util_est_pct=0.0,
        latency_sec=None,
    )


# -----------------------------
# 5) Burst-type: RMS guard always, when triggered run short sliding burst for burst_sec
# -----------------------------
def run_burst_type(
    frames: List[np.ndarray],
    chunk_sec: float,
    rms_trigger_th: float,
    burst_sec: float,
    burst_hop_sec: float,
    cooldown_sec: float,
    onset_sec: float,
    fake_low: float,
    fake_high: float,
    prob_th: float,
    mu: np.ndarray,
    sdv: np.ndarray,
    throttler: CPUThrottler,
    profiler: TimingProfiler,
    inference_delay_ms: float,
) -> MethodResult:
    chunk_samples = int(CAPTURE_SR * chunk_sec)
    win_samples = int(CAPTURE_SR * 2.0)

    # Always keep 2s ring buffer
    buf = np.zeros(win_samples, dtype=np.float32)
    filled = 0

    # Burst state
    in_burst = False
    burst_end_time = -1e9
    hop_samples = int(CAPTURE_SR * burst_hop_sec)
    since_last = 0

    ignore_until = -1e9

    first_detect_time = None
    first_prob = None
    last_prob = None

    infer_calls = 0
    total_ms = 0.0

    now_sec = 0.0

    for x in frames:
        if len(x) == 0:
            break

        # push ring
        n = len(x)
        if n >= win_samples:
            buf[:] = x[-win_samples:]
            filled = win_samples
        else:
            buf = np.roll(buf, -n)
            buf[-n:] = x
            filled = min(win_samples, filled + n)

        # RMS guard on current chunk
        if (not in_burst) and (now_sec >= ignore_until):
            if rms(x) >= rms_trigger_th:
                in_burst = True
                burst_end_time = now_sec + burst_sec
                since_last = hop_samples  # allow immediate infer if buffer filled

        # If burst active, do sliding inference only during burst window
        if in_burst and filled >= win_samples:
            since_last += n
            if since_last >= hop_samples:
                since_last = 0
                y48 = buf.copy()
                p = infer_once_profiled(
                    y48_2s=y48,
                    infer_time_sec=now_sec,
                    onset_sec=onset_sec,
                    fake_low=fake_low,
                    fake_high=fake_high,
                    mu=mu,
                    sdv=sdv,
                    throttler=throttler,
                    profiler=profiler,
                    inference_delay_ms=inference_delay_ms,
                )
                infer_calls += 1
                total_ms += profiler.total_rpi4_ms()
                last_prob = p

                detected = (p >= prob_th) and (now_sec >= ignore_until)
                if detected and first_detect_time is None:
                    first_detect_time = now_sec
                    first_prob = p
                    ignore_until = now_sec + cooldown_sec

            if now_sec >= burst_end_time:
                in_burst = False

        now_sec += chunk_sec

    ms_per = (total_ms / infer_calls) if infer_calls > 0 else float("nan")
    return MethodResult(
        name="burst_type",
        first_detect_time=first_detect_time,
        first_prob=first_prob,
        last_prob=last_prob,
        infer_calls=infer_calls,
        rpi4_total_ms=total_ms,
        rpi4_ms_per_infer=ms_per,
        cpu_util_est_pct=0.0,
        latency_sec=None,
    )


def finalize(result: MethodResult, duration_sec: float, onset_sec: float) -> MethodResult:
    cpu_pct = safe_pct(result.rpi4_total_ms / 1000.0, duration_sec)
    if result.first_detect_time is None:
        lat = None
    else:
        lat = result.first_detect_time - onset_sec
    result.cpu_util_est_pct = cpu_pct
    result.latency_sec = lat
    return result


def print_results(results: List[MethodResult]) -> None:
    print("=" * 78)
    print("COMPARE 5 METHODS (pure_trigger / sliding1.0 / sliding0.5 / capture / burst)")
    print("=" * 78)
    print(f"{'method':<16} {'det':>4} {'first_t':>10} {'latency':>10} {'calls':>7} {'ms/inf':>10} {'total_ms':>10} {'cpu%':>7}")
    print("-" * 78)
    for r in results:
        det = "YES" if r.first_detect_time is not None else "NO"
        ft = f"{r.first_detect_time:>10.3f}" if r.first_detect_time is not None else f"{'None':>10}"
        lat = f"{r.latency_sec:>10.3f}" if r.latency_sec is not None else f"{'None':>10}"
        ms_inf = f"{r.rpi4_ms_per_infer:>10.2f}" if np.isfinite(r.rpi4_ms_per_infer) else f"{'nan':>10}"
        print(f"{r.name:<16} {det:>4} {ft} {lat} {r.infer_calls:>7d} {ms_inf} {r.rpi4_total_ms:>10.2f} {r.cpu_util_est_pct:>7.1f}")
    print("-" * 78)
    print("notes")
    print("1) latency = first_detect_time - onset_sec (seconds)")
    print("2) cpu% = (total_rpi4_sec / audio_sec) * 100, includes preprocessing + simulated invoke")
    print("3) pure_trigger/capture_type run only on RMS events; sliding runs continuously; burst runs only during burst window")
    print("=" * 78)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("Compare 5 sneeze detection strategies on a full-length wav")
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--onset-sec", type=float, default=0.0)
    p.add_argument("--stats-path", type=Path, default=None)

    p.add_argument("--fake-low", type=float, default=0.01)
    p.add_argument("--fake-high", type=float, default=0.95)
    p.add_argument("--inference-delay-ms", type=float, default=10.0)

    p.add_argument("--prob-th", type=float, default=PROB_TH)
    p.add_argument("--cooldown-sec", type=float, default=1.5)

    p.add_argument("--no-throttle", action="store_true")

    # trigger / capture params
    p.add_argument("--frame-sec", type=float, default=0.10)
    p.add_argument("--pre-seconds", type=float, default=0.5)
    p.add_argument("--rms-trigger-th", type=float, default=0.008)

    # sliding params
    p.add_argument("--chunk-sec", type=float, default=0.10)

    # burst params
    p.add_argument("--burst-sec", type=float, default=3.0)
    p.add_argument("--burst-hop", type=float, default=0.5)

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    y48 = load_wav_48k_full(args.wav)
    duration_sec = len(y48) / CAPTURE_SR

    if args.stats_path:
        mu, sdv = load_stats(args.stats_path)
        stats_name = str(args.stats_path)
    else:
        mu, sdv = create_mock_stats()
        stats_name = "mock(mu=0, sd=1)"

    throttler = CPUThrottler(enabled=not args.no_throttle)
    profiler = TimingProfiler(budget_ms=999999, throttler=throttler)

    frame_samples = int(CAPTURE_SR * args.frame_sec)
    chunk_samples = int(CAPTURE_SR * args.chunk_sec)

    frames_trigger = make_frames(y48, frame_samples)
    frames_sliding = make_frames(y48, chunk_samples)

    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    print(f"wav                 : {args.wav}")
    print(f"duration-sec        : {duration_sec:.3f}")
    print(f"onset-sec           : {args.onset_sec:.3f}")
    print(f"stats               : {stats_name}")
    print(f"fake-low/high       : {args.fake_low} / {args.fake_high}")
    print(f"inference-delay-ms  : {args.inference_delay_ms}")
    print(f"throttle            : {'OFF' if args.no_throttle else 'ON'}")
    print()
    print("params")
    print(f"  prob-th           : {args.prob_th}")
    print(f"  cooldown-sec      : {args.cooldown_sec}")
    print(f"  rms-trigger-th    : {args.rms_trigger_th}")
    print(f"  frame-sec         : {args.frame_sec}")
    print(f"  pre-seconds       : {args.pre_seconds}")
    print(f"  chunk-sec         : {args.chunk_sec}")
    print(f"  burst-sec         : {args.burst_sec}")
    print(f"  burst-hop         : {args.burst_hop}")
    print("=" * 78)

    results: List[MethodResult] = []

    r1 = run_pure_trigger(
        frames=frames_trigger,
        frame_sec=args.frame_sec,
        rms_trigger_th=args.rms_trigger_th,
        cooldown_sec=args.cooldown_sec,
        onset_sec=args.onset_sec,
        fake_low=args.fake_low,
        fake_high=args.fake_high,
        prob_th=args.prob_th,
        mu=mu,
        sdv=sdv,
        throttler=throttler,
        profiler=profiler,
        inference_delay_ms=args.inference_delay_ms,
    )
    results.append(finalize(r1, duration_sec, args.onset_sec))

    r2 = run_sliding_always(
        frames=frames_sliding,
        chunk_sec=args.chunk_sec,
        hop_sec=1.0,
        cooldown_sec=args.cooldown_sec,
        onset_sec=args.onset_sec,
        fake_low=args.fake_low,
        fake_high=args.fake_high,
        prob_th=args.prob_th,
        mu=mu,
        sdv=sdv,
        throttler=throttler,
        profiler=profiler,
        inference_delay_ms=args.inference_delay_ms,
        name="sliding_1.0",
    )
    results.append(finalize(r2, duration_sec, args.onset_sec))

    r3 = run_sliding_always(
        frames=frames_sliding,
        chunk_sec=args.chunk_sec,
        hop_sec=0.5,
        cooldown_sec=args.cooldown_sec,
        onset_sec=args.onset_sec,
        fake_low=args.fake_low,
        fake_high=args.fake_high,
        prob_th=args.prob_th,
        mu=mu,
        sdv=sdv,
        throttler=throttler,
        profiler=profiler,
        inference_delay_ms=args.inference_delay_ms,
        name="sliding_0.5",
    )
    results.append(finalize(r3, duration_sec, args.onset_sec))

    r4 = run_capture_type(
        frames=frames_trigger,
        frame_sec=args.frame_sec,
        pre_seconds=args.pre_seconds,
        rms_trigger_th=args.rms_trigger_th,
        cooldown_sec=args.cooldown_sec,
        onset_sec=args.onset_sec,
        fake_low=args.fake_low,
        fake_high=args.fake_high,
        prob_th=args.prob_th,
        mu=mu,
        sdv=sdv,
        throttler=throttler,
        profiler=profiler,
        inference_delay_ms=args.inference_delay_ms,
    )
    results.append(finalize(r4, duration_sec, args.onset_sec))

    r5 = run_burst_type(
        frames=frames_sliding,
        chunk_sec=args.chunk_sec,
        rms_trigger_th=args.rms_trigger_th,
        burst_sec=args.burst_sec,
        burst_hop_sec=args.burst_hop,
        cooldown_sec=args.cooldown_sec,
        onset_sec=args.onset_sec,
        fake_low=args.fake_low,
        fake_high=args.fake_high,
        prob_th=args.prob_th,
        mu=mu,
        sdv=sdv,
        throttler=throttler,
        profiler=profiler,
        inference_delay_ms=args.inference_delay_ms,
    )
    results.append(finalize(r5, duration_sec, args.onset_sec))

    print_results(results)


if __name__ == "__main__":
    main()
# communication

Unified audio-input layer for the sneeze-detection pipeline.

Provides two interchangeable microphone sources and one sender:

| Class / Script | Role |
|---|---|
| `MicrophoneStream` | Local sounddevice microphone (re-exported from `microphone/`) |
| `AudioSender` | Captures a local mic and streams frames over UDP |
| `NetworkMicStream` | Receives UDP audio from `AudioSender`; identical API to `MicrophoneStream` |

---

## Architecture

```
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  Device A  (send.py)         │        │  Device B  (main.py)         │
│                              │        │                              │
│  MicrophoneStream            │        │  NetworkMicStream            │
│    └─ sounddevice InputStream│        │    └─ UDP socket (recv)      │
│         │ 10 ms frames       │  UDP   │         │ reassemble         │
│  AudioSender                 │───────▶│         │ → frame_sec frames │
│    └─ sock.sendto()          │        │    └─ queue.put(frame)       │
│       raw float32 bytes      │        │                              │
│       no header, no delay    │        │  main.py detection loop      │
│                              │        │    └─ mic.read()  ← same API │
└──────────────────────────────┘        └──────────────────────────────┘
```

**Protocol**: UDP, raw `float32` bytes, no header
**Sender block**: 10 ms (480 samples @ 48 kHz) → ~100 packets / second
**Receiver frame**: assembled to `frame_sec` (default 100 ms = 4 800 samples)

---

## Shared interface

Both `MicrophoneStream` and `NetworkMicStream` expose the exact same API:

```python
with mic:
    while True:
        frame = mic.read()          # np.ndarray shape (frame_samples,) float32 — blocking
        pre   = mic.pre_buffer      # collections.deque of recent frames
```

`main.py` works with either source without modification.

---

## Quickstart

### Local microphone (no network)

```python
from communication import MicrophoneStream

mic = MicrophoneStream(capture_sr=48000, frame_sec=0.10, pre_seconds=0.5)
with mic:
    frame = mic.read()
    print(frame.shape, frame.dtype)  # (4800,) float32
```

### Network microphone — sender side

```bash
# Same machine (loopback test)
python send.py

# Remote Raspberry Pi
python send.py --host 192.168.1.42 --port 12345
```

Or as a class:

```python
from communication import AudioSender

sender = AudioSender(host="192.168.1.42", port=12345, capture_sr=48000, block_ms=10)
sender.run()   # blocks until Ctrl+C
```

### Network microphone — receiver side

```python
from communication import NetworkMicStream

mic = NetworkMicStream(host="0.0.0.0", port=12345,
                       capture_sr=48000, frame_sec=0.10, pre_seconds=0.5)
with mic:
    while True:
        frame = mic.read()          # identical to MicrophoneStream.read()
        pre   = mic.pre_buffer
```

### main.py — switching between sources

```bash
# Local mic (default)
python main.py

# Receive audio from send.py over the network
python main.py --network --recv-host 0.0.0.0 --recv-port 12345
```

---

## CLI reference — send.py

```
python send.py [options]

--host          Destination IP (recv device)        default: 127.0.0.1
--port          UDP port                            default: 12345
--capture-sr    Microphone sample rate (Hz)         default: 48000
--block-ms      Packet size in ms                   default: 10
--device        sounddevice input device index/name default: system default
```

---

## Latency notes

| Stage | Latency |
|---|---|
| Mic capture block | 10 ms (configurable via `--block-ms`) |
| UDP transmission (LAN) | < 1 ms |
| Frame assembly on receiver | 0 ms overhead (fills as packets arrive) |
| **Total one-way** | **~10 ms** |

Reducing `--block-ms` lowers latency further but increases CPU load from more frequent `sendto()` calls.

---

## Dependencies

| Library | Usage |
|---|---|
| `sounddevice` | Microphone capture (`MicrophoneStream`, `AudioSender`) |
| `numpy` | Raw float32 array serialisation |
| Python `socket` | UDP send / receive |
| Python `threading` | Background receive loop in `NetworkMicStream` |

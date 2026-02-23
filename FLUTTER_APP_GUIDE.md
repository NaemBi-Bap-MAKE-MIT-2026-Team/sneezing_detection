# Flutter Sneeze Detection Streamer — Implementation Guide

## Overview

A production-ready Flutter app that replaces `send.py`, streaming real-time microphone audio (48 kHz, 10ms frames) as raw float32 UDP packets to Raspberry Pi's `main.py`. Supports both iOS (14.0+) and Android (API 21+) with auto-reconnect and exponential backoff.

---

## Project Structure

```
app/
├── lib/
│   ├── main.dart                      # AppEntry point, theme, routing
│   ├── screens/
│   │   └── stream_screen.dart         # Main UI (input, Connect/Disconnect button, status)
│   ├── services/
│   │   ├── audio_capture.dart         # AudioCaptureService (48kHz microphone capture)
│   │   ├── udp_sender.dart            # UDPAudioSender (raw float32 → UDP)
│   │   └── connection_manager.dart    # Orchestrates capture + send + reconnect logic
│   ├── models/
│   │   └── connection_state.dart      # ConnectionState enum + data class
│   ├── utils/
│   │   ├── constants.dart             # AppConstants (all tunable parameters)
│   │   └── logger.dart                # AppLogger (in-app debug console)
│   └── widgets/
│       └── status_indicator.dart      # StatusIndicator (LED-style connection display)
├── android/
│   └── app/src/main/AndroidManifest.xml  # Permissions: RECORD_AUDIO, INTERNET, ACCESS_NETWORK_STATE
├── ios/
│   └── Runner/Info.plist              # Permissions: NSMicrophoneUsageDescription, NSLocalNetworkUsageDescription
├── pubspec.yaml                       # Dependencies: record, permission_handler, connectivity_plus, get
└── README.md                          # User documentation
```

---

## Key Components

### 1. AudioCaptureService (`lib/services/audio_capture.dart`)

**Purpose**: Capture microphone audio and yield frames as a stream.

**API**:
```dart
Stream<List<double>> startCapture()        // Start capturing, yields frames (480 float32 samples)
Future<void> stopCapture()                 // Stop capturing
Future<bool> requestPermission()           // Show permission dialog
Future<void> dispose()                     // Cleanup
```

**Audio Format**:
- Sample Rate: 48,000 Hz
- Frame Size: 480 samples (10 ms)
- Data Type: float32 (range [-1.0, 1.0])
- Encoder: WAV (internally converted from PCM16)

**Implementation Notes**:
- Uses `record` package for cross-platform compatibility
- Internally buffers frames until `FRAME_SAMPLES` is reached
- Handles permission requests via `requestPermission()`

---

### 2. UDPAudioSender (`lib/services/udp_sender.dart`)

**Purpose**: Send audio frames as raw UDP packets to Raspberry Pi.

**API**:
```dart
Future<bool> initialize()              // Resolve host, create socket
bool sendFrame(List<double> samples)   // Send 1 frame (480 samples → 1,920 bytes)
Future<void> close()                   // Close socket
```

**Packet Format**:
- Raw float32 bytes (little-endian), no header
- Size: 480 samples × 4 bytes = 1,920 bytes per packet
- Timing: One packet every 10 ms (~100 packets/sec)

**Implementation Notes**:
- Uses `RawDatagramSocket` for UDP (no external `udp` package dependency)
- Converts `List<double>` → `Float32List` → `Uint8List`
- No error recovery—fires-and-forgets each packet
- Tracks packet count and last successful send time

---

### 3. ConnectionManager (`lib/services/connection_manager.dart`)

**Purpose**: Orchestrate audio capture + UDP sending + reconnection logic.

**State Machine**:
1. **User clicks Connect** → `ConnectionStatus.connecting`
2. **Audio + UDP initialized** → `ConnectionStatus.connected`
3. **Audio stream** → Frames sent via UDP every 10 ms
4. **Send fails or stream ends** → `ConnectionStatus.error`, schedule reconnect
5. **Reconnect timer fires** → Retry with exponential backoff (500ms → 1s → 2s → ... → 10s max)
6. **User clicks Disconnect** → `ConnectionStatus.disconnected`, cancel timers

**Backoff Algorithm**:
```
initial = 500 ms
retry_n = min(initial * 2^n, 10_000 ms)  // Exponential with cap
```

**API**:
```dart
Future<void> connect(String host, int port)     // Initiate connectionection
Future<void> disconnect()                       // Disconnect and cancel retries
// Rx<ConnectionState> connectionState          // Observable state (GetX)
```

---

### 4. StreamScreen (`lib/screens/stream_screen.dart`)

**UI Layout**:
1. **Header**: "Audio Streaming Setup"
2. **Status Card**: Connection indicator (green/orange/red LED), error message, packet count
3. **RPi Address Input**: Text field (enabled when disconnected)
4. **Port Input**: Text field (default 8080, enabled when disconnected)
5. **Connect/Disconnect Button**: Large button (green when disconnected, red when connected)
6. **Info Card**: Features list, "Show Debug Info" toggle
7. **Debug Info** (toggleable): Sample rate, frame size, packet rate, etc.

**Reactive Updates** (via GetX):
- Button state changes based on `ConnectionManager.connectionState`
- Packet count updates every 100 packets
- Error messages displayed in red
- Auto-enable/disable inputs based on connection status

---

### 5. Models & Utilities

**ConnectionState** (`lib/models/connection_state.dart`):
```dart
enum ConnectionStatus { disconnected, connecting, connected, error }

class ConnectionState {
  final ConnectionStatus status;
  final String? errorMessage;
  final int packetsSent;
  final DateTime? connectedSince;
  // ...
}
```

**AppConstants** (`lib/utils/constants.dart`):
```dart
CAPTURE_SAMPLE_RATE = 48000  // Hz (must match RPi config)
FRAME_SAMPLES = 480           // 10ms @ 48kHz
FRAME_BYTES = 1920            // 480 × 4 bytes
DEFAULT_PORT = 8080           // Port to send to RPi
// Reconnect: initial 500ms, max 10s, exponential backoff
// UI: button height 48px, padding 16px, etc.
```

---

## Installation & Build

### Prerequisites
- Flutter SDK 3.8+ ([flutter.dev](https://flutter.dev))
- iOS: Xcode 12+ with Command Line Tools
- Android: Android Studio, SDK API 21+

### Setup

```bash
cd app
flutter pub get

# iOS only
cd ios && pod install && cd ..
```

### Run on Device

```bash
# List available devices
flutter devices

# Run on specific device
flutter run -d <device_id>

# Run with logging
flutter run -v
```

### Build for Distribution

**Android APK** (not recommended for production, use AAB instead):
```bash
flutter build apk --release
# Output: app/build/app/outputs/apk/release/app-release.apk
```

**Android App Bundle** (for Google Play):
```bash
flutter build appbundle --release
# Output: app/build/app/outputs/bundle/release/app-release.aab
```

**iOS** (requires developer account):
```bash
flutter build ios --release
# Output: app/build/ios/iphoneos/Runner.app
# Then archive in Xcode for TestFlight/App Store
```

---

## Permissions

### Android (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

Requested at runtime when user taps **Connect**.

### iOS (Info.plist)

```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs access to your microphone to capture audio for the sneeze detection system.</string>
<key>NSLocalNetworkUsageDescription</key>
<string>This app needs access to local network to connect to your Raspberry Pi for audio streaming.</string>
<key>NSBonjourServices</key>
<array>
  <string>_sneeze._tcp</string>
</array>
```

User sees permission dialog on first **Connect** attempt.

---

## Network Flow

```
┌─────────────────────────────────────┐
│         Flutter App (iPhone/Android)│
│                                     │
│  1. User enters RPi IP: 192.168... │
│  2. User taps "Connect"             │
│                                     │
│  [AudioCaptureService]              │
│   ├─ Request mic permission         │
│   ├─ Start recording @ 48 kHz       │
│   └─ Yield 480-sample frames        │
│                                     │
│  [ConnectionManager]                │
│   ├─ For each frame:                │
│   │  └─ UDPAudioSender.sendFrame()  │
│   └─ Update UI with packet count    │
│                                     │
│  [UDPAudioSender]                   │
│   └─ Convert float32 → Uint8List    │
│       UDP socket → RPi:8080         │
│                                     │
└─────────────────────────────────────┘
         │
         │ UDP ~1,920 bytes per packet
         │ (480 × float32)
         │ ~100 packets/sec
         ↓
┌─────────────────────────────────────┐
│      Raspberry Pi (main.py)         │
│                                     │
│  [NetworkMicStream]                 │
│   ├─ Bind UDP port 8080             │
│   ├─ Accumulate frames from Flutter │
│   └─ Yield complete 2s windows      │
│                                     │
│  [HybridBurstDetector]              │
│   ├─ RMS guard (idle when quiet)    │
│   ├─ Burst phase (3s of inference)  │
│   └─ Detect sneeze prob > 0.95      │
│                                     │
│  ✨ BLESS YOU Response (optional)   │
│   ├─ Play bless_you.wav             │
│   ├─ TTS via ElevenLabs (if enabled)│
│   └─ Show GIF on LCD                │
│                                     │
└─────────────────────────────────────┘
```

---

## Testing Checklist

- [ ] **Permission**: App requests mic permission on Android/iOS
- [ ] **Audio Capture**: Microphone captures cleanly (test in voice app first)
- [ ] **UDP Send**: Packets sent every 10 ms (use tcpdump/Wireshark to verify)
- [ ] **Connection**: Status changes green when connected
- [ ] **RPi Reception**: `main.py` console shows "NetworkMicStream: received N packets"
- [ ] **Sneeze Detection**: Cheer or make noise near mic, RPi detects & says "Bless you!"
- [ ] **Auto-Reconnect**: Disconnect WiFi, watch app retry; reconnect WiFi, app resumes
- [ ] **Cleanly Disconnect**: Tap "Disconnect", audio stops, status turns gray

---

## Performance Targets

| Platform | CPU Usage | Memory | Latency |
|---|---|---|---|
| iOS 14+ (iPhone) | 10–20% | ~50 MB | ~2.1 s |
| Android 5+ | 15–25% | ~60 MB | ~2.1 s |

Latency breakdown:
- Audio frame assembly: ~10 ms (1 frame)
- UDP network: ~0–50 ms (LAN)
- RPi processing: ~100 ms (preprocessing + inference)
- RPi ring buffer: ~2,000 ms (2 s window)
- **Total**: ~2,100 ms (2.1 seconds)

---

## Debugging

### Enable Verbose Logging

```bash
flutter run -v
```

Output includes widget build details, frame drops, permission dialogs.

### Monitor Network Traffic

On macOS:
```bash
sudo tcpdump -i en0 'udp port 8080' -X
```

Example output:
```
20:35:45.123456 iPhone.52345 > RPi.8080: UDP (1920 bytes)
20:35:45.133456 iPhone.52347 > RPi.8080: UDP (1920 bytes)
...
```

### Check RPi Reception

On Raspberry Pi, in `main.py` or `NetworkMicStream`:
```python
# Add logging
print(f"[DEBUG] Received {len(x)} samples from {sender_ip}:{sender_port}")
```

### Common Issues

| Issue | Solution |
|---|---|
| "Permission denied" | Tap "Settings" → App → Permissions → Microphone → Allow |
| "Cannot resolve host" | Use IP address instead of hostname (IPv4 format) |
| "Connection timeout" | Check if both devices on same WiFi; ping RPi from phone |
| "No audio captured" | Test mic in Voice Recorder app first; restart app |
| "RPi not receiving" | Check `tcpdump`; verify port 8080 is reachable |

---

## Future Enhancements

- [ ] Hostname resolution (mDNS on iOS)
- [ ] Connection quality indicator (latency, packet loss %)
- [ ] Recording toggle (save streamed audio to file)
- [ ] Multiple device targets (send to multiple RPis)
- [ ] Wake-lock to prevent app from suspending during use

---

## References

- **Flutter**: https://flutter.dev/docs
- **record package**: https://pub.dev/packages/record
- **GetX state management**: https://pub.dev/packages/get
- **Raspberry Pi main.py**: [../src/main.py](../src/main.py)

---

## Support

For issues or questions:

1. Check Flutter app logs: `flutter run -v`
2. Check RPi logs: `python -c "import sys; sys.stderr.write('test')" 2>&1 | tail`
3. Review [../CLAUDE.md](../CLAUDE.md) for project architecture
4. Test with `python send.py` as a baseline



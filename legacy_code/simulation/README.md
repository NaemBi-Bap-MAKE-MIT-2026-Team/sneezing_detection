# RPi4 Sneeze Detection Simulation (v4 TFLite Pipeline)

Raspberry Pi 4 (2GB RAM) 환경에서 v4 TFLite 재채기 감지 파이프라인의 **리소스 사용량과 실행 시간을 시뮬레이션**하기 위한 독립 테스트 환경입니다.

실제 RPi4 하드웨어나 TFLite 모델 파일 없이도 데스크톱에서 파이프라인의 실행 가능성(viability)을 검증할 수 있습니다.

## 디렉토리 구조

```
legacy_code/simulation/
├── __init__.py                    # 패키지 초기화
├── mock_tflite_interpreter.py     # TFLite Interpreter Mock
├── rpi4_resource_simulator.py     # RPi4 리소스 시뮬레이터
├── v4_pipeline.py                 # v4 전처리 파이프라인 함수
├── run_simulation.py              # CLI 시뮬레이션 실행기
└── README.md
```

## 각 모듈 설명

### `mock_tflite_interpreter.py`

`tflite_runtime.interpreter.Interpreter`와 동일한 API를 가진 **MockInterpreter** 클래스.

- `allocate_tensors()`, `get_input_details()`, `get_output_details()`, `set_tensor()`, `invoke()`, `get_tensor()` 지원
- 결정론적 fake probability 값 반환 (단일 값 또는 리스트로 순차 반환)
- `invoke()` 시 configurable delay로 RPi4 추론 지연 시뮬레이션
- input shape/dtype 검증 포함

### `rpi4_resource_simulator.py`

3개의 시뮬레이션 컴포넌트:

| 클래스 | 역할 |
|--------|------|
| **CPUThrottler** | 파이프라인 단계별 ARM Cortex-A72 속도 시뮬레이션. `resample` 7x, `melspectrogram` 5x, `tflite_invoke` 4x 등의 slowdown factor 적용 |
| **MemoryMonitor** | `tracemalloc` 기반 메모리 추적. RPi4 2GB 한도 대비 사용량 모니터링 및 경고 |
| **TimingProfiler** | 단계별 데스크톱 실측 시간 vs RPi4 추정 시간 비교. 시간 budget(기본 1000ms) 초과 여부 판정 |

### `v4_pipeline.py`

`legacy_code/output/v4/test_saving_v4.py`에서 추출한 전처리 함수들 (독립 실행 가능, `sounddevice` 등 하드웨어 의존성 없음):

- `rms()` — RMS 계산
- `normalize_rms()` — RMS 정규화
- `logmel()` — log-mel spectrogram (center=False, v4 학습 설정과 동일)
- `preproc()` — 전체 전처리 파이프라인 (pad/trim -> RMS norm -> logmel -> z-score)
- `resample_48k_to_16k()` — 리샘플링
- `create_mock_stats()` — mock 정규화 통계 생성 (mu=0, sd=1)
- `load_stats()` — `.npz` 파일 로드 (없으면 mock으로 fallback)
- `MockLiteModel` — MockInterpreter 기반 LiteModel 대체 클래스

### `run_simulation.py`

CLI 실행기. `.wav` 파일을 입력받아 전체 v4 파이프라인을 실행하고 리포트를 출력합니다.

## 사용법

### 기본 실행

```bash
python -m legacy_code.simulation.run_simulation \
    --wav-dir /path/to/wav/files/
```

### 전체 옵션

```bash
python -m legacy_code.simulation.run_simulation \
    --wav-dir /path/to/wav/files/ \
    --fake-output 0.95 \
    --inference-delay-ms 10.0 \
    --rpi4-ram 2048 \
    --budget-ms 1000 \
    --repeat 5 \
    --no-throttle
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--wav-dir` | (필수) | 테스트 `.wav` 파일이 있는 디렉토리 |
| `--stats-path` | None | `v4_norm_stats.npz` 경로. 생략 시 mock stats 사용 |
| `--fake-output` | 0.95 | Mock 모델이 반환할 probability 값 |
| `--inference-delay-ms` | 10.0 | Mock 추론 지연 시간 (ms) |
| `--rpi4-ram` | 2048 | RPi4 총 RAM (MB) |
| `--budget-ms` | 1000.0 | 1회 추론 시간 budget (ms) |
| `--repeat` | 1 | 각 파일 반복 횟수 |
| `--no-throttle` | False | CPU throttling 비활성화 (타이밍만 기록) |

### 출력 예시

```
======================================================================
  RPi4 Sneeze Detection Simulation  (v4 pipeline)
======================================================================
  ...

  Stage                       Desktop (ms)   RPi4 Est (ms)   Factor
  ------------------------------------------------------------------
  file_io                             1.99            3.98      2.0x
  resample                            2.24           15.69      7.0x
  pad_trim                            0.01            0.02      2.0x
  normalize                           0.14            0.42      3.0x
  melspectrogram                      6.56           32.80      5.0x
  normalize                           0.05            0.14      3.0x
  tflite_invoke                      12.60           50.41      4.0x
  ------------------------------------------------------------------
  TOTAL                              23.59          103.45   budget
  Budget                                            1000.0     PASS

  ...
  VERDICT: VIABLE  (worst-case 205 ms <= 1000 ms budget)
======================================================================
```

## v4 파이프라인 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `CAPTURE_SR` | 48000 | 마이크 캡처 샘플레이트 |
| `MODEL_SR` | 16000 | 모델 입력 샘플레이트 |
| `CLIP_SECONDS` | 2.0 | 오디오 클립 길이 |
| `N_MELS` | 64 | Mel 필터뱅크 수 |
| `N_FFT` | 400 | FFT 윈도우 크기 |
| `HOP` | 160 | Hop 길이 |
| `CENTER` | False | FFT 센터링 (v4 학습과 동일) |
| `TARGET_RMS` | 0.1 | RMS 정규화 목표값 |

## 설계 원칙

- **기존 코드 무수정** — `legacy_code/output/` 및 `realtime_detection/`의 어떤 파일도 수정하지 않음
- **독립 실행** — `sounddevice`, TFLite runtime, 실제 모델 파일 불필요
- **필요 패키지** — `numpy`, `librosa`, `soundfile` (librosa 의존성으로 `numba` 필요)

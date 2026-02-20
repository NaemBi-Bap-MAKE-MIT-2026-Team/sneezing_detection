# Sneeze Detection Architecture Comparison Report

## 1. 실험 목적

본 실험의 목적은 Raspberry Pi 4 환경에서 재채기 감지를 수행할 때, 서로 다른 추론 구조들이 **반응 속도(latency)**, **CPU 사용량**, **추론 호출 횟수** 측면에서 어떤 차이를 가지는지를 정량적으로 비교하고, 실제 제품/데모에 적합한 아키텍처를 선택하는 것이다.

비교 대상은 다음 5가지 구조이다.

1. Pure Trigger 방식
2. Sliding Window (항상 추론, hop=1.0s)
3. Sliding Window (항상 추론, hop=0.5s)
4. Capture-type Hybrid (현재 코드 구조)
5. Burst-type Hybrid

---

## 2. 실험 환경

* 입력 오디오: sneeze_10s.wav (총 10초)
* 재채기 onset 시점: 5.0초
* 샘플레이트: 48kHz (전처리 후 16kHz)
* 윈도우 길이: 2.0초
* 특징 추출: log-mel spectrogram (center=False)
* 모델 추론: Mock inference (fake-low/high) + RPi4 CPU slowdown 시뮬레이션
* 추론 지연 시뮬레이션: 10 ms / inference

CPU 사용률(cpu%)는 다음과 같이 정의하였다.

* cpu% = (총 RPi4 추정 추론 시간 / 전체 오디오 길이) * 100

---

## 3. 실험 결과 요약

| Method         | Detected | Latency (s) | Infer Calls | CPU % |
| -------------- | -------- | ----------- | ----------- | ----- |
| Pure Trigger   | YES      | 1.900       | 1           | 8.6   |
| Sliding (1.0s) | YES      | 0.900       | 9           | 24.3  |
| Sliding (0.5s) | YES      | 0.400       | 17          | 45.2  |
| Capture-type   | YES      | 1.400       | 1           | 2.6   |
| Burst-type     | YES      | 0.500       | 7           | 19.6  |

---

## 4. 구조별 분석

### 4.1 Pure Trigger

* RMS 트리거 이후 미래 2초를 수집한 뒤 단 1회 추론
* 구조적으로 최소 2초 지연이 발생
* CPU 사용량은 낮지만 반응성이 매우 떨어짐

결론: 실시간 반응이 필요한 시스템에는 부적합

---

### 4.2 Sliding Window (항상 추론)

#### hop = 1.0s

* 평균 latency 약 0.9초
* CPU 사용률 약 24%
* 구조가 단순하고 예측 가능

#### hop = 0.5s

* 가장 빠른 반응(0.4초)
* CPU 사용률 45%로 매우 높음
* LCD, 오디오, 통신을 동시에 수행하기에는 부담

결론: 빠르지만 hop=0.5는 RPi4 기준으로 과부하 위험

---

### 4.3 Capture-type Hybrid (현재 구현)

* RMS 트리거 + 0.5초 프리롤 + 2초 수집 후 1회 추론
* CPU 사용률은 가장 낮음(2.6%)
* 구조적으로 1.4초 이상의 지연 발생

결론: 저부하이지만 반응성이 부족함

---

### 4.4 Burst-type Hybrid

* 평상시: RMS 계산만 수행
* RMS 트리거 시점부터 3초 동안만 슬라이딩 추론 수행 (hop=0.5)
* latency 약 0.5초
* CPU 사용률 약 20%

특징:

* Sliding(0.5)의 반응성을 유지
* Sliding(1.0)보다 빠름
* Capture/Trigger보다 훨씬 빠르면서 CPU는 제한적

결론: 반응 속도와 자원 사용의 균형이 가장 우수

---

## 5. 최종 결론

### 성능 관점 요약

* 가장 빠른 구조: Sliding 0.5s
* 가장 가벼운 구조: Capture-type
* 가장 균형 잡힌 구조: Burst-type Hybrid

from pathlib import Path
import argparse

import numpy as np
import librosa
import soundfile as sf


def make_sneeze_10s(
    input_wav: Path,
    output_wav: Path,
    onset_sec: float,
    total_sec: float = 10.0,
):
    # 원본 로드 (샘플레이트 유지)
    y, sr = librosa.load(str(input_wav), sr=None, mono=True)
    y = y.astype(np.float32)

    sneeze_len_sec = len(y) / sr

    if onset_sec < 0:
        raise ValueError("onset_sec must be >= 0")

    if onset_sec + sneeze_len_sec > total_sec:
        raise ValueError(
            f"sneeze does not fit: onset {onset_sec:.3f}s + "
            f"sneeze {sneeze_len_sec:.3f}s > total {total_sec:.3f}s"
        )

    # 샘플 단위 계산
    pre_silence_samples = int(round(onset_sec * sr))
    post_silence_sec = total_sec - onset_sec - sneeze_len_sec
    post_silence_samples = int(round(post_silence_sec * sr))

    pre_silence = np.zeros(pre_silence_samples, dtype=np.float32)
    post_silence = np.zeros(post_silence_samples, dtype=np.float32)

    y_out = np.concatenate([pre_silence, y, post_silence], axis=0)

    # 정확히 total_sec 맞추기(반올림 오차 보정)
    target_samples = int(round(total_sec * sr))
    if len(y_out) > target_samples:
        y_out = y_out[:target_samples]
    elif len(y_out) < target_samples:
        y_out = np.pad(y_out, (0, target_samples - len(y_out)))

    sf.write(str(output_wav), y_out, sr)

    print("DONE")
    print(f"  input          : {input_wav}")
    print(f"  output         : {output_wav}")
    print(f"  sample rate    : {sr} Hz")
    print(f"  total length   : {len(y_out)/sr:.3f} sec")
    print(f"  onset sec      : {onset_sec:.3f}")
    print(f"  sneeze length  : {sneeze_len_sec:.3f} sec")
    print(f"  post silence   : {post_silence_sec:.3f} sec")


def parse_args():
    p = argparse.ArgumentParser("Make 10s sneeze wav with known onset")
    p.add_argument("--input", type=Path, required=True, help="input sneeze.wav")
    p.add_argument("--output", type=Path, required=True, help="output 10s wav")
    p.add_argument("--onset-sec", type=float, required=True, help="sneeze onset time (sec)")
    p.add_argument("--total-sec", type=float, default=10.0, help="total output length (sec)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_sneeze_10s(
        input_wav=args.input,
        output_wav=args.output,
        onset_sec=args.onset_sec,
        total_sec=args.total_sec,
    )
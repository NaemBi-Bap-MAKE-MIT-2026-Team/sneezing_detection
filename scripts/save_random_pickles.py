import torchaudio
import torch

from tqdm import tqdm

import polars as pl
import random
import pickle
import time
import os

ESC50_AUDIO_PATH = "./esc-50/audio/"
ESC50_META_PATH = "./esc-50/meta/esc50.csv"

DATASET_AUDIO_PATH = "./datasets/"

random.seed(time.time())

def random_pickles():
    desc_csv = pl.read_csv(ESC50_META_PATH)
    count_of_sneeze_datasts = len(os.listdir(DATASET_AUDIO_PATH))

    filtered_desc_csv = desc_csv.filter(pl.col("category") != "sneezing")
    selected_rows = filtered_desc_csv.sample(n=count_of_sneeze_datasts, seed=42)

    not_sneeze_dataset = []
    target_length = 16000 * 2  # 2 seconds at 16000 Hz = 32000 samples

    for row in tqdm(selected_rows.iter_rows(), desc=f"Processing {count_of_sneeze_datasts} not-sneeze samples", total=count_of_sneeze_datasts):
        file_path = os.path.join(ESC50_AUDIO_PATH, row[0])
        audio, sr = torchaudio.load(file_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        
        if audio.shape[1] > target_length:
            audio = audio[:, :target_length]
        elif audio.shape[1] < target_length:
            pad_amount = target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        
        not_sneeze_dataset.append(audio)

    sneeze_dataset = []

    for filename in tqdm(os.listdir(DATASET_AUDIO_PATH), desc="Processing sneeze samples"):
        if filename.endswith(".wav"):
            file_path = os.path.join(DATASET_AUDIO_PATH, filename)
            audio, sr = torchaudio.load(file_path)
            
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
            
            if audio.shape[1] > target_length:
                audio = audio[:, :target_length]
            elif audio.shape[1] < target_length:
                pad_amount = target_length - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, pad_amount))
            
            sneeze_dataset.append(audio)

    return sneeze_dataset, not_sneeze_dataset



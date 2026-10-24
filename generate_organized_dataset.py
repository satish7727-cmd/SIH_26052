import os
import random
import glob
import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
NUM_SAMPLES = 40000            # Total samples
SAMPLE_RATE = 16000           # 16 kHz audio standard
DURATION_SEC = 3.0            # 3-second frames
TARGET_LEN = int(SAMPLE_RATE * DURATION_SEC)

OUTPUT_DIR = os.path.abspath("./military_anc_dataset")
DATASET_PATH = os.path.abspath("./LibriSpeechData")

SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]  # SNR range up to +20 dB

CATEGORIES = {
    "stationary": ["engine_rumble", "wind_turbulence", "generator_hum"],
    "non_stationary": ["drone_rotor", "siren_sweep", "radio_chatter"],
    "impulsive": ["gunshot_burst", "artillery_shock", "explosion_impact"]
}

def create_directory_structure():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "clean", "male_speech"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "clean", "female_speech"), exist_ok=True)
        for cat in CATEGORIES.keys():
            os.makedirs(os.path.join(OUTPUT_DIR, split, "noise", cat), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "mixed_noisy"), exist_ok=True)

# ---------------------------------------------------------
# NOISE GENERATORS
# ---------------------------------------------------------
def generate_stationary_noise(noise_type, duration_sec=3.0, sr=16000):
    total_samples = int(duration_sec * sr)
    if noise_type == "engine_rumble":
        signal = np.random.normal(0, 1, total_samples)
        signal = np.convolve(signal, np.ones(30)/30, mode='same')
    elif noise_type == "wind_turbulence":
        signal = np.random.normal(0, 1, total_samples)
        t = np.linspace(0, duration_sec, total_samples)
        signal *= (1.0 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    else:  # generator_hum
        t = np.linspace(0, duration_sec, total_samples)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        signal += np.random.normal(0, 0.1, total_samples)
    return signal / np.max(np.abs(signal))

def generate_non_stationary_noise(noise_type, duration_sec=3.0, sr=16000):
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples)
    if noise_type == "drone_rotor":
        fundamental = random.choice([100, 140, 180])
        signal = np.sin(2 * np.pi * fundamental * t)
        for h in range(2, 5):
            signal += (1.0 / h) * np.sin(2 * np.pi * fundamental * h * t)
        signal += np.random.normal(0, 0.2, total_samples)
    elif noise_type == "siren_sweep":
        freq_sweep = 400 + 400 * np.sin(2 * np.pi * 1.5 * t)
        signal = np.sin(2 * np.pi * freq_sweep * t)
    else:  # radio_chatter
        signal = np.random.normal(0, 1, total_samples)
        mask = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)
        signal *= mask
    return signal / np.max(np.abs(signal))

def generate_impulsive_noise(noise_type, duration_sec=3.0, sr=16000):
    total_samples = int(duration_sec * sr)
    signal = np.zeros(total_samples)
    if noise_type == "gunshot_burst":
        num_shots = random.randint(2, 5)
        for _ in range(num_shots):
            start = random.randint(0, max(0, total_samples - 3000))
            length = random.randint(1000, 3000)
            t = np.linspace(0, 1, length)
            signal[start:start+length] += np.random.normal(0, 1, length) * np.exp(-12 * t)
    elif noise_type == "artillery_shock":
        start = random.randint(0, max(0, total_samples - 8000))
        length = random.randint(4000, 8000)
        t = np.linspace(0, 1, length)
        low_wave = np.sin(2 * np.pi * 40 * t)
        signal[start:start+length] += (np.random.normal(0, 1, length) + low_wave) * np.exp(-5 * t)
    else:  # explosion_impact
        start = random.randint(0, max(0, total_samples - 10000))
        length = random.randint(6000, 10000)
        t = np.linspace(0, 1, length)
        signal[start:start+length] += np.random.normal(0, 1.5, length) * np.exp(-3 * t)
    return signal / (np.max(np.abs(signal)) + 1e-8)

def mix_speech_and_noise(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    scale = np.sqrt((clean_power / (10 ** (snr_db / 10.0))) / noise_power)
    noisy = clean + (noise * scale)
    
    max_val = np.max(np.abs(noisy))
    if max_val > 0.99:
        noisy = noisy / max_val * 0.95
        clean = clean / max_val * 0.95
    return noisy, clean

def main():
    create_directory_structure()
    
    print("=== Scanning Extracted Audio Files ===")
    flac_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.flac"), recursive=True)
    if not flac_files:
        raise FileNotFoundError(f"No .flac files found in {DATASET_PATH}. Did you extract the tar archive in Step 2?")

    print(f"Found {len(flac_files)} clean audio files.")

    metadata = []
    print(f"\n=== Generating {NUM_SAMPLES} Structured Samples ===")

    for i in tqdm(range(NUM_SAMPLES)):
        split = "train" if i < int(NUM_SAMPLES * 0.8) else "val"
        
        file_path = random.choice(flac_files)
        speech, sr = sf.read(file_path)
        
        # Determine speaker ID from filename path
        parts = file_path.split(os.sep)
        speaker_id = 1
        for part in parts:
            if part.isdigit():
                speaker_id = int(part)
                break
        gender_dir = "female_speech" if speaker_id % 2 == 0 else "male_speech"

        if len(speech) < TARGET_LEN:
            speech = np.pad(speech, (0, TARGET_LEN - len(speech)))
        else:
            start_f = random.randint(0, len(speech) - TARGET_LEN)
            speech = speech[start_f:start_f + TARGET_LEN]

        noise_category = random.choice(list(CATEGORIES.keys()))
        noise_type = random.choice(CATEGORIES[noise_category])

        if noise_category == "stationary":
            noise = generate_stationary_noise(noise_type, DURATION_SEC, SAMPLE_RATE)
        elif noise_category == "non_stationary":
            noise = generate_non_stationary_noise(noise_type, DURATION_SEC, SAMPLE_RATE)
        else:
            noise = generate_impulsive_noise(noise_type, DURATION_SEC, SAMPLE_RATE)

        snr = random.choice(SNR_LEVELS)
        noisy, clean = mix_speech_and_noise(speech, noise, snr)

        clean_rel_path = os.path.join(split, "clean", gender_dir, f"sample_{i:05d}_clean.wav")
        noise_rel_path = os.path.join(split, "noise", noise_category, f"sample_{i:05d}_noise.wav")
        mixed_rel_path = os.path.join(split, "mixed_noisy", f"sample_{i:05d}_snr{snr}_{noise_type}.wav")

        sf.write(os.path.join(OUTPUT_DIR, clean_rel_path), clean, SAMPLE_RATE)
        sf.write(os.path.join(OUTPUT_DIR, noise_rel_path), noise, SAMPLE_RATE)
        sf.write(os.path.join(OUTPUT_DIR, mixed_rel_path), noisy, SAMPLE_RATE)

        metadata.append({
            "sample_id": i,
            "split": split,
            "gender": gender_dir.replace("_speech", ""),
            "noise_category": noise_category,
            "noise_type": noise_type,
            "snr_db": snr,
            "clean_file": clean_rel_path,
            "noise_file": noise_rel_path,
            "mixed_file": mixed_rel_path
        })

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_DIR, "dataset_manifest.csv"), index=False)
    print("\n Clean, Noise, and Mixed subfolders generated successfully.")

if __name__ == "__main__":
    main()

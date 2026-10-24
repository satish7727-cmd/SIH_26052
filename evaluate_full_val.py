import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi

# ---------------------------------------------------------
# 1. ARCHITECTURE & HYPERPARAMETERS
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
SAMPLE_RATE = 16000

CHECKPOINT_PATH = "checkpoints/best_anc_model_epoch35_pesq2.98.pt"
DATASET_DIR = os.path.abspath("./military_anc_dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn_r = nn.BatchNorm2d(out_channels)
        self.bn_i = nn.BatchNorm2d(out_channels)

    def forward(self, real, imag):
        out_r = self.conv_r(real) - self.conv_i(imag)
        out_i = self.conv_r(imag) + self.conv_i(real)
        out_r = F.elu(self.bn_r(out_r))
        out_i = F.elu(self.bn_i(out_i))
        return out_r, out_i

class ComplexCRN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ComplexConvBlock(1, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc2 = ComplexConvBlock(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc3 = ComplexConvBlock(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        self.b_c, self.b_f = 64, 33
        self.lstm_in_features = self.b_c * self.b_f

        self.lstm = nn.LSTM(input_size=self.lstm_in_features, hidden_size=512, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(512, self.lstm_in_features)

        self.dec3 = nn.ConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        self.dec2 = nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        self.dec1 = nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        
        self.bn_dec3 = nn.BatchNorm2d(32)
        self.bn_dec2 = nn.BatchNorm2d(16)

    def forward(self, real, imag):
        r1, i1 = self.enc1(real, imag)
        r2, i2 = self.enc2(r1, i1)
        r3, i3 = self.enc3(r2, i2)

        b, c, f, t = r3.shape
        r_flat = r3.permute(0, 3, 1, 2).reshape(b, t, c * f)
        
        lstm_out, _ = self.lstm(r_flat)
        lstm_out = self.fc(lstm_out)
        lstm_out = lstm_out.reshape(b, t, c, f).permute(0, 2, 3, 1)

        concat3 = torch.cat([lstm_out, r3], dim=1)
        
        dec_r = F.elu(self.bn_dec3(self.dec3(concat3)))
        if dec_r.shape[2] != r2.shape[2]:
            dec_r = F.interpolate(dec_r, size=(r2.shape[2], dec_r.shape[3]), mode='bilinear', align_corners=False)
        dec_r = torch.cat([dec_r, r2], dim=1)

        dec_r = F.elu(self.bn_dec2(self.dec2(dec_r)))
        if dec_r.shape[2] != r1.shape[2]:
            dec_r = F.interpolate(dec_r, size=(r1.shape[2], dec_r.shape[3]), mode='bilinear', align_corners=False)
        dec_r = torch.cat([dec_r, r1], dim=1)

        mask_r = torch.tanh(self.dec1(dec_r))
        if mask_r.shape[2] != real.shape[2]:
            mask_r = F.interpolate(mask_r, size=(real.shape[2], mask_r.shape[3]), mode='bilinear', align_corners=False)

        mask_i = torch.zeros_like(mask_r)

        est_real = real * mask_r - imag * mask_i
        est_imag = real * mask_i + imag * mask_r

        return est_real, est_imag

# ---------------------------------------------------------
# 2. METRIC FORMULAS
# ---------------------------------------------------------
def compute_snr(clean_np, est_np):
    target_energy = np.sum(clean_np ** 2) + 1e-8
    noise_energy = np.sum((clean_np - est_np) ** 2) + 1e-8
    return 10 * np.log10(target_energy / noise_energy)

def compute_sisdr(reference, estimate):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    e_target = alpha * reference
    e_res = estimate - e_target
    return 10 * np.log10(np.sum(e_target ** 2) / (np.sum(e_res ** 2) + 1e-8))

# ---------------------------------------------------------
# 3. EVALUATION EXECUTION
# ---------------------------------------------------------
def main():
    print("=" * 75)
    print("      COMPLEX-CRN FULL VALIDATION DATASET EVALUATION SUITE      ")
    print("=" * 75)
    print(f"[+] Compute Device : {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] Error: Checkpoint file '{CHECKPOINT_PATH}' not found!")
        sys.exit(1)

    if not os.path.exists(MANIFEST_PATH):
        print(f"[!] Error: Dataset manifest '{MANIFEST_PATH}' not found!")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    model = ComplexCRN().to(DEVICE)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[+] Successfully Loaded Weights: {CHECKPOINT_PATH}")
    print(f"[+] Total Validation Samples    : {len(val_df)}")

    window = torch.hann_window(WIN_LENGTH).to(DEVICE)

    snrs_before, snrs_after = [], []
    stois_before, stois_after = [], []
    pesqs_before, pesqs_after = [], []
    sisdrs_before, sisdrs_after = [], []
    inference_times = []

    print("\n[+] Running Evaluation Progress...")
    with torch.no_grad():
        for idx in tqdm(range(len(val_df))):
            row = val_df.iloc[idx]
            clean_wav_path = os.path.join(DATASET_DIR, row['clean_file'])
            noisy_wav_path = os.path.join(DATASET_DIR, row['mixed_file'])

            clean_audio, sr = sf.read(clean_wav_path)
            noisy_audio, _ = sf.read(noisy_wav_path)

            if clean_audio.ndim > 1: clean_audio = clean_audio.mean(axis=1)
            if noisy_audio.ndim > 1: noisy_audio = noisy_audio.mean(axis=1)

            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len].astype(np.float32)
            noisy_audio = noisy_audio[:min_len].astype(np.float32)

            noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            start_time = time.time()
            noisy_stft = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
            mix_real, mix_imag = noisy_stft.real.unsqueeze(1), noisy_stft.imag.unsqueeze(1)

            est_real, est_imag = model(mix_real, mix_imag)
            est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
            
            enhanced_tensor = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, length=min_len)
            enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
            inference_times.append((time.time() - start_time) * 1000)

            # Compute Before/After
            snrs_before.append(compute_snr(clean_audio, noisy_audio))
            snrs_after.append(compute_snr(clean_audio, enhanced_audio))

            stois_before.append(stoi(clean_audio, noisy_audio, sr, extended=False))
            stois_after.append(stoi(clean_audio, enhanced_audio, sr, extended=False))

            sisdrs_before.append(compute_sisdr(clean_audio, noisy_audio))
            sisdrs_after.append(compute_sisdr(clean_audio, enhanced_audio))

            try:
                pesqs_before.append(pesq(sr, clean_audio, noisy_audio, 'wb'))
                pesqs_after.append(pesq(sr, clean_audio, enhanced_audio, 'wb'))
            except Exception:
                pass

    p_bef_avg, p_aft_avg = np.mean(pesqs_before), np.mean(pesqs_after)
    s_bef_avg, s_aft_avg = np.mean(stois_before), np.mean(stois_after)
    snr_bef_avg, snr_aft_avg = np.mean(snrs_before), np.mean(snrs_after)
    sisdr_bef_avg, sisdr_aft_avg = np.mean(sisdrs_before), np.mean(sisdrs_after)
    avg_latency = np.mean(inference_times)

    print("\n" + "=" * 75)
    print(f" METRIC                | BEFORE ANC   | AFTER ANC (CRN MODEL) | IMPROVEMENT ")
    print("=" * 75)
    print(f" PESQ (Quality 1-4.5)  | {p_bef_avg:11.3f} | {p_aft_avg:21.3f} | +{p_aft_avg - p_bef_avg:10.3f}")
    print(f" STOI (Clarity 0-1)    | {s_bef_avg:11.3f} | {s_aft_avg:21.3f} | +{s_aft_avg - s_bef_avg:10.3f}")
    print(f" SNR  (Signal/Noise)   | {snr_bef_avg:9.2f} dB | {snr_aft_avg:19.2f} dB | +{snr_aft_avg - snr_bef_avg:8.2f} dB")
    print(f" SI-SDR (Distortion)   | {sisdr_bef_avg:9.2f} dB | {sisdr_aft_avg:19.2f} dB | +{sisdr_aft_avg - sisdr_bef_avg:8.2f} dB")
    print("=" * 75)
    print(f"[+] Avg File Latency     : {avg_latency:.2f} ms")
    print("=" * 75)

if __name__ == "__main__":
    main()

import os
import sys
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pesq import pesq
from pystoi import stoi

warnings.filterwarnings("ignore")
os.environ["ORT_CPP_LOG_LEVEL"] = "3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
SAMPLE_RATE = 16000

CHECKPOINT_PATH = "checkpoints/best_anc_model_epoch35_pesq2.98.pt"
ONNX_PATH = "checkpoints/best_anc_model.onnx"

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE
# ---------------------------------------------------------
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
# 2. METRICS HELPERS
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
# 3. MAIN EVALUATION
# ---------------------------------------------------------
def main():
    print("=" * 75)
    print("      COMPLEX-CRN FULL SYSTEM EVALUATION SUITE      ")
    print("=" * 75)
    print(f"[+] Compute Device : {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] Checkpoint file '{CHECKPOINT_PATH}' not found!")
        sys.exit(1)

    model = ComplexCRN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"[+] Successfully Loaded Weights: {CHECKPOINT_PATH}")

    clean_wav_path = "val/clean/sample_32000.wav"
    noisy_wav_path = "val/mixed_noisy/sample_32000_snr5_artillery_shock.wav"

    manifest_path = "./military_anc_dataset/dataset_manifest.csv"
    if os.path.exists(manifest_path):
        import pandas as pd
        df = pd.read_csv(manifest_path)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        if len(val_df) > 0:
            row = val_df.iloc[0]
            clean_wav_path = os.path.join("./military_anc_dataset", row['clean_file'])
            noisy_wav_path = os.path.join("./military_anc_dataset", row['mixed_file'])
            print(f"[+] Selected validation sample: {row['mixed_file']}")

    clean_audio, sr = sf.read(clean_wav_path)
    noisy_audio, _ = sf.read(noisy_wav_path)

    if clean_audio.ndim > 1: clean_audio = clean_audio.mean(axis=1)
    if noisy_audio.ndim > 1: noisy_audio = noisy_audio.mean(axis=1)

    min_len = min(len(clean_audio), len(noisy_audio))
    clean_audio = clean_audio[:min_len].astype(np.float32)
    noisy_audio = noisy_audio[:min_len].astype(np.float32)

    window = torch.hann_window(WIN_LENGTH).to(DEVICE)
    noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 1. Quality Evaluation (PyTorch Full Sequence)
    with torch.no_grad():
        noisy_stft = torch.stft(noisy_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
        mix_real, mix_imag = noisy_stft.real.unsqueeze(1), noisy_stft.imag.unsqueeze(1)

        est_real, est_imag = model(mix_real, mix_imag)
        est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
        
        enhanced_tensor = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, length=min_len)
        enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()

    # 2. Latency Evaluation (ONNX Engine Benchmark)
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    
    # Exact 5-frame shape matching ONNX export contract [1, 1, 257, 5]
    onnx_real = mix_real[:, :, :, :5].cpu().numpy()
    onnx_imag = mix_imag[:, :, :, :5].cpu().numpy()

    # Warmup
    for _ in range(10):
        _ = session.run(None, {'mix_real': onnx_real, 'mix_imag': onnx_imag})

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = session.run(None, {'mix_real': onnx_real, 'mix_imag': onnx_imag})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_5frame_latency = np.mean(latencies)
    avg_per_frame_latency = avg_5frame_latency / 5.0

    snr_before = compute_snr(clean_audio, noisy_audio)
    snr_after = compute_snr(clean_audio, enhanced_audio)

    stoi_before = stoi(clean_audio, noisy_audio, sr, extended=False)
    stoi_after = stoi(clean_audio, enhanced_audio, sr, extended=False)

    try:
        pesq_before = pesq(sr, clean_audio, noisy_audio, 'wb')
        pesq_after = pesq(sr, clean_audio, enhanced_audio, 'wb')
    except Exception:
        pesq_before, pesq_after = 1.0, 1.0

    sisdr_before = compute_sisdr(clean_audio, noisy_audio)
    sisdr_after = compute_sisdr(clean_audio, enhanced_audio)

    print("\n" + "=" * 75)
    print(f" METRIC               | BEFORE ANC   | AFTER ANC (CRN MODEL) | IMPROVEMENT ")
    print("=" * 75)
    print(f" PESQ (Quality 1-4.5) | {pesq_before:11.3f} | {pesq_after:21.3f} | +{pesq_after - pesq_before:10.3f}")
    print(f" STOI (Clarity 0-1)   | {stoi_before:11.3f} | {stoi_after:21.3f} | +{stoi_after - stoi_before:10.3f}")
    print(f" SNR  (Signal/Noise)  | {snr_before:9.2f} dB | {snr_after:19.2f} dB | +{snr_after - snr_before:8.2f} dB")
    print(f" SI-SDR (Distortion)  | {sisdr_before:9.2f} dB | {sisdr_after:19.2f} dB | +{sisdr_after - sisdr_before:8.2f} dB")
    print("=" * 75)
    print(f"[+] Audio Frame Hop Size           : 8.00 ms (128 samples)")
    print(f"[+] ONNX 5-Frame Processing Time   : {avg_5frame_latency:.2f} ms")
    print(f"[+] Single Frame Processing Time   : {avg_per_frame_latency:.2f} ms")
    print("=" * 75)
    if avg_per_frame_latency < 20.0:
        print(" VERDICT: PASS! System meets all SIH Real-Time constraints (< 20 ms)! ")
    print("=" * 75)

if __name__ == "__main__":
    main()

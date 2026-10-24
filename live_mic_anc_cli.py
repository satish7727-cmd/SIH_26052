import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyaudio

# --------------------------------------------------
# MODEL V2 ARCHITECTURE (DTLNNoiseCanceller)
# --------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels)
        )
        self.activation = nn.PReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class DTLNNoiseCanceller(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 256
        self.hop_length = 128
        self.win_length = 256
        self.freq_bins = self.n_fft // 2 + 1

        self.encoder = nn.Sequential(
            nn.Conv1d(self.freq_bins, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.PReLU(),
            nn.Conv1d(128, 96, kernel_size=5, padding=2),
            nn.GroupNorm(8, 96),
            nn.PReLU(),
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )

        self.temporal = nn.Sequential(
            ResidualBlock(64, 1),
            ResidualBlock(64, 2),
            ResidualBlock(64, 4),
            ResidualBlock(64, 8),
            ResidualBlock(64, 16)
        )

        self.mask_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        self.real_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        self.imag_head = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(64, self.freq_bins, kernel_size=1)
        )

        self.vad_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, noisy):
        original_length = noisy.shape[-1]
        x = noisy.squeeze(1)

        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, return_complex=True
        )

        noisy_real, noisy_imag = spec.real, spec.imag
        magnitude = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2 + 1e-8)

        log_mag = torch.log1p(magnitude)
        features = self.encoder(log_mag)
        temporal = self.temporal(features)

        mask = torch.sigmoid(self.mask_head(temporal))
        mask = 0.02 + 0.98 * mask
        enhanced_mag = magnitude * mask

        real_residual = torch.tanh(self.real_head(temporal))
        imag_residual = torch.tanh(self.imag_head(temporal))
        correction_scale = 0.15 * magnitude

        enhanced_real = (enhanced_mag * noisy_real / (magnitude + 1e-8)) + (correction_scale * real_residual)
        enhanced_imag = (enhanced_mag * noisy_imag / (magnitude + 1e-8)) + (correction_scale * imag_residual)

        enhanced_complex_mag = torch.sqrt(enhanced_real ** 2 + enhanced_imag ** 2 + 1e-8)
        max_mag = magnitude * 1.05
        scale = torch.minimum(torch.ones_like(enhanced_complex_mag), max_mag / (enhanced_complex_mag + 1e-8))

        enhanced_real *= scale
        enhanced_imag *= scale

        enhanced_spec = torch.complex(enhanced_real, enhanced_imag)
        enhanced = torch.istft(
            enhanced_spec, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, length=original_length
        )

        vad_low = self.vad_head(temporal)
        vad_probability = torch.sigmoid(
            F.interpolate(vad_low, size=original_length, mode="linear", align_corners=False)
        )

        return enhanced.unsqueeze(1), vad_probability

# --------------------------------------------------
# NOISE CLASSIFIER
# --------------------------------------------------

def classify_noise(audio_chunk):
    fft_vals = np.abs(np.fft.rfft(audio_chunk))
    energy = np.mean(audio_chunk ** 2)
    if energy < 0.0005:
        return "CLEAN / SILENT"
    low_freq_ratio = np.sum(fft_vals[:20]) / (np.sum(fft_vals) + 1e-6)
    peak_ratio = np.max(fft_vals) / (np.mean(fft_vals) + 1e-6)
    
    if energy > 0.04 and peak_ratio > 10:
        return "ARTILLERY / EXPLOSIVE SHOCKWAVE"
    elif low_freq_ratio > 0.55:
        return "HEAVY TANK / VEHICLE ENGINE"
    elif peak_ratio > 6:
        return "HELICOPTER ROTOR NOISE"
    else:
        return "WHITE / AMBIENT BATTLEFIELD NOISE"

# --------------------------------------------------
# LIVE STREAMING PROCESSOR
# --------------------------------------------------

def main():
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024  # Low buffer size (~64ms) for real-time responsiveness

    print("=" * 65)
    print("   LIVE TERMINAL ANC ENGINE: MIC INPUT -> MODEL V2 -> SPEAKER   ")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Hardware Device : {device}")

    model = DTLNNoiseCanceller().to(device)
    
    checkpoint_path = "checkpoints/best_anc_model_epoch35_pesq2.98.pt"
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 100:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print(f"[+] Loaded weights from: {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Partial loading: {e}")
    else:
        print("[WARN] Running with initialized V2 weights.")

    model.eval()

    # Initialize PyAudio Streams
    p = pyaudio.PyAudio()

    try:
        in_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        out_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"\n[!] Audio Driver Error: {e}")
        print("[!] Ensure headphones are plugged in to prevent audio feedback loop.")
        p.terminate()
        sys.exit(1)

    print("\n[+] LIVE PROCESSING STARTED! Press Ctrl+C to stop.")
    print("💡 TIP: Wear headphones to avoid microphone feedback loop!\n")

    frame_count = 0

    try:
        while True:
            # 1. Read audio bytes from microphone
            raw_data = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
            
            # 2. Normalize to float [-1.0, 1.0]
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # 3. Model Inference
            t0 = time.time()
            input_tensor = torch.tensor(audio_float, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                enhanced_tensor, vad_tensor = model(input_tensor)

            t1 = time.time()
            latency_ms = (t1 - t0) * 1000

            # 4. Extract Output Audio & Metrics
            cleaned_float = enhanced_tensor.squeeze().cpu().numpy()
            vad_score = vad_tensor.squeeze().cpu().numpy().mean() * 100.0
            noise_label = classify_noise(audio_float)

            # 5. Convert back to int16 bytes and write to laptop speakers
            cleaned_int16 = (np.clip(cleaned_float, -1.0, 1.0) * 32767).astype(np.int16)
            out_stream.write(cleaned_int16.tobytes())

            frame_count += 1
            # Terminal dashboard printout
            sys.stdout.write(
                f"\r[STREAMING] Frame: {frame_count:05d} | Latency: {latency_ms:5.2f}ms | VAD: {vad_score:5.1f}% | Noise: {noise_label:<32}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[+] Stopping Live ANC Engine...")
    finally:
        in_stream.stop_stream()
        in_stream.close()
        out_stream.stop_stream()
        out_stream.close()
        p.terminate()
        print("[+] Audio streams safely closed.")

if __name__ == "__main__":
    main()

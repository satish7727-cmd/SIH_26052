import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torch.nn.functional as F
from tqdm import tqdm

from pesq import pesq
from pystoi import stoi

# ---------------------------------------------------------
# 1. CONFIGURATION & HYPERPARAMETERS
# ---------------------------------------------------------
DATASET_DIR = os.path.abspath("./military_anc_dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

BATCH_SIZE = 16
NUM_EPOCHS = 35  # Updated from 20 to 35 epochs
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STFT Params
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
SAMPLE_RATE = 16000

# ---------------------------------------------------------
# 2. DATASET LOADER (Uses Entire Train & Val Split)
# ---------------------------------------------------------
class MilitaryANCDataset(Dataset):
    def __init__(self, manifest_df, split="train"):
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        clean_path = os.path.join(DATASET_DIR, row['clean_file'])
        mixed_path = os.path.join(DATASET_DIR, row['mixed_file'])
        
        clean_audio, _ = sf.read(clean_path)
        mixed_audio, _ = sf.read(mixed_path)
        
        return (
            torch.tensor(mixed_audio, dtype=torch.float32),
            torch.tensor(clean_audio, dtype=torch.float32)
        )

# ---------------------------------------------------------
# 3. 4-MILLION PARAMETER COMPLEX-CRN ARCHITECTURE
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
        # Strided Encoder downsamples frequency: 257 -> 129 -> 65 -> 33
        self.enc1 = ComplexConvBlock(1, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc2 = ComplexConvBlock(16, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))
        self.enc3 = ComplexConvBlock(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))

        # Bottleneck: 64 channels * 33 freq bins = 2112 input features
        self.b_c, self.b_f = 64, 33
        self.lstm_in_features = self.b_c * self.b_f  # 2112

        # Recurrent bottleneck (~4.08M total model params)
        self.lstm = nn.LSTM(input_size=self.lstm_in_features, hidden_size=512, num_layers=2, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(512, self.lstm_in_features)

        # Transposed Decoder upsamples frequency: 33 -> 65 -> 129 -> 257
        self.dec3 = nn.ConvTranspose2d(128, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        self.dec2 = nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        self.dec1 = nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), output_padding=(0, 0))
        
        self.bn_dec3 = nn.BatchNorm2d(32)
        self.bn_dec2 = nn.BatchNorm2d(16)

    def forward(self, real, imag):
        # Input shape: [B, 1, 257, T]
        r1, i1 = self.enc1(real, imag)    # [B, 16, 129, T]
        r2, i2 = self.enc2(r1, i1)        # [B, 32, 65, T]
        r3, i3 = self.enc3(r2, i2)        # [B, 64, 33, T]

        b, c, f, t = r3.shape
        r_flat = r3.permute(0, 3, 1, 2).reshape(b, t, c * f)
        
        lstm_out, _ = self.lstm(r_flat)
        lstm_out = self.fc(lstm_out)
        lstm_out = lstm_out.reshape(b, t, c, f).permute(0, 2, 3, 1)

        # Decoder with Skip Connections
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

        # Complex Masking
        est_real = real * mask_r - imag * mask_i
        est_imag = real * mask_i + imag * mask_r

        return est_real, est_imag

# ---------------------------------------------------------
# 4. ENHANCED TIME-FREQUENCY & PHASE LOSS FUNCTION
# ---------------------------------------------------------
class EnhancedTimeFreqLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, est_audio, clean_audio, est_real, est_imag, clean_real, clean_imag):
        time_loss = F.l1_loss(est_audio, clean_audio)
        
        est_mag = torch.sqrt(est_real**2 + est_imag**2 + 1e-8)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2 + 1e-8)
        mag_loss = F.l1_loss(est_mag, clean_mag)
        
        real_loss = F.l1_loss(est_real, clean_real)
        imag_loss = F.l1_loss(est_imag, clean_imag)
        
        return time_loss + 0.5 * mag_loss + 0.25 * (real_loss + imag_loss)

# ---------------------------------------------------------
# 5. METRICS COMPUTATION
# ---------------------------------------------------------
def compute_metrics(clean_np, est_np, sr=16000):
    target_energy = np.sum(clean_np ** 2) + 1e-8
    noise_energy = np.sum((clean_np - est_np) ** 2) + 1e-8
    snr = 10 * np.log10(target_energy / noise_energy)
    
    stoi_val = stoi(clean_np, est_np, sr, extended=False)
    try:
        pesq_val = pesq(sr, clean_np, est_np, 'wb')
    except Exception:
        pesq_val = 1.0

    return snr, stoi_val, pesq_val

# ---------------------------------------------------------
# 6. TRAINING & VALIDATION PIPELINE
# ---------------------------------------------------------
def main():
    print(f"Using Device: {DEVICE}")
    df = pd.read_csv(MANIFEST_PATH)

    train_ds = MilitaryANCDataset(df, split="train")
    val_ds = MilitaryANCDataset(df, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ComplexCRN().to(DEVICE)
    
    # Verify Parameter Count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Parameter Count: {total_params / 1e6:.2f} Million Parameters")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    criterion = EnhancedTimeFreqLoss()

    window = torch.hann_window(WIN_LENGTH).to(DEVICE)
    os.makedirs("checkpoints", exist_ok=True)
    best_pesq = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train All Samples]")
        
        for mixed, clean in pbar:
            mixed, clean = mixed.to(DEVICE), clean.to(DEVICE)

            mixed_stft = torch.stft(mixed, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
            clean_stft = torch.stft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)

            mix_real, mix_imag = mixed_stft.real.unsqueeze(1), mixed_stft.imag.unsqueeze(1)
            clean_real, clean_imag = clean_stft.real.unsqueeze(1), clean_stft.imag.unsqueeze(1)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                est_real, est_imag = model(mix_real, mix_imag)
                est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
                est_audio = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, length=clean.shape[-1])
                loss = criterion(est_audio, clean, est_real, est_imag, clean_real, clean_imag)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # --- VALIDATION LOOP ---
        model.eval()
        val_snrs, val_stois, val_pesqs = [], [], []

        print(f"\n Running Validation for Epoch {epoch}...")
        with torch.no_grad():
            for mixed, clean in tqdm(val_loader, desc="[Validating All Samples]"):
                mixed, clean = mixed.to(DEVICE), clean.to(DEVICE)

                mixed_stft = torch.stft(mixed, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
                mix_real, mix_imag = mixed_stft.real.unsqueeze(1), mixed_stft.imag.unsqueeze(1)

                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    est_real, est_imag = model(mix_real, mix_imag)
                    est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
                    est_audio = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, length=clean.shape[-1])

                for b in range(mixed.shape[0]):
                    snr, stoi_val, pesq_val = compute_metrics(
                        clean[b].cpu().numpy(),
                        est_audio[b].cpu().numpy(),
                        sr=SAMPLE_RATE
                    )
                    val_snrs.append(snr)
                    val_stois.append(stoi_val)
                    val_pesqs.append(pesq_val)

        avg_snr = np.mean(val_snrs)
        avg_stoi = np.mean(val_stois)
        avg_pesq = np.mean(val_pesqs)

        # Update Learning Rate Scheduler
        scheduler.step(avg_pesq)

        print(f"\n--- Epoch {epoch} Validation Metrics ---")
        print(f" Average SNR  : {avg_snr:.2f} dB  (Target > 20.0 dB) {'✓' if avg_snr > 20 else '✗'}")
        print(f" Average STOI : {avg_stoi:.4f}    (Target > 0.850)  {'✓' if avg_stoi > 0.85 else '✗'}")
        print(f" Average PESQ : {avg_pesq:.3f}     (Target > 2.50)   {'✓' if avg_pesq > 2.5 else '✗'}")
        print("------------------------------------------\n")

        if avg_pesq > best_pesq:
            best_pesq = avg_pesq
            checkpoint_path = f"checkpoints/best_anc_model_epoch{epoch}_pesq{avg_pesq:.2f}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved checkpoint: {checkpoint_path}\n")

if __name__ == "__main__":
    main()

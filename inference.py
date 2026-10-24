import os
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = "./checkpoints/best_anc_model_epoch35_pesq2.98.pt"
DATASET_DIR = os.path.abspath("./military_anc_dataset")
MANIFEST_PATH = os.path.join(DATASET_DIR, "dataset_manifest.csv")

OUTPUT_NOISY = "./original_noisy_sample.wav"
OUTPUT_CLEANED = "./cleaned_output.wav"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
SAMPLE_RATE = 16000

# ---------------------------------------------------------
# 2. MODEL ARCHITECTURE
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
# 3. INFERENCE PIPELINE
# ---------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    # Load dataset manifest to find a sample file
    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    sample_row = val_df.iloc[0]

    input_wav_path = os.path.join(DATASET_DIR, sample_row['mixed_file'])
    print(f"Using sample audio file: {input_wav_path}")

    # Initialize Model
    model = ComplexCRN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Audio
    noisy_audio, sr = sf.read(input_wav_path)
    sf.write(OUTPUT_NOISY, noisy_audio, sr)  # Save original copy for listening comparison

    audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    window = torch.hann_window(WIN_LENGTH).to(DEVICE)

    print("Cleaning audio...")
    with torch.no_grad():
        stft = torch.stft(audio_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
        real, imag = stft.real.unsqueeze(1), stft.imag.unsqueeze(1)

        est_real, est_imag = model(real, imag)

        est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
        est_audio = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, length=audio_tensor.shape[-1])

    cleaned_np = est_audio.squeeze(0).cpu().numpy()
    sf.write(OUTPUT_CLEANED, cleaned_np, SAMPLE_RATE)

    print("\n--- Audio Processing Complete ---")
    print(f"1. Original Noisy Audio saved to : {OUTPUT_NOISY}")
    print(f"2. Cleaned Output Audio saved to : {OUTPUT_CLEANED}\n")

if __name__ == "__main__":
    main()

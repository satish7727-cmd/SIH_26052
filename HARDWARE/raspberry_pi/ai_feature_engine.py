import os
import torch
import torch.nn as nn
import numpy as np
import scipy.signal

# 1. Define Model Architecture (Matching ComplexCRN / ANC Model)
class ANCFeatureClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ANCFeatureClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class AIFeaturePath:
    def __init__(self, checkpoint_path="checkpoints/best_anc_model.pt", sample_rate=16000):
        self.sr = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ANCFeatureClassifier().to(self.device)
        
        # Load pre-trained weights if available
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 100:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[AI ENGINE] Successfully loaded model from {checkpoint_path}")
            except Exception as e:
                print(f"[WARN] Could not parse weights ({e}). Running with initialized architecture.")
        else:
            print("[WARN] Model checkpoint empty or missing. Initialized model architecture.")
        
        self.model.eval()

    def compute_stft_tensor(self, audio_frame):
        # Compute STFT spectrogram
        f, t, Zxx = scipy.signal.stft(audio_frame, fs=self.sr, nperseg=512, noverlap=384)
        spectrogram = np.abs(Zxx)
        # Convert to Tensor shape: [Batch=1, Channel=1, Freq, Time]
        tensor_spec = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor_spec

    def analyze_audio(self, audio_frame):
        spec_tensor = self.compute_stft_tensor(audio_frame)
        
        with torch.no_grad():
            logits = self.model(spec_tensor)
            pred_class = torch.argmax(logits, dim=1).item()
        
        # Calculate Energy and Peak metrics
        energy = np.mean(audio_frame ** 2)
        peak = np.max(np.abs(audio_frame))

        speech_status = 1 if (energy > 0.002 and pred_class != 3) else 0
        impulsive_status = 1 if (peak > 0.7 or pred_class == 2) else 0

        # Class Mapping: 0=Ambient, 1=Speech, 2=Impulsive/Artillery, 3=Heavy Noise
        noise_type_map = {0: 0x01, 1: 0x02, 2: 0x0A, 3: 0x0B}
        noise_type = noise_type_map.get(pred_class, 0x01)

        return {
            "speech_status": speech_status,
            "noise_type": noise_type,
            "impulsive_status": impulsive_status
        }

if __name__ == "__main__":
    engine = AIFeaturePath()
    test_audio = np.random.uniform(-0.1, 0.1, 2048)
    metrics = engine.analyze_audio(test_audio)
    print(f"[AI ENGINE LIVE TEST] Classification: {metrics}")

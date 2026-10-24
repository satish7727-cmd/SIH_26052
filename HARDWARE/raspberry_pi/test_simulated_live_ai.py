import time
import torch
import numpy as np
from ai_feature_engine import AIFeaturePath

def main():
    print("==========================================================")
    print("   LIVE ANC AI ENGINE TEST (SIMULATED STREAM / INFERENCE) ")
    print("==========================================================")
    
    engine = AIFeaturePath(checkpoint_path="checkpoints/best_anc_model.pt")
    
    sample_rate = 16000
    frame_len = 2048  # ~128ms frame

    print("\n[INFO] Simulating continuous audio frames into PyTorch model...")
    print("Press Ctrl+C to stop.\n")

    test_scenarios = [
        ("Quiet Ambient Noise", lambda: np.random.normal(0, 0.001, frame_len)),
        ("Speech Frequency Signal", lambda: 0.05 * np.sin(2 * np.pi * 440 * np.arange(frame_len)/sample_rate) + np.random.normal(0, 0.005, frame_len)),
        ("High-Energy Impulsive Burst", lambda: np.append(np.random.normal(0, 0.01, 1000), np.random.normal(0, 0.9, 1048))),
        ("Heavy Continuous Engine Noise", lambda: np.random.normal(0, 0.2, frame_len))
    ]

    try:
        idx = 0
        while True:
            scenario_name, generator = test_scenarios[idx % len(test_scenarios)]
            audio_frame = generator()
            
            # Perform live PyTorch model inference
            metrics = engine.analyze_audio(audio_frame)
            
            speech = "SPEECH" if metrics['speech_status'] == 1 else "NO SPEECH"
            impulse = "IMPULSE!" if metrics['impulsive_status'] == 1 else "NORMAL"
            
            noise_map = {0x01: "Ambient", 0x02: "Speech", 0x0A: "Artillery/Impulse", 0x0B: "Heavy Noise/Tank"}
            noise_label = noise_map.get(metrics['noise_type'], "Unknown")

            print(f"\r[Simulated Input: {scenario_name:30s}] -> Model Output: {speech:10s} | Class: {noise_label:18s} | State: {impulse}")
            
            time.sleep(1.0)
            idx += 1

    except KeyboardInterrupt:
        print("\n\n[INFO] Live inference test stopped.")

if __name__ == "__main__":
    main()

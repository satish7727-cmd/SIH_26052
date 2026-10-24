import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import onnxruntime as ort

from evaluate_complex_crn import ComplexCRN, CHECKPOINT_PATH, N_FFT, HOP_LENGTH, WIN_LENGTH, SAMPLE_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ONNX_PATH = "checkpoints/best_anc_model.onnx"

# ---------------------------------------------------------
# 1. EXPORT PYTORCH MODEL TO ONNX
# ---------------------------------------------------------
def export_to_onnx():
    print("=" * 70)
    print(" [STEP 1] EXPORTING MODEL TO ONNX FORMAT ")
    print("=" * 70)

    model = ComplexCRN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    dummy_real = torch.randn(1, 1, 257, 5, device=DEVICE)
    dummy_imag = torch.randn(1, 1, 257, 5, device=DEVICE)

    torch.onnx.export(
        model,
        (dummy_real, dummy_imag),
        ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['mix_real', 'mix_imag'],
        output_names=['est_real', 'est_imag'],
        dynamic_axes={
            'mix_real': {3: 'time_steps'},
            'mix_imag': {3: 'time_steps'},
            'est_real': {3: 'time_steps'},
            'est_imag': {3: 'time_steps'}
        }
    )
    print(f"[+] Successfully exported model to: {ONNX_PATH}\n")

# ---------------------------------------------------------
# 2. REAL-TIME FRAME-BY-FRAME STREAMING BENCHMARK
# ---------------------------------------------------------
def benchmark_realtime_streaming():
    print("=" * 70)
    print(" [STEP 2] RUNNING REAL-TIME FRAME STREAMING LATENCY TEST ")
    print("=" * 70)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
    print(f"[+] Active ONNX Execution Provider: {session.get_providers()[0]}")

    window = torch.hann_window(WIN_LENGTH).to(DEVICE)
    dummy_chunk = torch.randn(1, WIN_LENGTH, device=DEVICE)

    # Warmup GPU execution engine
    for _ in range(10):
        stft_res = torch.stft(dummy_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
        r = stft_res.real.unsqueeze(1).cpu().numpy()
        i = stft_res.imag.unsqueeze(1).cpu().numpy()
        _ = session.run(None, {'mix_real': r, 'mix_imag': i})

    # Benchmark 100 frame iterations
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()

        stft_res = torch.stft(dummy_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)
        r = stft_res.real.unsqueeze(1).cpu().numpy()
        i = stft_res.imag.unsqueeze(1).cpu().numpy()

        outputs = session.run(None, {'mix_real': r, 'mix_imag': i})

        est_real = torch.tensor(outputs[0]).to(DEVICE)
        est_imag = torch.tensor(outputs[1]).to(DEVICE)
        est_stft = torch.complex(est_real.squeeze(1), est_imag.squeeze(1))
        _ = torch.istft(est_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    audio_duration_per_chunk = (HOP_LENGTH / SAMPLE_RATE) * 1000

    print("=" * 70)
    print(f" Audio Frame Duration          : {audio_duration_per_chunk:.2f} ms")
    print(f" Average Processing Latency    : {avg_latency:.2f} ms per frame")
    print(f" 99th Percentile Max Latency   : {p99_latency:.2f} ms per frame")
    print("=" * 70)

    if avg_latency < 20.0:
        print(" VERDICT: PASS! System is fully qualified for Real-Time Execution! ")
    else:
        print(" VERDICT: Latency needs further optimization (Target < 20ms).")
    print("=" * 70)

if __name__ == "__main__":
    export_to_onnx()
    benchmark_realtime_streaming()

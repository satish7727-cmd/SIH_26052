# SIH_26052 — Adaptive Noise Cancellation for Defence Environments

**AI-Assisted Real-Time Acoustic Noise Cancellation System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FPGA](https://img.shields.io/badge/FPGA-Verilog-orange)](https://www.verilog.com/)
[![Platform](https://img.shields.io/badge/Platform-Zynq%20%7C%20Raspberry%20Pi-green)](https://www.raspberrypi.com/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)]()

## 1. Project Overview

SIH_26052 is an AI-assisted adaptive noise cancellation system designed for challenging acoustic environments, including defence and operational scenarios.

The system combines adaptive digital signal processing, deep learning-based acoustic scene analysis, and hardware-oriented real-time processing to reduce unwanted noise while preserving important speech information.

The proposed architecture integrates an FPGA-based adaptive noise cancellation pipeline with a Raspberry Pi-based AI processing engine.

The system aims to identify changing acoustic conditions, detect impulsive noise, and adapt cancellation parameters according to the surrounding environment.

## 2. Key Objectives

- Develop an adaptive noise cancellation system for complex acoustic environments.
- Implement an adaptive filtering algorithm based on Filtered-x Least Mean Squares (FxLMS).
- Classify acoustic scenes using a deep learning model.
- Detect impulsive noise and distinguish it from stationary and non-stationary noise.
- Integrate AI inference with an FPGA-oriented signal processing architecture.
- Evaluate noise reduction, speech intelligibility, and computational performance.
- Explore deployment on resource-constrained embedded platforms.

## 3. System Architecture

```text
        Acoustic Environment
                 |
          Microphone Inputs
                 |
                 v
      Audio Acquisition Pipeline
                 |
          +------+------+
          |             |
          v             v
    Reference Audio   Primary/Error
          |             Audio
          |             |
          v             v
     FPGA Adaptive Noise Cancellation
          |
          +----------------------+
          |                      |
          v                      v
      FxLMS Filter         Audio Output
          ^
          |
    Adaptive Control
          ^
          |
          +---------------------------+
                                      |
                                      v
                           Raspberry Pi AI Engine
                                      |
                           Feature Extraction
                                      |
                             Neural Network
                                      |
                   +------------------+------------------+
                   |                  |                  |
                   v                  v                  v
              Speech Status       Noise Type       Impulse Status
                   |                  |                  |
                   +------------------+------------------+
                                      |
                                      v
                            ANC Control Parameters
```

*Conceptual architecture. Final interfaces and real-time integration depend on the implemented hardware and software configuration.*

## 4. Hardware and Software

### Hardware targets

- **FPGA:** Zybo Z7-10 development board with Zynq-7000 SoC
- **Embedded processor:** Raspberry Pi 4 Model B
- **Audio interface:** Microphone inputs and an appropriate audio codec/interface
- **Communication:** FPGA-to-Raspberry Pi bridge, with the final transport determined by the implemented interface

### Software and development tools

- Python for AI development, training, and evaluation
- PyTorch-compatible training workflows, depending on the existing environment
- Verilog HDL for FPGA-oriented signal processing
- C for embedded firmware
- Linux for development and deployment
- Git and GitHub for version control

## 5. Core Technologies

### FxLMS adaptive noise cancellation

The Filtered-x Least Mean Squares algorithm is used as the basis for adaptive control in systems with a secondary acoustic path.

The algorithm uses a reference noise signal and an estimate of the secondary path to update filter coefficients and minimize the residual error.

Its effectiveness depends on the quality of the reference signal, secondary-path modelling, signal timing, and adaptation parameters.

### AI-based acoustic scene analysis

The AI processing pipeline is designed to extract useful acoustic features and classify the current sound environment.

The intended output categories are:

| Task | Output classes |
|---|---|
| Speech detection | No speech, speech |
| Noise classification | Stationary, non-stationary, impulsive, operational |
| Impulse detection | No impulse, impulse |

These labels describe the intended classification tasks. Actual model accuracy must be established through validation and testing.

### Hardware-oriented processing

The FPGA source tree contains Verilog modules for adaptive filtering, feature extraction, audio acquisition, and top-level integration.

The Raspberry Pi source tree contains Python modules for AI feature processing, communication, and simulated live-pipeline testing.

## 6. Repository Structure

```text
SIH_26052/
├── HARDWARE/
│   ├── config.ini
│   ├── fpga_zybo/
│   │   ├── constraints/
│   │   │   └── zybo_z710.xdc
│   │   ├── firmware/
│   │   │   └── anc_controller.c
│   │   └── src/
│   │       ├── anc_top.v
│   │       ├── fast_feature_extractor.v
│   │       ├── fxlms_filter.v
│   │       └── i2s_tdm_receiver.v
│   └── raspberry_pi/
│       ├── ai_feature_engine.py
│       ├── fpga_bridge.py
│       ├── requirements.txt
│       ├── start_pipeline.sh
│       └── test_simulated_live_ai.py
├── checkpoints/                 # Local model checkpoints
├── LibriSpeechData/             # Local speech dataset
├── military_anc_dataset/        # Local prepared dataset
├── evaluate_complex_crn.py
├── evaluate_full_val.py
├── export_and_benchmark.py
├── generate_organized_dataset.py
├── inference.py
├── live_mic_anc_cli.py
├── train_anc_model.py
├── .gitignore
└── README.md
```

Large datasets, audio samples, and model checkpoints are excluded from Git by the current `.gitignore` configuration.

## 7. Getting Started

### Prerequisites

- Linux or a compatible development environment
- Python 3.10 or a compatible version supported by the dependencies
- Git
- Sufficient storage for datasets and model training
- Optional GPU acceleration for supported training configurations

### Clone the repository

```bash
git clone https://github.com/satish7727-cmd/SIH_26052.git
cd SIH_26052
```

### Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Install the dependencies required by the root-level Python scripts using the appropriate requirements file or the dependencies documented in each script.

For the Raspberry Pi software:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r HARDWARE/raspberry_pi/requirements.txt
```

*Note: The Raspberry Pi environment should be configured on the target device. Review the dependency versions and hardware-specific requirements before deployment.*

## 8. Dataset Preparation

The project includes scripts for organizing audio data and preparing datasets for model training.

Relevant components:

- `generate_organized_dataset.py` — dataset organization
- `LibriSpeechData/` — local speech dataset directory
- `military_anc_dataset/` — local prepared dataset directory

Before training:

1. Obtain the required datasets from their legitimate sources.
2. Review their licences and permitted uses.
3. Check audio sample rates, channel counts, and file formats.
4. Verify labels and class distributions.
5. Separate training, validation, and test sets to avoid data leakage.
6. Document preprocessing parameters and dataset versions.

The dataset directories are local resources and are not automatically included when cloning this repository.

## 9. Model Training

The main training entry point is:

```bash
python train_anc_model.py --help
```

If the script does not implement a `--help` option, inspect its argument parser and configuration before running training.

A typical training workflow consists of:

1. Prepare and validate the audio dataset.
2. Configure model parameters and training settings.
3. Train the model using the available training script.
4. Monitor training and validation metrics.
5. Save checkpoints locally.
6. Evaluate the selected model on held-out data.

The actual command-line arguments, model architecture, and required dataset format should be confirmed from the current implementation.

## 10. Evaluation and Benchmarking

The repository contains scripts for evaluation and model export:

```bash
python evaluate_complex_crn.py --help
python evaluate_full_val.py --help
python export_and_benchmark.py --help
```

Check each script's supported arguments before execution.

### Evaluation metrics

The following metrics are relevant to assessing an ANC system:

| Metric | Purpose |
|---|---|
| Signal-to-Noise Ratio (SNR) | Measures signal quality relative to noise |
| STOI | Estimates speech intelligibility |
| PESQ | Estimates perceived speech quality under supported evaluation conditions |
| Classification accuracy | Measures acoustic classification performance |
| Precision, recall and F1-score | Evaluate classification and impulse detection |
| Model size and inference latency | Assess embedded deployment feasibility |
| CPU, memory and throughput | Assess resource consumption |

Results should be reported using measured values from reproducible experiments. No benchmark results are claimed in this README.

## 11. Live Audio and Embedded Pipeline

The project includes a live microphone CLI and Raspberry Pi pipeline components.

Relevant files:

- `live_mic_anc_cli.py`
- `HARDWARE/raspberry_pi/ai_feature_engine.py`
- `HARDWARE/raspberry_pi/fpga_bridge.py`
- `HARDWARE/raspberry_pi/test_simulated_live_ai.py`
- `HARDWARE/raspberry_pi/start_pipeline.sh`

Before running live audio:

1. Configure the correct microphone and audio device.
2. Verify the expected sample rate, channel count, and sample format.
3. Confirm that the trained model and required configuration files are available.
4. Test the AI pipeline using simulated or prerecorded audio.
5. Verify the communication protocol and timing before connecting physical hardware.

The presence of these source files does not by itself establish successful end-to-end real-time hardware operation.

## 12. FPGA Development

The FPGA source code is located in:

`HARDWARE/fpga_zybo/src/`

Key modules include:

- `anc_top.v` — top-level integration
- `fxlms_filter.v` — adaptive filtering
- `fast_feature_extractor.v` — feature extraction
- `i2s_tdm_receiver.v` — audio reception

The project also contains a board constraints file and embedded C firmware.

FPGA implementation requires appropriate synthesis and implementation tools, correct pin assignments, clock configuration, interface verification, and hardware testing. Confirm module interfaces and target-board compatibility before building a bitstream.

## 13. Project Status

**Development stage:** Software and hardware-oriented development.

| Component | Current repository contents |
|---|---|
| AI training | Training script available |
| Evaluation | Evaluation and benchmarking scripts available |
| Live audio | CLI implementation available |
| FPGA design | Verilog source modules available |
| Embedded integration | Raspberry Pi bridge and pipeline scripts available |
| End-to-end hardware validation | Requires confirmation from actual board testing |
| Performance benchmarks | To be documented from measured experiments |

## 14. Future Improvements

- Complete and validate FPGA–Raspberry Pi integration.
- Improve robustness under changing noise conditions.
- Evaluate speech preservation alongside noise attenuation.
- Optimize inference latency and memory usage for embedded deployment.
- Add reproducible test configurations and automated regression tests.
- Publish measured benchmark tables and experiment configurations.
- Document hardware setup, audio routing, and deployment procedures.

## 15. Safety and Limitations

This project is a research and engineering prototype. Its performance and reliability must be validated under representative conditions before any operational use.

Noise cancellation does not guarantee protection against hazardous sound levels. Appropriate hearing protection and established safety procedures remain necessary.

Do not rely on the system as the sole means of communication, hazard detection, or personnel protection.

## 16. Author and Repository

**Project:** SIH_26052 — Adaptive Noise Cancellation for Defence Environments

**GitHub:** https://github.com/satish7727-cmd/SIH_26052

---

*Developed as an engineering project exploring adaptive signal processing, AI-based acoustic analysis, and embedded hardware implementation.*# SIH_26052

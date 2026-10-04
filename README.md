# NeuroBrain

NeuroBrain is a local AI voice and robotics project running on a Raspberry Pi 4.

## Current Stable Version

NeuroBrain v1 combines three local AI components:

- Whisper.cpp for speech-to-text
- Qwen3 1.7B for language processing
- Piper for text-to-speech

## Architecture

Microphone -> Whisper STT -> Qwen3 LLM -> Piper TTS -> Speaker

All inference runs locally on the Raspberry Pi.

## Hardware

- Raspberry Pi 4 (4 GB RAM)
- Logitech C270 USB webcam and microphone
- Speaker / audio output
- GPIO available for future sensors and motor control

## Current Features

- Automatic voice activity detection
- Local English speech recognition
- Local Qwen3 1.7B language model
- Local Piper speech synthesis
- Resident Whisper and Qwen servers
- Fully local voice conversation

## Main Programs

- neurobrain.py - main voice interaction loop
- start_neurobrain.py - launches NeuroBrain and resident AI servers
- neurobrain_stable_v1.py - stable v1 checkpoint
- start_neurobrain_stable_v1.py - stable v1 launcher

## Start NeuroBrain

    source /home/drpeker/neurobrain/venv/bin/activate
    python /home/drpeker/neurobrain/start_neurobrain.py

## Repository Policy

Large AI models and runtime files are intentionally excluded from GitHub. GGUF models, Whisper models, Piper ONNX voices, llama.cpp, the Python virtual environment, WAV files, and logs must be installed or restored separately.

## Roadmap

Planned development includes conversation memory, hardware-state awareness, wake-word detection, GPIO sensors, motor control, camera vision, obstacle detection, autonomous navigation, and persistent robot identity.

## Goal

The goal of NeuroBrain is to create a self-contained robot intelligence platform combining local language AI with vision, sensors, and physical motor control without requiring cloud AI services.

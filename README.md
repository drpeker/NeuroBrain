# NeuroBrain

NeuroBrain is a fully local AI voice, vision and robotics platform running on a Raspberry Pi 4.

## Current Working Checkpoint
`fe85f97` — working voice + vision + safe GPIO integration.

## Architecture
```text
C270 microphone → Whisper Tiny Q4 ─┐
                                   ├→ NeuroBrain Core → actions/chat/vision → Piper → speaker
C270 camera → OpenCV → YOLOv8n → VisionState → VisionService ───────────────┘
                                   ↑
                            Qwen3 1.7B router/chat
```

## AI Components
- Whisper.cpp — speech-to-text.
- YOLOv8n via NCNN — real-time object detection.
- Qwen3 1.7B Q4 — semantic intent routing and conversation.
- Piper — neural text-to-speech.

## Hardware
- Raspberry Pi 4, 4 GB RAM.
- Logitech C270 camera + microphone.
- Speaker/audio output.
- LED on authorized BCM GPIO17.

## Working Features
- Local English speech recognition and conversation.
- Safe GPIO17 READ / SET / TOGGLE.
- Live C270 multi-object detection.
- Bounding boxes + NMS.
- Temporal VisionState stabilization.
- Thread-safe VisionService.
- Background VisionRuntime.
- Spoken “What do you see?” queries.
- Local Piper speech output.

## Start
```bash
/home/drpeker/neurobrain/venv/bin/python \
  /home/drpeker/neurobrain/start_neurobrain_vision_dev.py
```

## Project Memory / Recovery
If opened with no prior conversation context, read:
1. [Current State](docs/CURRENT_STATE.md)
2. [Complete System Diagrams](docs/SYSTEM_DIAGRAMS.md)
3. [Architecture](docs/ARCHITECTURE.md)
4. [Hardware](docs/HARDWARE.md)
5. [Models](docs/MODELS.md)
6. [Install / Runtime Layout](docs/INSTALL.md)
7. [Recovery](docs/RECOVERY.md)
8. [Design Decisions](docs/DESIGN_DECISIONS.md)

These files are the durable engineering memory of NeuroBrain.

## Safety Principle
The LLM does not directly control GPIO or motors. Physical actions pass through deterministic Core validation and hardware-control code.

## Repository Policy
Large AI models and runtime artifacts are excluded from GitHub and must be restored separately.

## Roadmap
Faster routing; real ToF/ultrasonic range; IMU/encoders; safe motor control; fused world state; persistent identity/memory; autonomous navigation.

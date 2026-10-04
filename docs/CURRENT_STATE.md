# NeuroBrain — Current Project State

Last updated: 2026-10-04

## Last known working checkpoint
- `fe85f97` — working integrated voice + vision + GPIO.
- `a2f80fe` — working independent live vision service.
- `49ffe01` — working Qwen semantic action router.
- `32bd600` — working GPIO17 READ/SET/TOGGLE.
- `8f87aa7` — stable local voice pipeline.

## Working system
Platform: Raspberry Pi 4, 4 GB RAM.

Voice: C270 microphone → Whisper Tiny Q4 → Qwen3 1.7B semantic router → NeuroBrain Core.
Actions: `GPIO_SET`, `GPIO_TOGGLE`, `GPIO_READ`, `VISION_QUERY`, `CHAT`.
GPIO actions pass through `actions.py` and `hardware.py`; only GPIO17 is authorized.
CHAT uses a second Qwen call.

Vision: C270 camera → V4L2/OpenCV → YOLOv8n NCNN → bbox decode/NMS → VisionState → VisionService → Core.
`VisionRuntime` runs continuously in a background thread.

Typical YOLO inference is about 85–90 ms (~11–12 inference/s).
`VisionState(history_size=5, min_hits=3)` provides temporal stabilization.
`near/medium/far` is based only on bounding-box area. It is NOT physical range and MUST NOT be used for collision safety.

Output: Core/Qwen response → resident Piper `en_US-lessac-medium` → WAV → `aplay`.

## Confirmed end-to-end tests
- Ordinary spoken conversation works.
- “What do you see?” routes to live VisionService state.
- Live objects are spoken correctly.
- “Turn on the LED” safely controls GPIO17.
- Vision remains active while voice interaction continues.

## Current bottleneck
Vision is fast enough. Qwen routing/chat is the main latency bottleneck: commonly ~7–14 s, with one cold/slow router request around 48 s.

## Protect these files
- `neurobrain_router_dev.py`
- `neurobrain_vision_dev.py`
- `vision_detect_working.py`
- `vision_live_working.py`
- `vision_state_v1_working.py`
- `neurobrain_clean_baseline.py` when present.

Develop risky features in new copies and checkpoint before experiments.

## Next priorities
1. Preserve this checkpoint.
2. Reduce routing latency without weakening the Core/action safety boundary.
3. Add ToF/ultrasonic before autonomous collision decisions.
4. Add IMU/encoders and richer world state.
5. Add motor actions behind deterministic safety control.
6. Add persistent identity/memory independently of the replaceable LLM.

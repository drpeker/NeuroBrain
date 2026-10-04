# NeuroBrain Architecture

## Principle
**NeuroBrain is not the LLM.**

NeuroBrain is identity + goals + memory + world/robot state + perception + hardware capabilities + safety/action control + a replaceable LLM backend.

No LLM should directly manipulate GPIO or motors.

## Current architecture
```text
C270 microphone → Whisper Tiny Q4 ──→ Qwen3 1.7B router ─┐
                                                        ├→ NeuroBrain Core
C270 camera → OpenCV/V4L2 → YOLOv8n/NCNN → VisionState  │
                                      → VisionService ───┘
                                                        │
                           ┌────────────────────────────┼─────────────────┐
                           ↓                            ↓                 ↓
                      VISION_QUERY                 GPIO ACTION          CHAT
                           ↓                            ↓                 ↓
                     VisionService                  actions.py       Qwen3 1.7B
                                                    hardware.py      second call
                                                        ↓                 │
                                                     GPIO17               │
                                                        └────────┬────────┘
                                                                 ↓
                                                           response text
                                                                 ↓
                                                              Piper
                                                                 ↓
                                                              speaker
```

YOLO is the visual AI. Qwen does not see raw camera pixels in the current design. YOLO creates structured visual state; the Core reads it.

## Safety boundary
```text
language → intent → Core validation → actions.py → hardware.py → physical device
```
Never:
```text
LLM → raw GPIO/motor pins
```

Future motor control must retain this boundary and add local reflex/safety rules.

## Future world state
Fuse YOLO object class/position with physical range, IMU orientation and encoder odometry. The Core owns the world state; the LLM may reason over it but is not its source of truth.

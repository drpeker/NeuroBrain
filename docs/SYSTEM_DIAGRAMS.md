# NeuroBrain — Complete System Diagrams and Engineering Map

**Purpose:** This is the high-level reconstruction map of NeuroBrain. If all chat history is lost, this document plus the linked recovery documents should explain what the working system is, why it is designed this way, how data moves through it, and which experiments must not be mistaken for the working architecture.

**Repository documentation checkpoint before this file:** `a91e524`  
**Last known working integrated code checkpoint:** `fe85f97` — voice + vision + safe GPIO.

---

## 1. The central idea

**NeuroBrain is NOT Qwen.**

NeuroBrain is the complete robot intelligence architecture:

```text
                         ┌──────────────────────────────┐
                         │          NEUROBRAIN          │
                         │                              │
                         │ identity / goals / memory    │
                         │ world state / perception     │
                         │ safety / actions / hardware  │
                         │ replaceable AI backends      │
                         └──────────────────────────────┘
```

The AI models are specialist components inside NeuroBrain:

```text
Whisper = hearing / speech recognition
YOLO    = visual perception
Qwen    = language understanding + intent + conversation
Piper   = speech generation

Core    = deterministic coordinator, state owner and safety boundary
```

Critical distinction:

```text
Qwen DOES NOT see camera pixels.
YOLO sees/detects objects.

Qwen DOES NOT directly drive GPIO.
Core/actions/hardware drive GPIO.

LLM output is NOT physical truth.
Verified sensor/hardware state is physical truth.
```

---

## 2. Complete current working system

```text
┌──────────────────────────────── Raspberry Pi 4 / 4 GB ────────────────────────────────┐
│                                                                                       │
│                                  NEUROBRAIN                                            │
│                                                                                       │
│   ┌──────────────────────────── VOICE / LANGUAGE ──────────────────────────────┐       │
│   │                                                                           │       │
│   │  Human speech                                                             │       │
│   │      │                                                                    │       │
│   │      ▼                                                                    │       │
│   │  Logitech C270 microphone                                                 │       │
│   │      │                                                                    │       │
│   │      ▼                                                                    │       │
│   │  sounddevice / ALSA                                                       │       │
│   │  16 kHz mono                                                              │       │
│   │      │                                                                    │       │
│   │      ▼                                                                    │       │
│   │  ┌──────────────────────────────┐                                         │       │
│   │  │ Whisper Tiny Q4              │                                         │       │
│   │  │ AI MODEL #1                  │                                         │       │
│   │  │ Speech → text                │                                         │       │
│   │  │ server: 127.0.0.1:8081       │                                         │       │
│   │  └──────────────┬───────────────┘                                         │       │
│   │                 │ user text                                                │       │
│   │                 ▼                                                          │       │
│   │  ┌──────────────────────────────┐                                         │       │
│   │  │ Qwen3 1.7B Q4               │                                         │       │
│   │  │ AI MODEL #3                  │                                         │       │
│   │  │ semantic intent router       │                                         │       │
│   │  │ server: 127.0.0.1:8080       │                                         │       │
│   │  └──────────────┬───────────────┘                                         │       │
│   └─────────────────┼─────────────────────────────────────────────────────────┘       │
│                     │                                                                  │
│                     ▼                                                                  │
│             ┌─────────────────────┐                                                    │
│             │   NEUROBRAIN CORE   │◄──────────────────────────────┐                    │
│             │ deterministic logic │                               │                    │
│             │ validation / routing│                               │                    │
│             └──────────┬──────────┘                               │                    │
│                        │                                          │                    │
│        ┌───────────────┼──────────────────┐                       │                    │
│        │               │                  │                       │                    │
│        ▼               ▼                  ▼                       │                    │
│   VISION_QUERY      GPIO ACTION          CHAT                     │                    │
│        │               │                  │                       │                    │
│        │               ▼                  ▼                       │                    │
│        │          ┌──────────┐      Qwen3 1.7B                    │                    │
│        │          │actions.py│      second call                   │                    │
│        │          └────┬─────┘           │                       │                    │
│        │               ▼                 │                       │                    │
│        │         ┌───────────┐            │                       │                    │
│        │         │hardware.py│            │                       │                    │
│        │         └────┬──────┘            │                       │                    │
│        │              ▼                   │                       │                    │
│        │           GPIO17 → LED           │                       │                    │
│        │                                  │                       │                    │
│        │                                  │                       │                    │
│        │       ┌──────────── VISION / PERCEPTION ──────────────┐  │                    │
│        │       │                                               │  │                    │
│        │       │ Logitech C270 camera                          │  │                    │
│        │       │       │                                       │  │                    │
│        │       │       ▼                                       │  │                    │
│        │       │ OpenCV / V4L2                                 │  │                    │
│        │       │ 640×480 MJPG                                  │  │                    │
│        │       │       │                                       │  │                    │
│        │       │       ▼                                       │  │                    │
│        │       │ ┌──────────────────────────────┐              │  │                    │
│        │       │ │ YOLOv8n / NCNN              │              │  │                    │
│        │       │ │ AI MODEL #2                  │              │  │                    │
│        │       │ │ object detection             │              │  │                    │
│        │       │ └─────────────┬────────────────┘              │  │                    │
│        │       │               ▼                               │  │                    │
│        │       │ bbox decode + class-aware NMS                 │  │                    │
│        │       │               │                               │  │                    │
│        │       │               ▼                               │  │                    │
│        │       │ VisionState                                   │  │                    │
│        │       │ history=5 / min_hits=3                        │  │                    │
│        │       │ left-center-right                             │  │                    │
│        │       │ upper-middle-lower                            │  │                    │
│        │       │ near-medium-far*                              │  │                    │
│        │       │               │                               │  │                    │
│        │       │               ▼                               │  │                    │
│        └──────►│ VisionService ─────────────────────────────────┼──┘                    │
│                │ thread-safe current visual state              │                       │
│                └───────────────────────────────────────────────┘                       │
│                                                                                       │
│                 Core/Qwen/vision result                                                │
│                         │                                                             │
│                         ▼                                                             │
│              ┌──────────────────────────┐                                             │
│              │ Piper Lessac Medium      │                                             │
│              │ AI MODEL #4 / neural TTS │                                             │
│              │ text → speech            │                                             │
│              └────────────┬─────────────┘                                             │
│                           ▼                                                           │
│                        speaker                                                        │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘

* near/medium/far is only a bounding-box-area heuristic, NOT measured distance.
```

---

## 3. The four AI models and their exact jobs

| Component | Type | Job | Current mode |
|---|---|---|---|
| Whisper Tiny Q4 | neural STT | speech → text | resident server |
| YOLOv8n NCNN | neural vision detector | frame → objects + boxes | continuous background |
| Qwen3 1.7B Q4 | LLM | semantic routing + conversation | resident server |
| Piper Lessac Medium | neural TTS | text → speech | model loaded resident |

The Core is deliberately **not** an AI model. It is deterministic Python logic.

---

## 4. “What do you see?” exact data flow

```text
Human:
"What do you see?"
       │
       ▼
C270 microphone
       │
       ▼
Whisper Tiny Q4
       │
       ▼
"What do you see?"
       │
       ▼
Qwen semantic router
       │
       ▼
{"action":"VISION_QUERY"}
       │
       ▼
NeuroBrain Core
       │
       │ reads current state
       ▼
VisionService ◄──── VisionState ◄──── YOLO ◄──── C270 camera
       │
       ▼
"I currently see:
 laptop at center-middle (near)."
       │
       ▼
Piper
       │
       ▼
speaker
```

Important: Qwen did not visually recognize the laptop. YOLO did.

---

## 5. GPIO17 action flow and safety boundary

Example spoken command:

```text
"Turn on the LED."
       │
       ▼
Whisper
       │
       ▼
Qwen router
       │
       ▼
{"action":"GPIO_SET","pin":17,"value":1}
       │
       ▼
NeuroBrain Core
       │
       ▼
actions.py
       │
       ▼
hardware.py
       │
       ├── Is pin authorized?  YES: GPIO17
       ├── Is operation valid? YES
       ▼
gpiozero / physical GPIO
       │
       ▼
LED ON
       │
       ▼
verified returned state
       │
       ▼
"GPIO 17 is now high."
       │
       ▼
Piper → speaker
```

The hard safety boundary is:

```text
LLM intent
   │
   ▼
CORE VALIDATION
   │
   ▼
actions.py
   │
   ▼
hardware.py
   │
   ▼
AUTHORIZED PHYSICAL HARDWARE
```

Forbidden design:

```text
Qwen ──────────────────────────────► raw GPIO / motors
```

Only GPIO17 is currently authorized. An LLM request for another GPIO must be rejected by deterministic code.

---

## 6. Normal CHAT flow

```text
speech
  │
  ▼
Whisper
  │
  ▼
text
  │
  ▼
Qwen router
  │
  ▼
{"action":"CHAT"}
  │
  ▼
Qwen3 1.7B second call
  │
  ▼
natural-language answer
  │
  ▼
Piper
  │
  ▼
speaker
```

The current architecture therefore may use Qwen twice for a normal chat request: first routing, then answering. This is one reason routing/chat latency is currently the major performance bottleneck.

---

## 7. Vision pipeline in detail

```text
/dev/video0
   │
   ▼
C270 640×480 MJPG @ 30 FPS
   │
   ▼
OpenCV CAP_V4L2
   │
   ▼
BGR → RGB
   │
   ▼
aspect resize toward target 320
   │
   ▼
padding to multiple of 32
   │
   ▼
normalize 1/255
   │
   ▼
YOLOv8n NCNN
input:  in0
output: out0
   │
   ▼
YOLOv8 output
64 DFL bbox logits + 80 COCO class logits
   │
   ├── sigmoid class scores
   ├── DFL softmax expectation
   ├── strides 8 / 16 / 32
   ▼
decoded bounding boxes
   │
   ▼
class-aware NMS
   │
   ▼
VisionState
   │
   ├── horizontal: left / center / right
   ├── vertical: upper / middle / lower
   ├── temporal history: 5 frames
   ├── stable threshold: 3 hits
   └── bbox-area hint: near / medium / far
   │
   ▼
VisionService
   │
   ├── publish(state)
   ├── get_state()
   ├── get_objects()
   ├── summary()
   ├── age()
   └── is_fresh()
   │
   ▼
NeuroBrain Core
```

Known working YOLO settings:
- target: 320
- confidence threshold: 0.35
- NMS IoU: 0.45
- NCNN threads: 4
- Vulkan: off
- COCO classes: 80
- typical inference: ~85–90 ms

Known successful detections included person, TV, cup, laptop, mouse, cell phone, umbrella, keyboard, remote, bowl, book and chair.

---

## 8. VisionRuntime ownership

```text
neurobrain_vision_dev.py
        │
        ▼
VisionRuntime()
        │
        ├── VisionService
        ├── VisionState(history_size=5,min_hits=3)
        ├── NCNN YOLO model
        ├── C270 camera
        ├── daemon thread: NeuroBrainVision
        └── stop event
```

Vision runs independently in the background. Voice requests do not trigger camera inference from scratch; they query the current published visual state.

This separation is intentional.

---

## 9. Active model paths and servers

### Qwen

```text
Model:
/home/drpeker/neurobrain/models/qwen3-1.7b-q4km.gguf

Runtime:
/home/drpeker/neurobrain/llama.cpp/build/bin/llama-server

Server:
127.0.0.1:8080

Context:
2048

Reasoning:
off
```

Known llama.cpp build during development:
`b11370-bed0a8566`

Observed Qwen performance:
- prompt: ~7.8–8.6 tokens/s
- generation: ~3.0–3.2 tokens/s
- semantic router usually ~7–14 s
- one observed first/cold integrated router call: ~48 s

### Whisper

```text
whisper.cpp:
/home/drpeker/whisper.cpp

Model:
/home/drpeker/whisper.cpp/models/ggml-tiny-q4_0.bin

VAD:
/home/drpeker/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin

Server:
127.0.0.1:8081

Threads:
4

Language:
English

Beam size:
3

Best-of:
3
```

The Q4 model with beam/best-of 3 was selected after testing as the speed/accuracy sweet spot.

### Piper

```text
Executable:
/home/drpeker/neurobrain/venv/bin/piper

Voice:
/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx
```

PiperVoice is loaded once and retained in memory.

### YOLO

```text
/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.param
/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.bin
```

NCNN used in working environment:
`1.0.20260526`

OpenCV observed:
`5.0.0`

---

## 10. Runtime/process map

```text
start_neurobrain_vision_dev.py
          │
          ├──────────────► llama-server / Qwen / :8080
          │
          ├──────────────► whisper-server / :8081
          │
          └──────────────► neurobrain_vision_dev.py
                                   │
                                   ├── loads PiperVoice
                                   ├── starts VisionRuntime thread
                                   ├── records C270 microphone
                                   ├── calls Whisper HTTP API
                                   ├── calls Qwen HTTP API
                                   ├── validates Core actions
                                   └── speaks result with Piper
```

Main integrated launcher:

```bash
/home/drpeker/neurobrain/venv/bin/python \
  /home/drpeker/neurobrain/start_neurobrain_vision_dev.py
```

---

## 11. Important source files

```text
neurobrain_vision_dev.py
    current integrated voice + vision + GPIO system

start_neurobrain_vision_dev.py
    launches current integrated system

neurobrain_router_dev.py
    known-good pre-vision semantic-router baseline

vision_runtime.py
    background camera + YOLO + VisionState + VisionService runtime

vision_state.py
    temporal/spatial visual-state logic

vision_service.py
    thread-safe interface to current visual state

vision_detect_working.py
    preserved working detector checkpoint

vision_live_working.py
    preserved working live YOLO checkpoint

vision_state_v1_working.py
    preserved working VisionState checkpoint

actions.py
    authorized semantic hardware operations

hardware.py
    physical GPIO implementation and authorization boundary
```

Known local immutable baseline when present:

```text
neurobrain_clean_baseline.py
SHA-256:
d0078f04e8102c57d0d3d2954f4c42bf9f2c9321d6dea36d56b7277831a23724
```

Do not overwrite known-good files for experiments.

---

## 12. Known working physical tests

### Vision

The complete chain has been verified:

```text
C270
→ OpenCV
→ YOLO
→ bbox/NMS
→ VisionState
→ VisionService
→ VisionRuntime
→ NeuroBrain
→ spoken answer
```

Examples observed in the integrated system included:

```text
I currently see: laptop at center-middle (near).
I currently see: laptop at center-upper (near), keyboard at center-upper (near).
I currently see: person at center-middle (near).
I currently see: cell phone at center-middle (near).
```

### GPIO

Spoken:

```text
Turn on the LED.
```

produced semantic action:

```text
{"action":"GPIO_SET","pin":17,"value":1}
```

and the physical LED on GPIO17 turned on.

### Chat

Normal speech routed to CHAT and Qwen generated a spoken response while the background vision runtime continued operating.

---

## 13. Known limitations — do not forget these

### Vision distance is not real distance

```text
YOLO bbox area
      │
      ▼
near / medium / far
```

This is only a visual heuristic. It must never be the sole collision-avoidance input.

Future:

```text
YOLO object identity/position
             +
ToF / ultrasonic measured range
             +
IMU orientation
             +
encoder odometry
             ▼
        fused world state
```

### Temporal VisionState behavior

The current filter requires an object to occur in at least 3 of the last 5 frames, but publication still depends on the current frame. A one-frame miss can therefore remove an object temporarily. This was observed during fast camera movement and intentionally left alone because the working system was good enough.

### Router latency

The visual detector is not the main delay. Qwen semantic routing is.

---

## 14. The abandoned unified-Qwen experiment

This is NOT the current architecture.

Goal was to replace router + chat with one Qwen call producing response plus operations:

```text
user request
    │
    ▼
single Qwen call
    │
    ├── response
    └── operations
```

It failed as a reliable Pi architecture.

Observed problems:

1. JSON could contain an extra trailing quote, causing `json.loads` “Extra data”.
2. A first large prompt took about 51.75 s.
3. A shorter ~105-token prompt still took about 17 s total.
4. Server-level `--json-schema` and even minimal object schema produced HTTP 400:
   `Failed to initialize samplers: std::exception`
5. Compact command protocol experiment for:
   `What is the capital city of France? Turn off the LED.`
   produced:
   ```text
   @SET 17 0
   @TOGGLE 17
   ```
   It omitted the conversational answer and generated contradictory actions.

Decision:

```text
ABANDON unified single-call architecture.
RETURN to known-good semantic router architecture.
```

Do not revive this experiment unless it is isolated from the working system and there is a specific new technical reason.

Experimental files may exist locally:

```text
neurobrain_unified_dev.py
start_neurobrain_unified_dev.py
neurobrain_unified_before_single_call.py
```

They are NOT the active system.

---

## 15. Failed Ultralytics/PyTorch path

Do not reinstall Ultralytics/PyTorch on this Pi simply to run YOLO.

The attempted installation pulled very large ARM64 CUDA/NVIDIA dependencies and failed for disk-space reasons. The project deliberately switched to native NCNN.

Working vision backend:

```text
YOLOv8n
   │
   ▼
NCNN ARM64
   │
   ▼
~85–90 ms inference on Pi 4
```

This is the selected path.

---

## 16. Recovery/checkpoint map

```text
8f87aa7
Stable local voice pipeline
    │
    ▼
32bd600
Working GPIO17 READ/SET/TOGGLE
    │
    ▼
49ffe01
Working Qwen semantic action router
    │
    ▼
a2f80fe
Working independent live vision service
    │
    ▼
fe85f97
WORKING INTEGRATED
voice + vision + GPIO
    │
    ▼
a91e524
Complete architecture/recovery documentation
    │
    ▼
[this SYSTEM_DIAGRAMS documentation commit]
```

Other important checkpoint:

```text
4d3adbb
NeuroBrain v1 physical recovery PDF checkpoint
```

If recovery is required, do not blindly reset the working tree. Read `docs/RECOVERY.md`, inspect `git status`, preserve current work, then use Git history deliberately.

---

## 17. Recovery decision tree

```text
SYSTEM DOES NOT WORK
        │
        ├── No microphone input?
        │      └─► ALSA/C270 → sounddevice → Whisper
        │
        ├── No speech recognition?
        │      └─► Whisper :8081 → model/VAD paths
        │
        ├── No Qwen response?
        │      └─► llama-server :8080 → health/model path
        │
        ├── No vision?
        │      └─► /dev/video0
        │           → OpenCV
        │           → vision_detect_working.py
        │           → VisionState
        │           → VisionService
        │           → VisionRuntime
        │           → integrated NeuroBrain
        │
        ├── LED/action fails?
        │      └─► GPIO17 physical test
        │           → hardware.py
        │           → actions.py
        │           → Core/router
        │
        └── Everything unclear?
               └─► return to known checkpoint fe85f97
                   only after preserving current work
```

---

## 18. Future full robot architecture

```text
                           ┌────────────────────┐
                           │    NeuroBrain      │
                           │       Core         │
                           └─────────┬──────────┘
                                     │
              ┌──────────────────────┼────────────────────────┐
              │                      │                        │
              ▼                      ▼                        ▼
        PERCEPTION                 WORLD                  LANGUAGE
              │                    STATE                      │
     ┌────────┼────────┐             │                      Qwen
     │        │        │             │                        │
     ▼        ▼        ▼             │                        │
   YOLO      ToF      IMU            │                        │
 camera     range  orientation       │                        │
     │        │        │             │                        │
     └────────┼────────┘             │                        │
              │                      │                        │
           encoders ─────────────────┘                        │
              │                                               │
              └──────────────────┬────────────────────────────┘
                                 ▼
                         SAFETY / ACTION CORE
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
                motors         servos        GPIO
                   │
                   ▼
             physical robot
```

Example future fused state:

```text
OBJECT person_1
class: person
bearing: +4 degrees
distance: 0.73 m
confidence: 0.94
visible: true
```

YOLO should provide identity/image location; ToF/ultrasonic should provide real range; IMU/encoders should provide robot motion/orientation.

---

## 19. Pi Zero / distributed future

A smaller board can act as the robot body/reflex controller while Pi 4 remains the cognitive node:

```text
Pi 4 NeuroBrain
Whisper / Qwen / Piper / YOLO
Memory / world model / planning
          │
          │ Wi-Fi / structured protocol
          ▼
Pi Zero / Zero 2 W body controller
GPIO / motors / servos
ToF / ultrasonic / IMU / encoders
local emergency stop / reflexes
```

Safety-critical reflexes should not wait for an LLM.

---

## 20. Golden rules for future development

```text
1. A working physical state is precious.
2. Never overwrite a known-good baseline for an experiment.
3. Copy → experiment → test physically → checkpoint → commit → push.
4. LLMs interpret; Core validates.
5. Sensors/hardware are the source of physical truth.
6. YOLO perception is not measured distance.
7. No direct LLM-to-motor/GPIO path.
8. Keep AI backends replaceable.
9. Keep dynamic robot/world state outside model weights.
10. Update CURRENT_STATE and this document after every major working milestone.
```

---

## 21. If an AI assistant opens this repository with zero prior context

Read, in order:

```text
README.md
docs/CURRENT_STATE.md
docs/SYSTEM_DIAGRAMS.md
docs/ARCHITECTURE.md
docs/HARDWARE.md
docs/MODELS.md
docs/INSTALL.md
docs/DESIGN_DECISIONS.md
docs/RECOVERY.md
```

Then inspect:

```text
neurobrain_vision_dev.py
start_neurobrain_vision_dev.py
vision_runtime.py
vision_state.py
vision_service.py
actions.py
hardware.py
```

Before changing anything:

```bash
git status
git log --oneline --decorate -15
```

The known-good integrated reference is `fe85f97`. Documentation checkpoint `a91e524` records that working state.

**Do not infer the active architecture from experimental filenames. Use CURRENT_STATE and Git checkpoints.**

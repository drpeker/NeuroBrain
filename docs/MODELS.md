# NeuroBrain Models

Large model binaries are intentionally not stored in Git.

## Qwen
Model: `/home/drpeker/neurobrain/models/qwen3-1.7b-q4km.gguf`
Runtime: `/home/drpeker/neurobrain/llama.cpp/build/bin/llama-server`

Known working:
```bash
/home/drpeker/neurobrain/llama.cpp/build/bin/llama-server \
  -m /home/drpeker/neurobrain/models/qwen3-1.7b-q4km.gguf \
  --reasoning off -c 2048 --host 127.0.0.1 --port 8080
```
Known llama.cpp build: `b11370-bed0a8566`.
Observed speed: ~7.8–8.6 prompt tok/s and ~3.0–3.2 generation tok/s.

## Whisper
whisper.cpp: `/home/drpeker/whisper.cpp`
Model: `/home/drpeker/whisper.cpp/models/ggml-tiny-q4_0.bin`
VAD: `/home/drpeker/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin`

Known working:
```bash
/home/drpeker/whisper.cpp/build/bin/whisper-server \
  -m /home/drpeker/whisper.cpp/models/ggml-tiny-q4_0.bin \
  -t 4 -l en -nt -bs 3 -bo 3 --vad \
  -vm /home/drpeker/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin \
  -vt 0.50 -vsd 500 --host 127.0.0.1 --port 8081
```
Q4 + beam/best-of 3 is the selected accuracy/speed setting.

## Piper
Executable: `/home/drpeker/neurobrain/venv/bin/piper`
Voice: `/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx`
PiperVoice is loaded once and kept resident.

## YOLO / NCNN
- `/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.param`
- `/home/drpeker/neurobrain/models/yolov8n-ncnn/yolov8n.ncnn.bin`

Working environment observed:
- ncnn `1.0.20260526`
- OpenCV `5.0.0`
- target 320
- confidence 0.35
- NMS IoU 0.45
- 4 NCNN threads
- Vulkan off
- 80 COCO classes
- YOLOv8 DFL bbox decoding

Other model files may exist locally as experiments/fallbacks. Their presence does not mean they are active.

# NeuroBrain Installation / Runtime Layout

## Expected paths
- Project: `/home/drpeker/neurobrain`
- Venv: `/home/drpeker/neurobrain/venv`
- llama.cpp: `/home/drpeker/neurobrain/llama.cpp`
- whisper.cpp: `/home/drpeker/whisper.cpp`
- Piper: `/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx`
- YOLO: `/home/drpeker/neurobrain/models/yolov8n-ncnn/`

## Dependencies
Working code uses numpy, sounddevice, Piper Python API, gpiozero, OpenCV, ncnn and standard-library modules.

## Start
```bash
/home/drpeker/neurobrain/venv/bin/python \
  /home/drpeker/neurobrain/start_neurobrain_vision_dev.py
```

The launcher starts Qwen on 8080, Whisper on 8081, then the integrated NeuroBrain process.

## Diagnostics
```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

One frame:
```bash
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=MJPG \
  --stream-mmap --stream-count=1 \
  --stream-to=/home/drpeker/neurobrain/c270_test.jpg
```

Qwen health: `http://127.0.0.1:8080/health`
Whisper: `http://127.0.0.1:8081/`

Large models, compiled runtimes, venv, WAV files and logs may not be recoverable from Git and must be restored separately.

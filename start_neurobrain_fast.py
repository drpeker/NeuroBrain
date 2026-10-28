"""Start NeuroBrain (fast): Qwen and Whisper servers, then the voice loop.

The servers boot in the background while neurobrain_fast.py loads Piper, YOLO
and the microphone, so the slow steps overlap instead of running one by one.
"""

import os
import subprocess
import time
import signal
import sys

BASE = "/home/drpeker/neurobrain"

LLAMA = f"{BASE}/llama.cpp/build/bin/llama-server"
QWEN_MODEL = f"{BASE}/models/qwen3-1.7b-q4km.gguf"
QWEN_NAME = "Qwen3 1.7B (Q4_K_M)"
QWEN_THREADS = 4
QWEN_CTX = 2048

WHISPER = "/home/drpeker/whisper.cpp/build/bin/whisper-server"
# Measured in the running app (10 spoken questions): tiny Q4_0 80% word match, 1.9 s;
# base Q5_0 88% (all LED commands right), 3.7 s; base f16 89%, 6.4 s; base Q4_0 67%.
# ggml-base-q5_0.bin is made locally: whisper-quantize ggml-base.bin ggml-base-q5_0.bin q5_0
WHISPER_MODEL = "/home/drpeker/whisper.cpp/models/ggml-base-q5_0.bin"
WHISPER_NAME = "Whisper base (Q5_0), English, greedy, 10 s window"
VAD_MODEL = "/home/drpeker/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin"

PIPER_NAME = "Piper en_US-lessac-medium"
VISION_NAME = "YOLOv8n (NCNN), 1 core, 2 fps, paused while NeuroBrain answers"
DEVICE = "C270"
CAMERA = "/dev/video0"

PYTHON = f"{BASE}/venv/bin/python"
NEUROBRAIN = f"{BASE}/neurobrain_fast.py"

qwen = None
whisper = None
qwen_log = None
whisper_log = None


def size_gb(path):
    try:
        return f"{os.path.getsize(path) / 1e9:.2f} GB"
    except OSError:
        return "MISSING"


def check_hardware():
    """The C270 is both camera and microphone; without it NeuroBrain cannot listen."""
    mic = subprocess.run(
        [PYTHON, "-c", f"import sounddevice as sd; print(sd.query_devices({DEVICE!r}, 'input')['name'])"],
        capture_output=True, text=True,
    )
    mic_ok = mic.returncode == 0
    cam_ok = os.path.exists(CAMERA)

    print(f"  Microphone : {mic.stdout.strip() if mic_ok else 'NOT FOUND'}")
    print(f"  Camera     : {CAMERA if cam_ok else 'NOT FOUND'}")

    if not mic_ok:
        print()
        print("ERROR: the C270 webcam (camera + microphone) is not connected.")
        print("  Unplug the C270 and plug it back in, preferably directly into a")
        print("  Pi USB port, then check it with:  lsusb | grep -i 046d")
        print("  Kernel messages:  dmesg | grep -iE 'usb|uvc' | tail")
        sys.exit(1)

    if not cam_ok:
        print("  WARNING: no camera; 'What do you see?' will not work.")


def cleanup():
    global qwen, whisper

    # Ctrl+C and the finally block can both call this; shut down once
    processes = [p for p in (qwen, whisper) if p is not None]
    qwen = whisper = None
    if not processes:
        return

    print("\nShutting down NeuroBrain...")

    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print()
print("==================================================")
print("            NeuroBrain (fast) starting")
print("==================================================")
print(f"  LLM        : {QWEN_NAME}, {size_gb(QWEN_MODEL)}")
print(f"               {os.path.basename(QWEN_MODEL)}")
print(f"               llama.cpp, {QWEN_THREADS} threads, context {QWEN_CTX}, reasoning off, port 8080")
print(f"  Speech in  : {WHISPER_NAME} + Silero VAD, port 8081")
print(f"  Speech out : {PIPER_NAME}")
print(f"  Vision     : {VISION_NAME}")
print("--------------------------------------------------")
print("Hardware check")
check_hardware()
print()

# Stop any Qwen and Whisper servers left over from an earlier run.
subprocess.run(
    ["pkill", "-f", LLAMA],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

subprocess.run(
    ["pkill", "-f", WHISPER],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

time.sleep(1)

try:
    print(f"Starting LLM ({QWEN_NAME})...  log: /tmp/neurobrain-qwen.log")
    qwen_log = open("/tmp/neurobrain-qwen.log", "w")

    qwen = subprocess.Popen(
        [
            LLAMA,
            "-m", QWEN_MODEL,
            "--reasoning", "off",
            "-c", str(QWEN_CTX),
            # All 4 cores: neurobrain_fast pauses YOLO while Qwen runs, so nothing competes
            # (measured: 1.4 tokens/s with YOLO running non-stop, 3.1 tokens/s without)
            "-t", str(QWEN_THREADS),
            # One slot: the system prompt is read once and reused from the cache
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", "8080",
        ],
        stdout=qwen_log,
        stderr=subprocess.STDOUT,
    )

    print(f"Starting speech recognition ({WHISPER_NAME})...  log: /tmp/neurobrain-whisper.log")
    whisper_log = open("/tmp/neurobrain-whisper.log", "w")

    whisper = subprocess.Popen(
        [
            WHISPER,
            "-m", WHISPER_MODEL,
            # Measured on a 3.5 s sentence: 4.1 s with -t 3 -bs 3 -bo 3, 0.95 s with these, same text
            "-t", "4",
            "-l", "en",
            "-nt",
            "-bs", "1",
            "-bo", "1",
            # ~10 s audio window instead of 30 s; longer speech may be cut
            "-ac", "512",
            "--vad",
            "-vm", VAD_MODEL,
            "-vt", "0.50",
            "-vsd", "500",
            "--host", "127.0.0.1",
            "--port", "8081",
        ],
        stdout=whisper_log,
        stderr=subprocess.STDOUT,
    )

    print("Starting NeuroBrain; it loads Piper, YOLO and the microphone while the servers boot.")
    print()

    subprocess.run([PYTHON, NEUROBRAIN])

finally:
    cleanup()

    if qwen_log is not None:
        qwen_log.close()

    if whisper_log is not None:
        whisper_log.close()

import subprocess
import time
import urllib.request
import signal
import sys

BASE = "/home/drpeker/neurobrain"

LLAMA = f"{BASE}/llama.cpp/build/bin/llama-server"
QWEN_MODEL = f"{BASE}/models/qwen3-1.7b-q4km.gguf"

WHISPER = "/home/drpeker/whisper.cpp/build/bin/whisper-server"
WHISPER_MODEL = "/home/drpeker/whisper.cpp/models/ggml-tiny.bin"
VAD_MODEL = "/home/drpeker/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin"

PYTHON = f"{BASE}/venv/bin/python"
NEUROBRAIN = f"{BASE}/neurobrain.py"

qwen = None
whisper = None
qwen_log = None
whisper_log = None


def wait_for(url, name, timeout=60):
    print(f"{name} bekleniyor...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                pass
            print(f"{name}: HAZIR")
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(f"{name} zamanında başlayamadı.")


def cleanup():
    global qwen, whisper

    print("\nNeuroBrain kapatılıyor...")

    for process in (qwen, whisper):
        if process is not None and process.poll() is None:
            process.terminate()

    for process in (qwen, whisper):
        if process is not None and process.poll() is None:
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
print("==============================")
print("       NeuroBrain Start")
print("==============================")
print()

# Önceden kalmış Qwen ve Whisper server süreçlerini kapat.
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
    print("Qwen başlatılıyor...")
    qwen_log = open("/tmp/neurobrain-qwen.log", "w")

    qwen = subprocess.Popen(
        [
            LLAMA,
            "-m", QWEN_MODEL,
            "--reasoning", "off",
            "-c", "2048",
            "--host", "127.0.0.1",
            "--port", "8080",
        ],
        stdout=qwen_log,
        stderr=subprocess.STDOUT,
    )

    print("Whisper başlatılıyor...")
    whisper_log = open("/tmp/neurobrain-whisper.log", "w")

    whisper = subprocess.Popen(
        [
            WHISPER,
            "-m", WHISPER_MODEL,
            "-t", "4",
            "-l", "en",
            "-nt",
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

    wait_for("http://127.0.0.1:8080/health", "Qwen")
    wait_for("http://127.0.0.1:8081/", "Whisper")

    print()
    print("==============================")
    print(" Qwen    : HAZIR")
    print(" Whisper : HAZIR")
    print(" Piper   : NeuroBrain yükleyecek")
    print("==============================")
    print()

    subprocess.run([PYTHON, NEUROBRAIN])

finally:
    cleanup()

    if qwen_log is not None:
        qwen_log.close()

    if whisper_log is not None:
        whisper_log.close()

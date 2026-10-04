import sounddevice as sd
import numpy as np
import wave
import time
import subprocess
import json
import urllib.request

WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

PIPER = "/home/drpeker/neurobrain/venv/bin/piper"
PIPER_MODEL = "/home/drpeker/neurobrain/tts/tr_TR-dfki-medium.onnx"

AUDIO = "/tmp/neurobrain_input.wav"
OUTPUT = "/tmp/neurobrain_output.wav"

DEVICE = 1
RATE = 16000
BLOCK = 1024

THRESHOLD = 0.03
SILENCE_TIME = 0.8

SYSTEM = """
Adın NeuroBrain.
Raspberry Pi üzerinde çalışan bir robot yapay zekasısın.
Kullanıcıyla Türkçe konuş.
Kısa, doğal ve doğrudan cevap ver.
Gereksiz açıklama yapma.
"""

def record_speech():

    print("🎤 Konuşmanızı bekliyorum...")

    frames = []
    recording = False
    last_sound = None

    with sd.InputStream(
        device=DEVICE,
        samplerate=RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK
    ) as stream:

        while True:

            data, overflowed = stream.read(BLOCK)

            audio = data.astype(np.float32) / 32768.0
            level = np.sqrt(np.mean(audio ** 2))

            if not recording:

                if level > THRESHOLD:
                    recording = True
                    last_sound = time.monotonic()
                    frames.append(data.copy())
                    print("🔴 Dinliyorum...")

            else:

                frames.append(data.copy())

                if level > THRESHOLD:
                    last_sound = time.monotonic()

                elif time.monotonic() - last_sound >= SILENCE_TIME:
                    print("⏹️ Tamam.")
                    break

    with wave.open(AUDIO, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(x.tobytes() for x in frames))


print("\n=== NeuroBrain ===")
print("Whisper : RAM'de hazır (8081)")
print("Qwen    : RAM'de hazır (8080)")
print("Mikrofon: C270")
print("Sistem hazır.\n")


while True:

    # -------------------------------------------------
    # SES KAYDI
    # -------------------------------------------------

    record_speech()

    # -------------------------------------------------
    # WHISPER
    # -------------------------------------------------

    whisper = subprocess.run([
        "curl", "-s",
        WHISPER_URL,
        "-F", f"file=@{AUDIO}",
        "-F", "response_format=json",
        "-F", "language=tr"
    ], capture_output=True, text=True)

    try:
        result = json.loads(whisper.stdout)
        text = result["text"].strip()
    except Exception:
        print("❌ Whisper hatası:", whisper.stdout)
        continue

    if not text:
        print("Sessizlik.\n")
        continue

    print("👤", text)

    # -------------------------------------------------
    # QWEN
    # -------------------------------------------------

    data = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}
        ],
        "max_tokens": 100
    }

    request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:

        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ Qwen hatası:", e)
        continue

    if not answer:
        print("❌ Qwen cevap üretmedi.")
        continue

    print("🤖", answer)

    # -------------------------------------------------
    # PIPER
    # -------------------------------------------------

    p = subprocess.Popen([
        PIPER,
        "--model", PIPER_MODEL,
        "--output_file", OUTPUT
    ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True
    )

    p.communicate(answer)

    subprocess.run([
        "aplay", OUTPUT
    ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print()

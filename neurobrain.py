import sounddevice as sd
import numpy as np
import wave
import time
import subprocess
import json
import urllib.request

from piper import PiperVoice

WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

PIPER_MODEL = "/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx"

AUDIO = "/tmp/neurobrain_input.wav"
OUTPUT = "/tmp/neurobrain_output.wav"

DEVICE = 1
RATE = 16000
BLOCK = 1024

THRESHOLD = 0.03
SILENCE_TIME = 0.8

SYSTEM = """
You are NeuroBrain, an AI running on a Raspberry Pi robot.
Always speak English.
Answer briefly, naturally, and accurately.
Use one or two short sentences whenever possible.
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
print("Piper yükleniyor...")

t0 = time.monotonic()
piper_voice = PiperVoice.load(PIPER_MODEL)

print(f"Piper   : RAM'de hazır ({time.monotonic()-t0:.2f} sn)")
print("Whisper : RAM'de hazır (8081)")
print("Qwen    : RAM'de hazır (8080)")
print("Mikrofon: C270")
print("Sistem hazır.\n")


while True:

    record_speech()

    # WHISPER

    t0 = time.monotonic()

    whisper = subprocess.run([
        "curl", "-s",
        WHISPER_URL,
        "-F", f"file=@{AUDIO}",
        "-F", "response_format=json",
        "-F", "language=en"
    ], capture_output=True, text=True)

    whisper_time = time.monotonic() - t0

    try:
        result = json.loads(whisper.stdout)
        text = result["text"].strip()
    except Exception:
        print("❌ Whisper hatası:", whisper.stdout)
        continue

    if not text:
        print(f"Sessizlik. [Whisper {whisper_time:.2f} sn]\n")
        continue

    print(f"👤 {text}  [Whisper {whisper_time:.2f} sn]")

    # QWEN

    data = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}
        ],
        "max_tokens": 60
    }

    request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t0 = time.monotonic()

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ Qwen hatası:", e)
        continue

    qwen_time = time.monotonic() - t0

    if not answer:
        print("❌ Qwen cevap üretmedi.")
        continue

    print(f"🤖 {answer}  [Qwen {qwen_time:.2f} sn]")

    # PIPER - MODEL ZATEN RAM'DE

    t0 = time.monotonic()

    with wave.open(OUTPUT, "wb") as wav:
        piper_voice.synthesize_wav(answer, wav)

    piper_time = time.monotonic() - t0

    print(f"🔊 [Piper {piper_time:.2f} sn]")

    subprocess.run(
        ["aplay", OUTPUT],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print()

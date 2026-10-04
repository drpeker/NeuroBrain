import sounddevice as sd
import numpy as np
import wave
import time
import subprocess
import json
import re
import urllib.request
import actions

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

    # QWEN INTENT ROUTER
    ROUTER_SYSTEM = """
You are the intent router for NeuroBrain, a Raspberry Pi robot.
Interpret the user's intention. Speech recognition may contain errors.

Hardware:
- An LED is connected to GPIO 17.
- GPIO 17 is authorized for control.

Allowed actions:
GPIO_SET
GPIO_TOGGLE
GPIO_READ
CHAT

Return ONLY one JSON object.

Examples:
Turn the LED on -> {"action":"GPIO_SET","pin":17,"value":1}
Turn the LED off -> {"action":"GPIO_SET","pin":17,"value":0}
Toggle GPIO17 -> {"action":"GPIO_TOGGLE","pin":17}
Change the logic level of GPIO 17 -> {"action":"GPIO_TOGGLE","pin":17}
What is the state of GPIO17? -> {"action":"GPIO_READ","pin":17}
Ordinary conversation -> {"action":"CHAT"}

Do not claim that an action was executed.
"""

    router_data = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": text}
        ],
        "temperature": 0,
        "max_tokens": 40
    }

    router_request = urllib.request.Request(
        LLM_URL,
        data=json.dumps(router_data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        t0 = time.monotonic()

        with urllib.request.urlopen(router_request) as response:
            router_result = json.loads(response.read().decode("utf-8"))

        router_raw = router_result["choices"][0]["message"]["content"].strip()
        router_time = time.monotonic() - t0

        print(f"🧠 {router_raw}  [Router {router_time:.2f} sn]")

        intent = json.loads(router_raw)

    except Exception as e:
        print("❌ Router hatası:", e)
        continue

    action = intent.get("action", "CHAT")
    answer = None

    if action == "GPIO_SET":
        pin = intent.get("pin")
        value = intent.get("value")

        result = actions.gpio_execute("SET", pin, value)
        print("⚙️", result)

        if result["success"]:
            answer = f"GPIO {pin} is now {'high' if result['value'] else 'low'}."
        else:
            answer = result["message"] + "."

    elif action == "GPIO_TOGGLE":
        pin = intent.get("pin")

        result = actions.gpio_execute("TOGGLE", pin)
        print("⚙️", result)

        if result["success"]:
            answer = f"GPIO {pin} is now {'high' if result['value'] else 'low'}."
        else:
            answer = result["message"] + "."

    elif action == "GPIO_READ":
        pin = intent.get("pin")

        result = actions.gpio_execute("READ", pin)
        print("⚙️", result)

        if result["success"]:
            answer = f"GPIO {pin} is {'high' if result['value'] else 'low'}."
        else:
            answer = result["message"] + "."

    elif action != "CHAT":
        print(f"⚠️ Unknown router action rejected: {action}")
        answer = "That action is not available."

    if answer is not None:
        print(f"🤖 {answer}  [Core]")

        t0 = time.monotonic()

        with wave.open(OUTPUT, "wb") as wav:
            piper_voice.synthesize_wav(answer, wav)

        piper_time = time.monotonic() - t0
        print(f"🔊 [Piper {piper_time:.2f} sn]")

        subprocess.run(["aplay", "-q", OUTPUT])
        continue

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

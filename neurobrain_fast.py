"""NeuroBrain (fast): voice loop with a keyword router, streaming LLM and parallel TTS.

Started by start_neurobrain_fast.py, which launches the Qwen and Whisper servers.
This program loads Piper, YOLO and the microphone while those servers boot.
"""

import collections
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import warnings
import wave

import numpy as np
import sounddevice as sd

# Without lgpio, gpiozero tries several pin libraries first; hide that noise
warnings.filterwarnings("ignore", module="gpiozero")
import actions
from vision_runtime_fast import VisionRuntime

from piper import PiperVoice

WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLM_HEALTH = "http://127.0.0.1:8080/health"
WHISPER_HEALTH = "http://127.0.0.1:8081/"

PIPER_MODEL = "/home/drpeker/neurobrain/tts/en_US-lessac-medium.onnx"

DEVICE = "C270"  # select by name: the ALSA card number can change between boots
RATE = 16000
BLOCK = 1024  # 64 ms per block

# Speech detection follows the room: the C270 hears ~0.024 RMS of background noise
# (peaks 0.04), so a fixed 0.03 threshold mistook silence for speech and recordings
# ran to MAX_RECORD. The noise floor is tracked while nobody speaks.
MIN_START = 0.03  # RMS that starts a recording, at least
MAX_START = 0.06  # ... but never more, or a normal voice would not be heard
START_FACTOR = 2.0  # ... and at least this many times the noise floor
STOP_FACTOR = 1.5  # quieter than this many times the noise floor counts as silence
SILENCE_TIME = 0.8  # seconds of quiet that end a recording
MAX_RECORD = 8.0  # Whisper window is ~10 s (-ac 512); noise must not record forever
PRE_ROLL = 5  # blocks (~0.3 s) kept from before the trigger, so first syllables survive

# No Whisper prompt on purpose: a prompt full of commands made Whisper turn unclear
# speech into those commands ("capital of Finland" -> "set GPIO 17 high") and
# sometimes repeat the prompt in a loop for 6-11 s.

# Physical commands are only executed when Whisper is confident about the words,
# and questions heard this badly are asked again instead of sent to the LLM
MIN_COMMAND_LOGPROB = -0.5
MIN_CHAT_LOGPROB = -0.6

MAX_TOKENS = 60
# Chat memory is off: llama.cpp could not reuse the cache for earlier answers, so every
# question re-read 45-60 tokens (+5-7 s), and one misheard answer coloured the next ones.
MEMORY_TURNS = 0  # chat exchanges remembered; 0 disables memory
MEMORY_IDLE = 120.0  # seconds of silence after which the conversation starts fresh

SYSTEM = """
Your name is NeuroBrain.
Always speak English.
Answer briefly, naturally, and accurately.
Use one or two short sentences whenever possible.
"""


# --- Startup helpers ---

def post_json(url, data, timeout=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wait_for(url, name, started, timeout=90):
    while time.monotonic() - started < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1):
                pass
            return
        except Exception:
            time.sleep(0.3)
    print(f"\nERROR: {name} server did not start within {timeout} s. See /tmp/neurobrain-*.log")
    sys.exit(1)


def llm_model_name():
    try:
        with urllib.request.urlopen(LLM_URL.replace("chat/completions", "models"), timeout=2) as r:
            return json.load(r)["data"][0]["id"].rsplit("/", 1)[-1]
    except Exception:
        return "unknown"


# --- Microphone: one stream for the whole session ---

class Microphone:

    def __init__(self):
        self.stream = sd.InputStream(
            device=DEVICE,
            samplerate=RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCK
        )
        self.stream.start()
        self.noise = self.calibrate()

    def level(self, data):
        return float(np.sqrt(np.mean((data.astype(np.float32) / 32768.0) ** 2)))

    def calibrate(self, seconds=1.0):
        """Measure the room's background noise (RMS) at startup."""
        levels = [self.level(self.stream.read(BLOCK)[0]) for _ in range(int(seconds * RATE / BLOCK))]
        return float(np.median(levels))

    def thresholds(self):
        start = min(MAX_START, max(MIN_START, self.noise * START_FACTOR))
        return start, min(start * 0.8, max(self.noise * STOP_FACTOR, start * 0.6))

    def drain(self):
        """Drop audio captured while NeuroBrain was thinking or speaking."""
        available = self.stream.read_available
        if available:
            self.stream.read(available)

    def record(self):
        """Wait for speech and return (wav bytes, seconds of audio)."""
        print("🎤 Listening for speech...")
        self.drain()

        before = collections.deque(maxlen=PRE_ROLL)
        frames = []
        recording = False
        started = last_sound = 0.0

        start_level, stop_level = self.thresholds()

        while True:
            data, _ = self.stream.read(BLOCK)
            level = self.level(data)
            now = time.monotonic()

            if not recording:
                before.append(data.copy())
                if level > start_level:
                    recording = True
                    started = last_sound = now
                    frames.extend(before)
                    print("🔴 Recording...")
                continue

            frames.append(data.copy())

            if level > stop_level:
                last_sound = now
            elif now - last_sound >= SILENCE_TIME:
                print("⏹️ Done.")
                break

            if now - started >= MAX_RECORD:
                print("⏹️ Done (max length).")
                break

        # Follow slow changes in room noise (only from the quiet blocks before speech)
        quiet = [self.level(b) for b in list(before)[:-1]]
        if quiet:
            self.noise = 0.8 * self.noise + 0.2 * float(np.median(quiet))

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(x.tobytes() for x in frames))

        return buffer.getvalue(), len(frames) * BLOCK / RATE


def transcribe(wav_bytes):
    """Send audio to the Whisper server from memory (no temp file, no curl process).

    Returns (text, confidence); confidence is the lowest segment avg_logprob
    (0 is certain, below about -0.5 the words are often guessed).
    """
    boundary = uuid.uuid4().hex
    fields = {"response_format": "verbose_json", "language": "en"}

    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
               f'Content-Type: audio/wav\r\n\r\n'.encode())
    body.write(wav_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        WHISPER_URL,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)

    scores = [seg.get("avg_logprob", 0.0) for seg in result.get("segments", [])]
    return dedupe(result.get("text", "").strip()), min(scores, default=0.0)


def dedupe(text):
    """Whisper sometimes repeats a sentence ("Is it on? Is it on? Is it on?"): keep one."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not out or sentence.lower().strip(" .!?") != out[-1].lower().strip(" .!?"):
            out.append(sentence)
    return " ".join(out)


# --- Keyword router ---

# Known commands are matched by keywords; the old Qwen router added 5-45 s per request.
# Physical actions still go through actions.py; the LLM never touches hardware.
LED = r"\b(led|light|lamp|lights|gpio|pin ?17)\b"
# A switch command needs a command verb, or must be a short phrase like "LED on"
VERB = r"^(please |can you |could you |would you )?(turn|switch|set|put|make|power|light|toggle|flip)\b"
QUESTION = r"^(what|who|where|when|why|how|which|tell|can|could|do|does|is|are)\b"

# Common Whisper mishearings: "L-E-D", "DPI0.17", "GPI 17", "CPIO" ...
FIXES = [
    (r"\bl e d\b", "led"),
    (r"\bg p i o\b", "gpio"),
    (r"\b(gpio|gpi|gpo|dpi|cpio|jpio|gbio)(?: 0)?(?: 17)?\b", "gpio"),
    (r"\b(high)\b", "on"),
    (r"\b(low)\b", "off"),
]
ROUTES = [
    (r"\b(what|which|whats)\b.*\b(see|seeing|front of you|camera|look)\b"
     r"|\bcan you see\b|\bdo you see\b|\blook around\b", {"action": "VISION_QUERY"}),
    (r"\b(toggle|flip|change)\b.*" + LED, {"action": "GPIO_TOGGLE", "pin": 17}),
    # Questions first: "is the LED on?" is a read, not a switch
    (r"^(is|are)\b.*" + LED + r"|\bif\b.*" + LED + r".*\b(on|off)\b|\b(state|status|level)\b.*" + LED + r"|" + LED + r".*\b(state|status|level)\b",
     {"action": "GPIO_READ", "pin": 17}),
    (r"\boff\b.*" + LED + r"|" + LED + r".*\boff\b", {"action": "GPIO_SET", "pin": 17, "value": 0}),
    (r"\bon\b.*" + LED + r"|" + LED + r".*\bon\b", {"action": "GPIO_SET", "pin": 17, "value": 1}),
]
SWITCHES = ("GPIO_SET", "GPIO_TOGGLE")


def route(text):
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = re.sub(r"(?<=[a-z])(?=[0-9])", " ", t)  # "dpi0" -> "dpi 0"
    t = re.sub(r" +", " ", t).strip()
    for pattern, replacement in FIXES:
        t = re.sub(pattern, replacement, t)
    for pattern, intent in ROUTES:
        if re.search(pattern, t):
            if intent["action"] in SWITCHES and not (
                    re.search(VERB, t) or (len(t.split()) <= 4 and not re.search(QUESTION, t))):
                continue  # mentions the LED but is not an order: let the LLM answer
            return dict(intent)
    return {"action": "CHAT"}


# --- Speech output: synthesis and playback run in parallel ---

sentences = queue.Queue()  # text waiting for Piper
playback = queue.Queue()  # audio waiting for the speaker


def synth_worker():
    while True:
        sentence = sentences.get()
        try:
            audio = b"".join(chunk.audio_int16_bytes for chunk in piper_voice.synthesize(sentence))
            playback.put(audio)
        except Exception as e:
            print("❌ Piper error:", e)
        finally:
            sentences.task_done()


def play_worker():
    while True:
        audio = playback.get()
        try:
            subprocess.run(
                ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(piper_rate)],
                input=audio
            )
        finally:
            playback.task_done()


def say(sentence):
    sentence = sentence.strip()
    if sentence:
        sentences.put(sentence)


def wait_until_spoken():
    sentences.join()
    playback.join()


# --- LLM chat with short memory ---

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
history = []  # [(user, assistant), ...]
last_chat = 0.0


def chat_stream(text):
    """One streaming Qwen call; every finished sentence is spoken immediately."""
    global history, last_chat

    # Memory is cleared instead of trimmed: the cached prompt prefix stays valid,
    # so a reset costs nothing, while dropping the oldest turn would re-read everything.
    if len(history) >= MEMORY_TURNS or time.monotonic() - last_chat > MEMORY_IDLE:
        history = []

    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in history:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": text})

    data = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "cache_prompt": True,
    }

    t0 = time.monotonic()
    first = None
    finish = None
    pending, spoken = "", []

    with post_json(LLM_URL, data) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            choice = json.loads(payload)["choices"][0]
            finish = choice.get("finish_reason") or finish
            pending += choice.get("delta", {}).get("content") or ""
            parts = SENTENCE_END.split(pending)
            for sentence in parts[:-1]:
                if first is None:
                    first = time.monotonic() - t0
                say(sentence)
                spoken.append(sentence.strip())
            pending = parts[-1]

    # Hit the token limit mid-sentence: drop the fragment unless it is all we have
    truncated = finish == "length" and spoken
    if pending.strip() and not truncated:
        if first is None:
            first = time.monotonic() - t0
        say(pending)
        spoken.append(pending.strip())

    answer = " ".join(spoken)
    if MEMORY_TURNS and answer:
        history.append((text, answer))
    last_chat = time.monotonic()

    return answer, first, time.monotonic() - t0, truncated


def gpio_answer(operation, intent):
    result = actions.gpio_execute(operation, intent["pin"], intent.get("value"))
    print("⚙️", result)
    if not result["success"]:
        return result["message"] + "."
    verb = "is" if operation == "READ" else "is now"
    return f"GPIO {intent['pin']} {verb} {'high' if result['value'] else 'low'}."


# --- Startup: load local models while the servers boot ---

boot = time.monotonic()
print("=== NeuroBrain (fast) ===")

t0 = time.monotonic()
piper_voice = PiperVoice.load(PIPER_MODEL)
piper_rate = piper_voice.config.sample_rate
print(f"  TTS     : Piper {PIPER_MODEL.rsplit('/', 1)[-1]} ({time.monotonic()-t0:.1f} s)")

threading.Thread(target=synth_worker, name="NeuroBrainSynth", daemon=True).start()
threading.Thread(target=play_worker, name="NeuroBrainPlayer", daemon=True).start()

try:
    mic = Microphone()
except (ValueError, sd.PortAudioError) as e:
    print(f"\nERROR: microphone {DEVICE!r} is not available: {e}")
    print("  The C270 webcam is probably disconnected. Replug it and start NeuroBrain again.")
    sys.exit(1)
start_level, stop_level = mic.thresholds()
print(f"  Mic     : {DEVICE}, always open, {PRE_ROLL * BLOCK / RATE:.1f} s pre-roll, "
      f"noise {mic.noise:.3f} -> start {start_level:.3f} / stop {stop_level:.3f} RMS")

vision_runtime = VisionRuntime()
vision_runtime.start()

wait_for(WHISPER_HEALTH, "Whisper", boot)
print(f"  STT     : Whisper server ready, port 8081 ({time.monotonic()-boot:.1f} s after start)")
wait_for(LLM_HEALTH, "LLM", boot)
print(f"  LLM     : {llm_model_name()} ready, port 8080 ({time.monotonic()-boot:.1f} s after start)")

print("  Prompt  : loading system prompt into the LLM cache...", end="", flush=True)
t0 = time.monotonic()
try:
    # Read the system prompt once now, so the first question is not slower
    warm = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi"}],
            "max_tokens": 1, "cache_prompt": True}
    post_json(LLM_URL, warm, timeout=120).read()
    print(f" done ({time.monotonic()-t0:.1f} s)")
except Exception as e:
    print(f" failed: {e}")

print(f"  Memory  : " + (f"last {MEMORY_TURNS} chat exchanges, cleared after {MEMORY_IDLE:.0f} s idle"
                        if MEMORY_TURNS else "off (every question is answered on its own)"))
print("  Router  : LED / GPIO 17 and 'What do you see?' answered without the LLM")
print(f"System ready ({time.monotonic()-boot:.1f} s).\n")


# --- Main loop ---

def main_loop():
    while True:
        try:
            wav_bytes, seconds = mic.record()
        except sd.PortAudioError as e:
            print(f"\nERROR: microphone {DEVICE!r} stopped: {e}")
            print("  The C270 webcam is probably disconnected. Replug it and start NeuroBrain again.")
            sys.exit(1)

        # YOLO rests from here until the answer has been spoken: Whisper, Qwen and
        # Piper all need the CPU more than a fresh object list does.
        vision_runtime.pause()
        try:
            t0 = time.monotonic()
            try:
                text, confidence = transcribe(wav_bytes)
            except Exception as e:
                print("❌ Whisper error:", e)
                continue
            whisper_time = time.monotonic() - t0

            if not text:
                print(f"(silence) [Whisper {whisper_time:.2f} s, audio {seconds:.1f} s]\n")
                continue

            print(f"👤 {text}  [Whisper {whisper_time:.2f} s, audio {seconds:.1f} s, confidence {confidence:.2f}]")

            # ROUTING (keywords, no LLM)
            intent = route(text)
            action = intent["action"]
            answer = None
            print(f"🧭 {intent}")

            if action in SWITCHES and confidence < MIN_COMMAND_LOGPROB:
                answer = "Sorry, I am not sure I heard that right. Please say the command again."
            elif action == "CHAT" and confidence < MIN_CHAT_LOGPROB:
                answer = "Sorry, I did not catch that. Could you say it again?"
            elif action == "GPIO_SET":
                answer = gpio_answer("SET", intent)
            elif action == "GPIO_TOGGLE":
                answer = gpio_answer("TOGGLE", intent)
            elif action == "GPIO_READ":
                answer = gpio_answer("READ", intent)
            elif action == "VISION_QUERY":
                # Take a fresh look before answering
                vision_runtime.resume()
                if vision_runtime.wait_fresh(max_age=1.0, timeout=2.0):
                    answer = vision_runtime.summary()
                else:
                    answer = "My camera is not providing a current view."
                vision_runtime.pause()

            if answer is not None:
                print(f"🤖 {answer}  [Core]")
                say(answer)
                wait_until_spoken()
                print()
                continue

            # QWEN (one streaming call)
            try:
                answer, first, total, truncated = chat_stream(text)
            except Exception as e:
                print("❌ LLM error:", e)
                continue

            if not answer:
                print("❌ LLM gave no answer.")
                continue

            note = ", cut at token limit" if truncated else ""
            print(f"🤖 {answer}  [LLM first sentence {first:.2f} s, total {total:.2f} s{note}]")
            wait_until_spoken()  # do not listen while speaking (it would hear itself)
            print()

        finally:
            vision_runtime.resume()


try:
    main_loop()
except KeyboardInterrupt:
    pass
finally:
    vision_runtime.stop()
    print("NeuroBrain stopped.")

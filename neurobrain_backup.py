import subprocess
import json
import urllib.request

WHISPER_URL = "http://127.0.0.1:8081/inference"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

PIPER = "/home/drpeker/neurobrain/venv/bin/piper"
PIPER_MODEL = "/home/drpeker/neurobrain/tts/tr_TR-dfki-medium.onnx"

AUDIO = "/tmp/neurobrain_input.wav"
OUTPUT = "/tmp/neurobrain_output.wav"

SYSTEM = """
Adın NeuroBrain.
Raspberry Pi üzerinde çalışan bir robot yapay zekasısın.
Kullanıcıyla Türkçe konuş.
Kısa, doğal ve doğrudan cevap ver.
Gereksiz açıklama yapma.
"""

print("\n=== NeuroBrain ===")
print("Whisper : RAM'de hazır (8081)")
print("Qwen    : RAM'de hazır (8080)")
print("Sistem hazır.\n")

while True:

    print("🎤 Konuşun...")

    # 5 saniye kayıt
    subprocess.run([
        "arecord",
        "-D", "plughw:1,0",
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", "5",
        AUDIO
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL)

    # Whisper server
    whisper = subprocess.run([
        "curl", "-s",
        WHISPER_URL,
        "-F", f"file=@{AUDIO}",
        "-F", "response_format=json",
        "-F", "language=auto"
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

    # Qwen server
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

    # Piper TTS
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
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL)

    print()


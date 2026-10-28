"""Audio test data: spoken scripts (Windows TTS), phone-line degradation, and real LibriSpeech clips.

Synthetic voices are clean and uniform, so they test the pipeline mechanics; the LibriSpeech clips are
real human speech and give the honest word-error-rate figure.
"""

import io
import json
from pathlib import Path
import re
import subprocess
import sys
import wave

import httpx
import numpy as np

from evaluation.datasets import CACHE_DIR, load_scam_cases

AUDIO_DIR = CACHE_DIR / "audio"

# Written as natural spoken scripts. label: deceptive = a scam call/voice note.
SCAM_SCRIPTS = [
    "This is Officer Daniels from the Social Security Administration. Your number has been linked to illegal activity "
    "and your benefits will be frozen. Do not tell anyone about this call. You must pay the fine today using gift cards.",
    "Hello, this is Microsoft technical support. We detected a virus on your computer. Please install AnyDesk so our "
    "engineer can get remote access and fix it right away.",
    "This is the fraud department of your bank. Someone tried to withdraw five thousand dollars from your account. "
    "To cancel it, read me the one time password we just sent you.",
    "Hi mum, it's me. I dropped my phone in the sea so this is my new number. I'm in trouble and need you to send "
    "two hundred pounds for the repair, please hurry.",
    "Congratulations, you have won a free cruise! To claim your prize you must pay a small processing fee by wire "
    "transfer today.",
    "This is a courier delivery notice. Your parcel could not be delivered. Pay a small redelivery fee with a prepaid "
    "card or the package will be returned.",
    "Send point one bitcoin to this address and we will return double within one hour. This is a limited time giveaway.",
    "Grandma, please don't tell mom, I've had an accident and I need bail money. Can you send three hundred dollars "
    "by Western Union?",
]
BENIGN_SCRIPTS = [
    "Hi, it's Sam. Just confirming dinner at seven on Friday. Let me know if the restaurant works for you.",
    "Reminder from the clinic: your appointment is tomorrow at ten in the morning. Please arrive fifteen minutes early.",
    "Hey, I'm running about ten minutes late. Start the meeting without me and I will catch up.",
    "This is Amazon customer service calling about the delivery you asked about. Your parcel will arrive tomorrow afternoon.",
    "Hello, this is your dentist's office. We would like to reschedule your cleaning to next Tuesday. Please call us back when convenient.",
    "Mum, can you pick up some milk on the way home? Thanks, see you soon.",
    "The technical support team has resolved your ticket. No further action is needed from you.",
    "Please stay on the line while I transfer you to the billing department.",
]
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]

_POWERSHELL = r"""
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json
foreach ($item in $items) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($item.voice)
    $s.SetOutputToWaveFile($item.file)
    $s.Speak($item.text)
    $s.Dispose()
}
"""


def spoken_items() -> list[dict[str, str]]:
    """(name, text, label) for every spoken test script and every scam-suite message."""
    items = [{"name": f"scam_script_{i}", "text": t, "label": "deceptive"} for i, t in enumerate(SCAM_SCRIPTS)]
    items += [{"name": f"benign_script_{i}", "text": t, "label": "not_deceptive"} for i, t in enumerate(BENIGN_SCRIPTS)]
    for case in load_scam_cases():
        items.append(
            {"name": f"message_{case['id']}", "text": case["message"], "label": "deceptive" if case["label"] == "deceptive" else "not_deceptive"}
        )
    return items


def ensure_tts_audio() -> list[dict[str, str]]:
    """Synthesise every item once (Windows only) and return items with file paths."""
    if sys.platform != "win32":
        raise RuntimeError("Spoken test audio is generated with Windows text-to-speech.")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    items = spoken_items()
    todo = []
    for index, item in enumerate(items):
        item["file"] = str(AUDIO_DIR / f"{item['name']}.wav")
        if not Path(item["file"]).exists():
            todo.append({"file": item["file"], "text": item["text"], "voice": VOICES[index % len(VOICES)]})
    if todo:
        manifest = AUDIO_DIR / "manifest.json"
        manifest.write_text(json.dumps(todo), encoding="utf-8")
        script = AUDIO_DIR / "synthesise.ps1"
        script.write_text(_POWERSHELL, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(manifest)],
            check=True,
            timeout=600,
        )
        missing = [item["file"] for item in todo if not Path(item["file"]).exists()]
        if missing:
            raise RuntimeError(f"Text-to-speech did not produce {len(missing)} files, e.g. {missing[0]}")
    return items


def degrade_to_phone_line(wav_bytes: bytes, snr_db: float = 15.0, seed: int = 3) -> bytes:
    """Telephone bandwidth (8 kHz) plus white noise at the given signal-to-noise ratio."""
    with wave.open(io.BytesIO(wav_bytes)) as source:
        rate, channels, width = source.getframerate(), source.getnchannels(), source.getsampwidth()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16).astype(np.float64)
    if width != 2:
        raise ValueError("expected 16-bit PCM audio")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    factor = max(1, round(rate / 8000))
    smoothed = np.convolve(samples, np.ones(factor) / factor, mode="same")  # crude low-pass before decimating
    narrow = smoothed[::factor]
    noise = np.random.default_rng(seed).normal(0, np.sqrt(np.mean(narrow**2) / 10 ** (snr_db / 10)), narrow.shape)
    out = np.clip(narrow + noise, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as sink:
        sink.setnchannels(1)
        sink.setsampwidth(2)
        sink.setframerate(rate // factor)
        sink.writeframes(out.tobytes())
    return buffer.getvalue()


def load_librispeech(n: int = 25) -> list[dict[str, object]]:
    """Real human speech with reference transcripts (LibriSpeech validation sample). Cached."""
    meta_path = AUDIO_DIR / "librispeech.json"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        response = httpx.get(
            "https://datasets-server.huggingface.co/first-rows",
            params={"dataset": "hf-internal-testing/librispeech_asr_dummy", "config": "clean", "split": "validation"},
            timeout=60.0, follow_redirects=True,
        )
        response.raise_for_status()
        rows = []
        for index, row in enumerate(response.json()["rows"]):
            path = AUDIO_DIR / f"libri_{index}.wav"
            path.write_bytes(httpx.get(row["row"]["audio"][0]["src"], timeout=60.0, follow_redirects=True).content)
            rows.append({"file": path.name, "text": row["row"]["text"]})
        meta_path.write_text(json.dumps(rows), encoding="utf-8")
    rows = json.loads(meta_path.read_text(encoding="utf-8"))[:n]
    return [{"audio": (AUDIO_DIR / r["file"]).read_bytes(), "text": r["text"]} for r in rows]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the reference length."""
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        raise ValueError("reference must contain words.")
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)

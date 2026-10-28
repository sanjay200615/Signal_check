"""A multi-channel scam benchmark: the same scenario delivered as text, link, screenshot and voice note.

Four scenario types for each of six brands (24 cases). Labels come from construction:
  scam_phish      text + lookalike link + screenshot of a fake sign-in page            -> deceptive
  scam_campaign   mild pressure in a text AND a mild demand in a voice note, no link   -> deceptive
  brand_conflict  text names one brand, the link and screenshot belong to another      -> not_genuine
  genuine         receipt text + official link + screenshot of the official page + voice reminder -> genuine

The scenarios are synthetic (rendered pages, text-to-speech), so they test the reasoning, not real-world
robustness. The brand map and cue patterns were written before this benchmark but with scams like these in mind.
"""

import json
from pathlib import Path
import subprocess
import sys

from evaluation.audio_data import AUDIO_DIR, VOICES, _POWERSHELL
from evaluation.synth_images import render_login_page

BRANDS = ["paypal", "amazon", "microsoft", "apple", "netflix", "google"]
ACCENTS = {"paypal": (0, 48, 135), "amazon": (35, 47, 62), "microsoft": (0, 120, 212), "apple": (60, 60, 60), "netflix": (180, 20, 25), "google": (66, 133, 244)}


def _official(reputation, brand: str) -> str:
    return sorted(reputation.brands[brand])[0]


def build_cases(reputation) -> list[dict]:
    """Return cases: name, gold, and the parts available for each channel."""
    cases = []
    for index, brand in enumerate(BRANDS):
        name = brand.capitalize()
        official = _official(reputation, brand)
        other = BRANDS[(index + 1) % len(BRANDS)]
        other_official = _official(reputation, other)
        fake = f"{brand}-secure-verify.top"
        cases.append({
            "name": f"phish_{brand}", "gold": "deceptive",
            "text": f"{name}: there is a problem with your account. Verify your details today or it will be suspended.",
            "url": f"https://{fake}/login",
            "image": render_login_page(name, f"{fake}/login", accent=ACCENTS[brand]),
            "audio_text": None,
        })
        cases.append({
            "name": f"campaign_{brand}", "gold": "deceptive",
            "text": f"Your {name} account will be suspended immediately.",
            "url": None, "image": None,
            "audio_text": f"Hi, it's {name} support. Please read me the code we sent you.",
        })
        cases.append({
            "name": f"conflict_{brand}", "gold": "not_genuine",
            "text": f"Your {name} receipt is ready.",
            "url": f"https://www.{other_official}/account",
            "image": render_login_page(name, f"{official}/account", accent=ACCENTS[brand]),
            "audio_text": None,
        })
        cases.append({
            "name": f"genuine_{brand}", "gold": "genuine",
            "text": f"Your {name} receipt for your recent payment is available.",
            "url": f"https://www.{official}/account",
            "image": render_login_page(name, f"{official}/account", accent=ACCENTS[brand]),
            "audio_text": f"Hi, this is a reminder that your {name} receipt is ready in your account.",
        })
    return cases


def synthesise_voices(cases: list[dict]) -> dict[str, bytes]:
    """Text-to-speech (Windows) for every case that has a voice part; cached on disk."""
    if sys.platform != "win32":
        raise RuntimeError("Spoken test audio is generated with Windows text-to-speech.")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    todo, paths = [], {}
    for index, case in enumerate(c for c in cases if c["audio_text"]):
        path = AUDIO_DIR / f"bench_{case['name']}.wav"
        paths[case["name"]] = path
        if not path.exists():
            todo.append({"file": str(path), "text": case["audio_text"], "voice": VOICES[index % len(VOICES)]})
    if todo:
        manifest, script = AUDIO_DIR / "bench_manifest.json", AUDIO_DIR / "synthesise.ps1"
        manifest.write_text(json.dumps(todo), encoding="utf-8")
        script.write_text(_POWERSHELL, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(manifest)],
            check=True, timeout=600,
        )
    return {name: path.read_bytes() for name, path in paths.items()}

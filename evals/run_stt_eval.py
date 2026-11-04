"""Measure speech-to-text accuracy (word error rate) on your own recordings.

    python evals/run_stt_eval.py                      # recordings in evals/audio/
    python evals/run_stt_eval.py --dir path/to/clips

Put each clip next to its exact transcript: evals/audio/leaky-tap.wav + evals/audio/leaky-tap.txt.
Record real tenant-style sentences in your own accent, unit numbers included ("my sink is leaking in 12B").
Uses SARVAM_API_KEY from .env and the model and language the app uses. Fails when the average WER is
above stt.wer_max in evals/thresholds.json. Clips are git-ignored because they are your voice.
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from app import config  # noqa: E402
from metrics import wer  # noqa: E402

TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".webm": "audio/webm", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "audio"))
    a = ap.parse_args()
    if not config.SARVAM_API_KEY:
        print("Set SARVAM_API_KEY in .env first.")
        return 2
    clips = sorted(p for p in Path(a.dir).glob("*") if p.suffix.lower() in TYPES and p.with_suffix(".txt").exists())
    if not clips:
        print(f"No clips with transcripts found in {a.dir} (see this file's docstring).")
        return 2
    scores = []
    for clip in clips:
        ref = clip.with_suffix(".txt").read_text(encoding="utf-8").strip()
        r = httpx.post("https://api.sarvam.ai/speech-to-text", headers={"api-subscription-key": config.SARVAM_API_KEY},
                       files={"file": (clip.name, clip.read_bytes(), TYPES[clip.suffix.lower()])},
                       data={"language_code": config.SARVAM_STT_LANGUAGE}, timeout=60)
        r.raise_for_status()
        hyp = (r.json().get("transcript") or "").strip()
        scores.append(wer(ref, hyp))
        print(f"{clip.name:<30} WER {scores[-1]:.0%}\n   said: {ref}\n   heard: {hyp}")
    mean = sum(scores) / len(scores)
    limit = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))["stt"]["wer_max"]
    print(f"\nMean WER {mean:.1%} over {len(scores)} clips (limit {limit:.0%})")
    print("PASS" if mean <= limit else "FAIL")
    return 0 if mean <= limit else 1


if __name__ == "__main__":
    sys.exit(main())

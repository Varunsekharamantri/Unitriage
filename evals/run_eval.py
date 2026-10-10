"""Measure how well the vision step triages real photos.

    .venv/Scripts/python.exe evals/run_eval.py                  # first provider in VISION_PROVIDERS
    .venv/Scripts/python.exe evals/run_eval.py --provider groq  # or gemini
    .venv/Scripts/python.exe evals/run_eval.py --delay 25       # wait longer between calls (free tiers)

The folder a photo is in is its label (see evals/images/README.txt):

    evals/images/minor-diy/   minor-not-diy/   major/   emergency/

Optional: photo.jpg + photo.txt (same name) tests a tenant's note with that photo.
evals/cases.json holds extra cases by hand, e.g. the same photo with a misleading note.
Use plenty of MINOR photos: over-escalating everything is exactly what this is here to catch.

    --list    show the cases that would run, without calling the AI
    --limit N run only the first N (each case is one call on your free-tier quota)

Labels are opinions: argue with them by moving a photo to a different folder.
"""
import argparse
import base64
import json
import mimetypes
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.services.imaging import prepare_image  # noqa: E402
from app.services.vision_service import GeminiProvider, GroqProvider, parse_assessment  # noqa: E402

RANK = {"minor": 0, "major": 1, "emergency": 2}

# folder name -> (expected severity, expected is_safe_for_diy)
FOLDERS = {
    "minor-diy": ("minor", True),
    "minor-not-diy": ("minor", False),
    "major": ("major", False),
    "emergency": ("emergency", False),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def folder_key(name: str) -> str:
    """'Minor_NOT_DIY', 'minor not diy' and 'minor-not-diy' are all the same folder."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def discover(images_dir: Path, base: Path) -> list:
    """One case per photo, labelled by its folder; photo.txt next to it is the tenant's note."""
    cases = []
    for folder in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        label = FOLDERS.get(folder_key(folder.name))
        if label is None:
            print(f"  (skipping folder {folder.name!r}: not one of {', '.join(FOLDERS)})")
            continue
        severity, diy = label
        for photo in sorted(folder.glob("*")):
            if photo.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            note_file = photo.with_suffix(".txt")
            cases.append({
                "image": photo.relative_to(base).as_posix(),
                "note": note_file.read_text(encoding="utf-8").strip() if note_file.exists() else "",
                "expect_severity": severity,
                "expect_diy": diy,
                "comment": f"folder: {folder.name}",
            })
    return cases


def make_provider(name: str):
    if name == "gemini" and config.GEMINI_API_KEY:
        return GeminiProvider(config.GEMINI_API_KEY, config.GEMINI_MODEL)
    if name == "groq" and config.GROQ_API_KEY:
        return GroqProvider(config.GROQ_API_KEY, config.GROQ_MODEL)
    sys.exit(f"No usable provider {name!r}: unknown, or its API key is not set in .env")


def run_case(provider, case, base: Path, retries: int, backoff: float):
    path = base / case["image"]
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    # the same shrinking the app does, so big photos are tested the way they would really be sent
    img, mime = prepare_image(base64.b64encode(path.read_bytes()).decode(), mime)
    last = None
    for attempt in range(retries):
        try:
            raw = provider.generate(img, mime, case.get("note", ""))
            return parse_assessment(raw.text, config.CONFIDENCE_THRESHOLD, require_evidence=True)
        except Exception as exc:  # rate limit, overload, or a bad reply: wait and retry
            last = exc
            time.sleep(backoff)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:80]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=config.VISION_PROVIDERS[0])
    ap.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    ap.add_argument("--delay", type=float, default=20, help="seconds between calls")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--list", action="store_true", help="show the cases without calling the AI")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--only", nargs="+", default=[], metavar="TEXT",
                    help="run only cases whose photo path contains any of these words")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    base = cases_path.parent
    cases = discover(base / "images", base)
    if cases_path.exists():
        cases += json.loads(cases_path.read_text(encoding="utf-8-sig"))  # -sig: Notepad adds a BOM
    missing = [c["image"] for c in cases if not (base / c["image"]).exists()]
    for image in missing:
        print(f"  (skipping a cases.json entry: photo not found: {image})")
    cases = [c for c in cases if c["image"] not in missing]
    if args.only:
        words = [w.lower() for w in args.only]
        cases = [c for c in cases if any(w in c["image"].lower() for w in words)]
    if args.limit:
        cases = cases[: args.limit]

    if args.list:
        by_label = Counter((c["expect_severity"], c["expect_diy"]) for c in cases)
        for c in cases:
            note = f"  note: {c['note'][:40]!r}" if c.get("note") else ""
            print(f"  {c['expect_severity']:9} diy={c['expect_diy']!s:5}  {c['image']}{note}")
        print(f"\n{len(cases)} cases:", ", ".join(f"{s}/diy={d} x{n}" for (s, d), n in sorted(by_label.items())))
        return 0

    provider = make_provider(args.provider)
    print(f"{provider.name} / {provider.model}: {len(cases)} cases\n")

    sev_ok = diy_ok = over = under = done = 0
    confusion = Counter()
    for i, case in enumerate(cases):
        label = f"{Path(case['image']).stem}" + (f" + note {case['note'][:28]!r}" if case.get("note") else "")
        try:
            a = run_case(provider, case, cases_path.parent, args.retries, backoff=args.delay)
        except RuntimeError as exc:
            print(f"  ??  {label}: {exc}")
            continue
        done += 1
        want, got = case["expect_severity"], a.severity
        confusion[(want, got)] += 1
        sev_ok += got == want
        diy_ok += a.is_safe_for_diy == case["expect_diy"]
        over += RANK[got] > RANK[want]
        under += RANK[got] < RANK[want]
        mark = "ok" if got == want else ("UP" if RANK[got] > RANK[want] else "DN")
        print(f"  {mark:3} {label}\n      expected {want}/diy={case['expect_diy']}  got {got}/diy={a.is_safe_for_diy}"
              f"  conf {a.confidence:.2f}{'  AMBIGUOUS' if a.ambiguous else ''}  ({a.reasons[0]})")
        e = a.evidence
        print(f"      saw: water={e['water']}, damage={e['damage']}, simple_fix={e['simple_fix']}, "
              f"hazards={[k for k in ('electrical_hazard', 'gas_or_fire', 'structural_risk', 'sewage') if e[k]] or 'none'}")
        if i < len(cases) - 1:
            time.sleep(args.delay)

    if not done:
        print("\nNo case completed. Likely rate-limited: wait a minute, or use --delay 40.")
        return 1
    print(f"\nSeverity correct: {sev_ok}/{done}   DIY correct: {diy_ok}/{done}")
    print(f"Over-escalated (UP): {over}   Under-escalated (DN, the dangerous way): {under}")
    print("Expected -> got:", ", ".join(f"{w}->{g} x{n}" for (w, g), n in sorted(confusion.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

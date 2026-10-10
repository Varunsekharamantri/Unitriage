"""Measure how well the tenant-note screener spots problems, and pick its threshold.

    .venv/Scripts/python.exe evals/run_note_eval.py                  # keywords and the AI check
    .venv/Scripts/python.exe evals/run_note_eval.py --only keywords  # no AI calls
    .venv/Scripts/python.exe evals/run_note_eval.py --only llm --delay 3

evals/notes.json lists notes and the problems each really reports (gas, no_water, ...). Any flag
makes a ticket urgent or an emergency, so what matters is:
    caught        a note reporting a problem raised at least one flag (a miss is the bad mistake)
    false alarm   a harmless note raised a flag (wasteful: it sends a technician urgently)
    exact         the flags were exactly the right ones (matters for which advice is shown)
    vulnerable    notes marked "vulnerable": true were recognised as such, and others were not
Borderline notes are listed but not scored.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.services.note_screen import (  # noqa: E402
    HAZARDS,
    KeywordNoteScreener,
    LLMNoteScreener,
    build_llm_callers,
    flags_for,
)

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


def score(rows, threshold):
    scored = [r for r in rows if not r["case"].get("borderline")]
    danger = [r for r in scored if r["case"]["expect"]]
    harmless = [r for r in scored if not r["case"]["expect"]]
    flags = lambda r: set(flags_for(r["probs"], threshold))  # noqa: E731
    caught = sum(bool(flags(r)) for r in danger)
    false_alarms = sum(bool(flags(r)) for r in harmless)
    exact = sum(flags(r) == set(r["case"]["expect"]) for r in danger)
    return len(danger), caught, len(harmless), false_alarms, exact


def vulnerable_score(rows, threshold):
    scored = [r for r in rows if not r["case"].get("borderline")]
    want = [r for r in scored if r["case"].get("vulnerable")]
    found = sum(r["probs"]["vulnerable"] >= threshold for r in want)
    wrong = sum(r["probs"]["vulnerable"] >= threshold for r in scored if not r["case"].get("vulnerable"))
    return len(want), found, len(scored) - len(want), wrong


def run(name, screener, cases, delay=0.0):
    print(f"\n=== {name} ===")
    rows, t, failures = [], time.time(), 0
    for case in cases:
        try:
            probs = screener.screen(case["note"])
        except Exception as exc:  # e.g. a rate limit: record it, don't let one note end the run
            failures += 1
            print(f"  (could not screen {case['note'][:40]!r}: {exc})", flush=True)
            probs = {k: 0.0 for k in HAZARDS}
        rows.append({"case": case, "probs": probs})
        time.sleep(delay)
    if failures:
        print(f"WARNING: {failures} notes could not be screened and count as 'no problem' below")
    print(f"{len(cases)} notes in {time.time() - t:.1f}s\n")

    print("threshold | reported caught | harmless false alarms | exact flags")
    for th in THRESHOLDS:
        n_d, caught, n_h, fa, exact = score(rows, th)
        mark = "  <- current setting" if abs(th - config.NOTE_HAZARD_THRESHOLD) < 1e-9 else ""
        print(f"   {th:.1f}    |   {caught:2}/{n_d:<2}        |   {fa:2}/{n_h:<2}               |  {exact:2}/{n_d:<2}{mark}")
    th = config.NOTE_HAZARD_THRESHOLD
    n_v, found, n_o, wrong = vulnerable_score(rows, th)
    print(f"\nvulnerable person recognised: {found}/{n_v}; wrongly assumed on other notes: {wrong}/{n_o}")

    for r in rows:
        if (r["probs"]["vulnerable"] >= th) != bool(r["case"].get("vulnerable")) and not r["case"].get("borderline"):
            what = "NOT recognised as vulnerable" if r["case"].get("vulnerable") else "wrongly assumed vulnerable"
            print(f"  {what}: {r['case']['note'][:70]!r}")

    print(f"\nat the current threshold ({th}):")
    for r in rows:
        got, want = flags_for(r["probs"], th), r["case"]["expect"]
        tag = " (borderline)" if r["case"].get("borderline") else ""
        if set(got) != set(want):
            kind = "MISSED " if not got else ("FALSE ALARM" if not want else "WRONG FLAG")
            top = ", ".join(f"{k}={r['probs'][k]:.2f}" for k in sorted(r["probs"], key=r["probs"].get, reverse=True)[:2])
            print(f"  {kind:11}{tag} {r['case']['note'][:70]!r}  wanted={want} got={got}  [{top}]")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["keywords", "llm"])
    ap.add_argument("--notes", default=str(Path(__file__).with_name("notes.json")))
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between AI calls (free-tier limits)")
    args = ap.parse_args()
    cases = json.loads(Path(args.notes).read_text(encoding="utf-8-sig"))  # -sig: Notepad adds a BOM
    assert all(set(c["expect"]) <= set(HAZARDS) for c in cases), "notes.json uses an unknown problem name"
    scored = [c for c in cases if not c.get("borderline")]
    print(f"{len(cases)} notes ({len(scored)} scored: {sum(bool(c['expect']) for c in scored)} reporting a problem, "
          f"{sum(not c['expect'] for c in scored)} harmless; {len(cases) - len(scored)} borderline)")

    results = {}
    if args.only in (None, "keywords"):
        results["keywords"] = run("keywords (no AI)", KeywordNoteScreener(), cases)
    if args.only in (None, "llm"):
        callers = build_llm_callers(config.NOTE_LLM_PROVIDERS)
        if not callers:
            sys.exit("No API key for " + ", ".join(config.NOTE_LLM_PROVIDERS) + " in .env")
        print(f"\n(AI providers, in order: {', '.join(n for n, _ in callers)})")
        results["llm"] = run("AI text check (llm)", LLMNoteScreener(callers), cases, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

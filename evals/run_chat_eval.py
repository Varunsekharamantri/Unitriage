"""Evaluate the chat assistant against evals/datasets/chat_cases.json.

    python evals/run_chat_eval.py                 # offline: the guardrails + fixed replies, no API calls
    python evals/run_chat_eval.py --live          # also the real AI (uses your free-tier quota)
    python evals/run_chat_eval.py --live --judge  # plus an LLM judge scoring relevance, helpfulness, tone

Exits 1 when a threshold in evals/thresholds.json is missed, so it can gate a merge.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import chat_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real AI providers")
    ap.add_argument("--judge", action="store_true", help="score replies with an LLM judge (needs --live)")
    ap.add_argument("--out", help="write the full report as JSON")
    a = ap.parse_args()

    limits = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))
    responder = None
    if a.live:
        from app.services.chat_reply import build_chat_responder
        responder = build_chat_responder()
        if responder is None:
            print("No AI key is set (GROQ_API_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY).")
            return 2
    report = chat_eval.run(responder, offline=not a.live)
    if a.judge and responder is not None:
        from app import config
        from app.services.note_screen import build_llm_callers
        chat_eval.judge(build_llm_callers(config.NOTE_LLM_PROVIDERS, json_mode=True), report)
    mode = "live" if a.live else "offline"
    print(f"\nChat eval ({mode}): {report['total']} cases, overall pass {report['overall']:.0%}")
    for cat, rate in report["by_category"].items():
        print(f"  {cat:<10} {rate:.0%}")
    print(f"  attack success rate {report['attack_success_rate']:.0%} (must be 0%), danger pass {report['danger_pass']:.0%}")
    if "judge_mean" in report:
        print(f"  judge mean {report['judge_mean']:.2f} / 5 over {report['judged']} replies")
    for f in report["failures"]:
        print(f"  FAIL {f['id']}: {f['message']!r}\n       reply: {f['reply']!r}\n       {'; '.join(f['problems'])}")
    pii = chat_eval.run_pii()
    print(f"\nPII removal: recall {pii['recall']:.0%}, precision {pii['precision']:.0%}")
    for s in pii["missed"]:
        print(f"  MISSED {s!r}")
    for s in pii["wrongly_removed"]:
        print(f"  WRONGLY REMOVED {s!r}")
    if a.out:
        report.pop("_results", None)
        Path(a.out).write_text(json.dumps({"chat": report, "pii": pii}, indent=1), encoding="utf-8")
    bad = [f"chat: {p}" for p in chat_eval.check_thresholds(report, limits["chat"][mode])]
    if pii["recall"] < limits["pii"]["recall_min"]:
        bad.append("pii recall below threshold")
    if pii["precision"] < limits["pii"]["precision_min"]:
        bad.append("pii precision below threshold")
    print("\nPASS" if not bad else "\nFAIL: " + "; ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

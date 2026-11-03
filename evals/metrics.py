"""Word error rate, the standard measure for speech to text (lower is better; 0 is perfect)."""
import re


def _words(text: str) -> list:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    r, h = _words(reference), _words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)

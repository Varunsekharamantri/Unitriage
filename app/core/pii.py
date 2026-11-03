"""Strips personal details from free text before it is stored or sent to an AI.

Tenants type (or say) things like "call me on 98765 43210" or "my name is Priya". Those details are
not needed to fix a leak, so they are replaced with a tag before the message reaches a record. This
is pattern based: it catches contact details and ID numbers reliably, and names only when they are
introduced as one ("my name is ..."). Unit numbers such as 12B are kept.
"""
import re

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"\b[A-Za-z0-9._-]{2,}@[a-z]{2,}\b"), "[upi]"),                  # UPI handle: name@bank
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[id]"),                              # PAN
    (re.compile(r"(?<![\w])\+?\d[\d\s().-]{5,}\d(?![\w])"), None),                # phone / card / Aadhaar: decided below
    (re.compile(r"\b(?:my name is|i am called|i'm called|this is|name's)\s+[A-Za-z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)?", re.I), "[name]"),
]


def _digits(m: re.Match) -> str:
    # 7 or more digits (spaces and dashes allowed) is a phone, card or ID number, never a unit number.
    return "[number]" if sum(c.isdigit() for c in m.group(0)) >= 7 else m.group(0)


def redact(text: str) -> str:
    out = text or ""
    for pattern, tag in _PATTERNS:
        out = pattern.sub(_digits if tag is None else tag, out)
    return out
